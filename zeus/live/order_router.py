"""
Routage des ordres du bot P11 — paper (simulation interne) ou MT5 réel.

Règles non négociables :
    - JAMAIS d'ordre marché sans stop-loss attaché à la requête.
    - Dimensionnement par tick-value MT5 : lots = risque$ / (ticks_SL × valeur_tick)
      → correct pour toutes les devises de cotation (USD, JPY, CHF, EUR…).
    - Tout rejet broker → le trade n'existe pas côté moteur (fail closed).
    - dry_run : la requête est construite et loggée mais jamais envoyée.

MT5Router exige un terminal MetaTrader 5 sous Windows avec le package
Python `MetaTrader5`.  Sur Linux/CI, l'import échoue silencieusement et
l'instanciation lève RuntimeError — PaperRouter reste utilisable partout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from zeus.utils.logger import logger

try:                                     # Windows uniquement
    import MetaTrader5 as mt5            # type: ignore[import]
    _MT5_AVAILABLE = True
except ImportError:                      # Linux/CI : PaperRouter seulement
    mt5 = None                           # type: ignore[assignment]
    _MT5_AVAILABLE = False


@dataclass(frozen=True)
class Fill:
    """Résultat d'une ouverture de position."""
    price:  float
    lots:   float
    ticket: int          # ticket de position MT5 (0 en paper)


class OrderRouter(Protocol):
    def open_position(self, symbol: str, direction: str,
                      sl_price: float, risk_usd: float,
                      ref_price: float) -> Optional[Fill]: ...
    def partial_close(self, symbol: str, ticket: int, direction: str,
                      lots: float) -> bool: ...
    def modify_sl(self, symbol: str, ticket: int, sl_price: float) -> bool: ...
    def close_position(self, symbol: str, ticket: int, direction: str,
                       lots: float) -> bool: ...


# ── Paper ─────────────────────────────────────────────────────────────────────

class PaperRouter:
    """Simulation : aucun ordre externe, fills au prix de référence fourni.

    La taille en lots est calculée avec la même formule que le réel quand
    les métadonnées symbole sont fournies, sinon 1.0 lot symbolique — le
    moteur paper comptabilise en R, la taille n'affecte pas ses métriques.
    """

    def open_position(self, symbol, direction, sl_price, risk_usd, ref_price):
        return Fill(price=ref_price, lots=1.0, ticket=0)

    def partial_close(self, symbol, ticket, direction, lots):
        return True

    def modify_sl(self, symbol, ticket, sl_price):
        return True

    def close_position(self, symbol, ticket, direction, lots):
        return True


# ── MT5 réel ──────────────────────────────────────────────────────────────────

class MT5Router:
    """Ordres réels via MetaTrader 5 — SL attaché, magic number, partiels."""

    def __init__(self, magic_number: int, max_slippage_points: int = 20,
                 dry_run: bool = False) -> None:
        if not _MT5_AVAILABLE and not dry_run:
            raise RuntimeError(
                "MetaTrader5 indisponible — MT5Router exige Windows + terminal MT5 "
                "(utilise mode paper ou dry_run sur cette machine)"
            )
        self.magic = magic_number
        self.deviation = max_slippage_points
        self.dry_run = dry_run

    # — dimensionnement —

    def _lots_for_risk(self, symbol: str, sl_dist: float, risk_usd: float) -> float:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbole {symbol!r} introuvable dans MT5")
        if info.trade_tick_size <= 0 or info.trade_tick_value <= 0:
            raise RuntimeError(f"{symbol}: tick_size/tick_value invalides")
        ticks = sl_dist / info.trade_tick_size
        loss_per_lot = ticks * info.trade_tick_value    # devise du compte
        if loss_per_lot <= 0:
            raise RuntimeError(f"{symbol}: perte/lot nulle — SL trop proche")
        lots = risk_usd / loss_per_lot
        step = info.volume_step or 0.01
        lots = max(info.volume_min, min(info.volume_max, round(lots / step) * step))
        return round(lots, 8)

    # — API —

    def open_position(self, symbol, direction, sl_price, risk_usd, ref_price):
        tick = mt5.symbol_info_tick(symbol) if not self.dry_run else None
        if not self.dry_run and tick is None:
            raise RuntimeError(f"pas de tick pour {symbol!r}")

        is_long = direction == "long"
        price = (tick.ask if is_long else tick.bid) if tick else ref_price
        sl_dist = abs(price - sl_price)
        if sl_dist <= 0:
            logger.warning("SL nul — ordre refusé", symbol=symbol)
            return None
        # SL du mauvais côté du prix réel → refus
        if is_long != (sl_price < price):
            logger.warning("SL incohérent avec le prix broker — ordre refusé",
                           symbol=symbol, price=price, sl=sl_price)
            return None

        if self.dry_run:
            logger.info("[DRY-RUN] open", symbol=symbol, direction=direction,
                        price=price, sl=sl_price, risk_usd=risk_usd)
            return Fill(price=price, lots=0.0, ticket=0)

        lots = self._lots_for_risk(symbol, sl_dist, risk_usd)
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "volume":       lots,
            "type":         mt5.ORDER_TYPE_BUY if is_long else mt5.ORDER_TYPE_SELL,
            "price":        price,
            "sl":           sl_price,                  # SL ATTACHÉ — non négociable
            "deviation":    self.deviation,
            "magic":        self.magic,
            "comment":      "zeus-p11",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            rc = result.retcode if result else "None"
            cm = result.comment if result else "no response"
            logger.error("Ordre rejeté", symbol=symbol, retcode=rc, comment=cm)
            return None

        # Retrouver le ticket de position (ordres marché → position du même ordre)
        ticket = getattr(result, "order", 0)
        positions = mt5.positions_get(symbol=symbol) or []
        for p in positions:
            if p.magic == self.magic and p.ticket == getattr(result, "order", -1):
                ticket = p.ticket
                break
        else:
            if positions:
                mine = [p for p in positions if p.magic == self.magic]
                if mine:
                    ticket = max(mine, key=lambda p: p.time).ticket

        logger.info("Position ouverte", symbol=symbol, direction=direction,
                    lots=result.volume, price=result.price, sl=sl_price,
                    ticket=ticket)
        return Fill(price=float(result.price), lots=float(result.volume),
                    ticket=int(ticket))

    def partial_close(self, symbol, ticket, direction, lots):
        if self.dry_run:
            logger.info("[DRY-RUN] partial_close", symbol=symbol,
                        ticket=ticket, lots=lots)
            return True
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error("partial_close: pas de tick", symbol=symbol)
            return False
        is_long = direction == "long"
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       symbol,
            "position":     ticket,
            "volume":       round(lots, 8),
            "type":         mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY,
            "price":        tick.bid if is_long else tick.ask,
            "deviation":    self.deviation,
            "magic":        self.magic,
            "comment":      "zeus-p11-tp",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if not ok:
            logger.error("partial_close rejeté", symbol=symbol, ticket=ticket,
                         retcode=result.retcode if result else "None")
        return ok

    def modify_sl(self, symbol, ticket, sl_price):
        if self.dry_run:
            logger.info("[DRY-RUN] modify_sl", symbol=symbol,
                        ticket=ticket, sl=sl_price)
            return True
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "symbol":   symbol,
            "position": ticket,
            "sl":       sl_price,
        }
        result = mt5.order_send(request)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if not ok:
            logger.error("modify_sl rejeté", symbol=symbol, ticket=ticket,
                         retcode=result.retcode if result else "None")
        return ok

    def close_position(self, symbol, ticket, direction, lots):
        return self.partial_close(symbol, ticket, direction, lots)
