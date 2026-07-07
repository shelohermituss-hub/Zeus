"""
Lanceur du bot P11-V4.

Usage
-----
    # Paper trading (défaut, aucune connexion broker requise pour démarrer)
    python -m zeus.live.run_p11

    # Config alternative + surcharges rapides
    python -m zeus.live.run_p11 --config config/p11_v4.yaml --risk 0.005
    python -m zeus.live.run_p11 --pairs XAUUSD,EURUSD,CADJPY

    # Live MT5 (Windows + terminal MT5 + variables ZEUS_MT5_*)
    python -m zeus.live.run_p11 --mode live
    python -m zeus.live.run_p11 --mode live --dry-run    # tout sauf l'envoi réel

Garde-fous au démarrage :
    - mode live exigé À LA FOIS dans le YAML et en CLI (double opt-in) ;
    - le paper trading doit être validé avant tout live (règle du projet) ;
    - risque CLI borné à ]0, 0.0075] (maximum validé par backtest).
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from zeus.live.p11_config import ConfigError, load_p11_config
from zeus.live.p11_engine import P11Engine
from zeus.live.order_router import MT5Router, PaperRouter
from zeus.utils.logger import logger

_DEFAULT_CONFIG = Path(__file__).parent.parent.parent / "config" / "p11_v4.yaml"
_MAX_VALIDATED_RISK = 0.0075


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Bot propfirm P11-V4 (paper/live MT5)")
    p.add_argument("--config", default=str(_DEFAULT_CONFIG),
                   help="chemin du YAML (défaut: config/p11_v4.yaml)")
    p.add_argument("--mode", choices=["paper", "live"], default=None,
                   help="surcharge le mode du YAML (live exige les deux)")
    p.add_argument("--risk", type=float, default=None,
                   help="surcharge risk_per_trade_pct (ex. 0.005)")
    p.add_argument("--pairs", default=None,
                   help="liste de symboles à activer, ex. XAUUSD,EURUSD "
                        "(les autres sont désactivés)")
    p.add_argument("--dry-run", action="store_true",
                   help="construit les ordres sans jamais les envoyer")
    p.add_argument("--max-ticks", type=int, default=None,
                   help="arrête après N ticks (tests/smoke)")
    return p.parse_args(argv)


def build_engine(args, connector=None) -> P11Engine:
    cfg = load_p11_config(args.config)

    # ── Surcharges CLI ────────────────────────────────────────────────────────
    if args.risk is not None:
        if not 0 < args.risk <= _MAX_VALIDATED_RISK:
            raise ConfigError(
                f"--risk {args.risk} hors bornes validées ]0, {_MAX_VALIDATED_RISK}]"
            )
        propfirm = dataclasses.replace(cfg.propfirm, risk_per_trade_pct=args.risk)
        cfg = dataclasses.replace(cfg, propfirm=propfirm)

    if args.pairs is not None:
        wanted = {s.strip().upper() for s in args.pairs.split(",") if s.strip()}
        unknown = wanted - set(cfg.symbols)
        if unknown:
            raise ConfigError(f"--pairs : symboles inconnus/désactivés {sorted(unknown)}")
        symbols = {k: v for k, v in cfg.symbols.items() if k in wanted}
        cfg = dataclasses.replace(cfg, symbols=symbols)

    # ── Mode : double opt-in pour le live ─────────────────────────────────────
    mode = args.mode or cfg.mode
    if mode == "live" and (cfg.mode != "live" or args.mode != "live"):
        raise ConfigError(
            "mode live exige mode: live dans le YAML ET --mode live en CLI "
            "(double confirmation) — et un paper trading validé au préalable"
        )
    cfg = dataclasses.replace(cfg, mode=mode)

    dry_run = args.dry_run or cfg.execution.dry_run

    # ── Connecteur + routeur ──────────────────────────────────────────────────
    if connector is None:
        try:
            from zeus.exchange.mt5 import MT5Connector   # import tardif (Windows)
            from zeus.config import get_settings
            s = get_settings()
            connector = MT5Connector(
                login=s.mt5_login, password=s.mt5_password, server=s.mt5_server,
            )
        except RuntimeError as e:
            raise ConfigError(
                f"connecteur MT5 indisponible sur cette machine : {e} — "
                "le bot (paper comme live) doit tourner sous Windows avec "
                "un terminal MT5 et le package Python MetaTrader5"
            ) from e

    if mode == "live" or dry_run:
        router = MT5Router(
            magic_number=cfg.execution.magic_number,
            max_slippage_points=cfg.execution.max_slippage_points,
            dry_run=dry_run,
        )
    else:
        router = PaperRouter()

    notifier = None
    if cfg.telegram_enabled:
        from zeus.monitoring.telegram import TelegramNotifier
        import os
        notifier = TelegramNotifier(
            bot_token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            chat_id=os.environ.get("TELEGRAM_CHAT_ID", ""),
        )

    return P11Engine(cfg, connector, notifier=notifier, router=router)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        engine = build_engine(args)
    except ConfigError as e:
        logger.error("Configuration refusée", error=str(e))
        print(f"ERREUR CONFIG : {e}", file=sys.stderr)
        return 2

    logger.info(
        "P11 bot prêt",
        mode=engine.config.mode,
        symbols=",".join(engine.config.enabled_symbols),
        risk_pct=engine.config.propfirm.risk_per_trade_pct,
        risk_usd=engine.risk_usd,
        dry_run=args.dry_run,
    )
    engine.run(max_ticks=args.max_ticks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
