"""
Bullish/Bearish Engulfing + RSI — port fidèle de la logique de
"Free Robots/BullishBearish Engulfing RSI.mq5" (robot gratuit MetaQuotes
livré avec MT5), envoyé par l'utilisateur pour évaluation.

Logique EXACTE de la source MQL5 (voir CheckPattern/CheckConfirmation/
CheckCloseSignal) — indices MQL5 1/2 (dernière bougie close / précédente)
traduits en indices chronologiques i / i-1 :

Engulfing baissier (SELL) — bougie i-1 haussière, bougie i baissière :
    open[i-1] < close[i-1]                       (bougie i-1 haussière)
    open[i] - close[i] > avg_body(i)              (corps i > corps moyen 12 barres)
    close[i] < open[i-1]                          (englobe vers le bas)
    mid_oc[i-1] > sma5[i-1]                       (contexte de tendance haussière)
    open[i] > close[i-1]                          (ouverture i au-dessus de close i-1)
    confirmation : RSI(i) > 60

Engulfing haussier (BUY) — miroir exact, RSI(i) < 40.

Sortie : SL/TP fixes en points (bracket), OU expiration après N barres,
OU croisement RSI (sortie anticipée si le momentum qui a confirmé
l'entrée s'essouffle — croisement à la baisse de 70 ou de 30 pour un
long, à la hausse de 30 ou de 70 pour un short). Les trois conditions
sont évaluées dans le même ordre que l'EA d'origine à chaque barre après
l'entrée ; SL/TP prioritaire (résolution pessimiste, comme le reste du
projet).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_EPS = 1e-8


@dataclass(frozen=True)
class EngulfingRsiSignal:
    direction:   str            # "long" | "short"
    bar_index:   int             # index de la bougie de pattern (i)
    formed_at:   pd.Timestamp
    entry_ref:   float           # close[i] — référence seulement, le fill réel = open[i+1]


def _sma(values: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(values).rolling(period, min_periods=period).mean().to_numpy()


def _rsi(closes: np.ndarray, period: int) -> np.ndarray:
    """RSI Wilder via lissage EWM (alpha=1/period) — même formule que le
    filtre RSI optionnel de sd_strategy.py, pour cohérence dans le dépôt."""
    delta  = np.diff(closes, prepend=closes[0])
    gains  = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    alpha  = 1.0 / period
    g_ema  = pd.Series(gains).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    l_ema  = pd.Series(losses).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(l_ema > 0, g_ema / l_ema, 100.0)
    return 100.0 - 100.0 / (1.0 + rs)


def _avg_body(opens: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    body = np.abs(opens - closes)
    return pd.Series(body).rolling(period, min_periods=period).mean().to_numpy()


class EngulfingRsiStrategy:
    """
    Détecte le pattern Engulfing + confirmation RSI sur une série OHLCV.

    Paramètres — mêmes défauts que la source MQL5 :
        avg_body_period : fenêtre de calcul du corps moyen (12)
        ma_period        : période de la SMA de tendance (5)
        rsi_period        : période RSI (37)
        rsi_buy_max       : confirmation long si RSI < ce seuil (40)
        rsi_sell_min      : confirmation short si RSI > ce seuil (60)
    """

    def __init__(
        self,
        avg_body_period: int   = 12,
        ma_period:       int   = 5,
        rsi_period:      int   = 37,
        rsi_buy_max:     float = 40.0,
        rsi_sell_min:    float = 60.0,
    ) -> None:
        self.avg_body_period = avg_body_period
        self.ma_period        = ma_period
        self.rsi_period       = rsi_period
        self.rsi_buy_max      = rsi_buy_max
        self.rsi_sell_min     = rsi_sell_min

    def run(self, df: pd.DataFrame) -> list[EngulfingRsiSignal]:
        n = len(df)
        min_needed = max(self.avg_body_period, self.ma_period, self.rsi_period) + 2
        if n < min_needed:
            return []

        opens  = df["open"].to_numpy(dtype=float)
        highs  = df["high"].to_numpy(dtype=float)
        lows   = df["low"].to_numpy(dtype=float)
        closes = df["close"].to_numpy(dtype=float)

        avg_body = _avg_body(opens, closes, self.avg_body_period)
        sma      = _sma(closes, self.ma_period)
        rsi      = _rsi(closes, self.rsi_period)
        mid_oc   = (opens + closes) / 2.0

        signals: list[EngulfingRsiSignal] = []
        start = min_needed - 1
        for i in range(start, n):
            p = i - 1
            if np.isnan(avg_body[i]) or np.isnan(sma[p]):
                continue

            # Bearish engulfing → short
            if (opens[p] < closes[p]
                    and (opens[i] - closes[i]) > avg_body[i]
                    and closes[i] < opens[p]
                    and mid_oc[p] > sma[p]
                    and opens[i] > closes[p]):
                if rsi[i] > self.rsi_sell_min:
                    signals.append(EngulfingRsiSignal(
                        direction="short", bar_index=i,
                        formed_at=df.index[i], entry_ref=closes[i],
                    ))
                continue

            # Bullish engulfing → long
            if (opens[p] > closes[p]
                    and (closes[i] - opens[i]) > avg_body[i]
                    and closes[i] > opens[p]
                    and mid_oc[p] < sma[p]
                    and opens[i] < closes[p]):
                if rsi[i] < self.rsi_buy_max:
                    signals.append(EngulfingRsiSignal(
                        direction="long", bar_index=i,
                        formed_at=df.index[i], entry_ref=closes[i],
                    ))

        return signals


@dataclass(frozen=True)
class EngulfingTradeResult:
    direction:   str
    entry_bar:   int
    exit_bar:    int
    entry_price: float
    exit_price:  float
    outcome:     str    # "win" | "loss" | "scratch" | "timeout"
    pnl_usd:     float
    bars_held:   int


def simulate_engulfing_rsi(
    signals:        list[EngulfingRsiSignal],
    df:              pd.DataFrame,
    sl_points:       float,
    tp_points:       float,
    point_size:      float,
    duration_bars:   int,
    lot:             float,
    contract_size:   float,
    spread_points:   float = 0.0,
    rsi_period:      int   = 37,
) -> list[EngulfingTradeResult]:
    """
    Rejoue chaque signal indépendamment (comme l'EA source : une position à
    la fois par sens, fermée avant qu'une nouvelle s'ouvre dans les faits —
    ici on simule chaque signal isolément, cohérent avec la validation brute
    de l'étape 2 du plan momentum scalp).

    Ordre de résolution par barre (identique à l'EA : le SL/TP broker est
    prioritaire car exécuté en continu, la sortie sur croisement RSI et
    l'expiration par durée ne sont vérifiées qu'ensuite) :
        1. SL/TP (résolution pessimiste : SL d'abord si les deux sont
           atteints sur la même barre)
        2. Croisement RSI (sortie anticipée)
        3. Expiration après duration_bars barres
    """
    closes = df["close"].to_numpy(dtype=float)
    highs  = df["high"].to_numpy(dtype=float)
    lows   = df["low"].to_numpy(dtype=float)
    opens  = df["open"].to_numpy(dtype=float)
    rsi    = _rsi(closes, rsi_period)
    n      = len(df)

    results: list[EngulfingTradeResult] = []
    spread_price = spread_points * point_size

    for sig in signals:
        entry_bar = sig.bar_index + 1
        if entry_bar >= n:
            continue
        is_long = sig.direction == "long"
        raw_open = opens[entry_bar]
        entry = raw_open + spread_price if is_long else raw_open - spread_price
        sl = entry - sl_points * point_size if is_long else entry + sl_points * point_size
        tp = entry + tp_points * point_size if is_long else entry - tp_points * point_size

        for bar_idx in range(entry_bar, n):
            hi, lo = highs[bar_idx], lows[bar_idx]

            sl_hit = (lo <= sl) if is_long else (hi >= sl)
            tp_hit = (hi >= tp) if is_long else (lo <= tp)
            if sl_hit:
                pnl_usd = -(sl_points * point_size) * lot * contract_size
                results.append(EngulfingTradeResult(
                    sig.direction, entry_bar, bar_idx, entry, sl, "loss",
                    pnl_usd, bar_idx - entry_bar,
                ))
                break
            if tp_hit:
                pnl_usd = (tp_points * point_size) * lot * contract_size
                results.append(EngulfingTradeResult(
                    sig.direction, entry_bar, bar_idx, entry, tp, "win",
                    pnl_usd, bar_idx - entry_bar,
                ))
                break

            if bar_idx > entry_bar:
                r_now, r_prev = rsi[bar_idx], rsi[bar_idx - 1]
                rsi_close = False
                if is_long and ((r_now < 70 and r_prev > 70) or (r_now < 30 and r_prev > 30)):
                    rsi_close = True
                if not is_long and ((r_now > 30 and r_prev < 30) or (r_now > 70 and r_prev < 70)):
                    rsi_close = True
                if rsi_close:
                    exit_px = closes[bar_idx]
                    move = (exit_px - entry) if is_long else (entry - exit_px)
                    pnl_usd = move * lot * contract_size
                    outcome = "win" if pnl_usd > 0 else ("loss" if pnl_usd < 0 else "scratch")
                    results.append(EngulfingTradeResult(
                        sig.direction, entry_bar, bar_idx, entry, exit_px, outcome,
                        pnl_usd, bar_idx - entry_bar,
                    ))
                    break

            if bar_idx - entry_bar >= duration_bars:
                exit_px = closes[bar_idx]
                move = (exit_px - entry) if is_long else (entry - exit_px)
                pnl_usd = move * lot * contract_size
                outcome = "win" if pnl_usd > 0 else ("loss" if pnl_usd < 0 else "scratch")
                results.append(EngulfingTradeResult(
                    sig.direction, entry_bar, bar_idx, entry, exit_px, outcome,
                    pnl_usd, bar_idx - entry_bar,
                ))
                break
        # sinon : expire en fin de données, exclu (comme sd_simulation)

    return results
