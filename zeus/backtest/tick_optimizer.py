"""
Tick-Level Signal Optimizer
============================
Affine les signaux SDSignal (entrée et SL) en utilisant les données ticks
brutes exportées depuis MT5, pour améliorer le ratio Risque/Rendement.

Deux raffinements appliqués
----------------------------
1. **Entrée tick** (refine_entry)
   Au lieu de se filler à l'open de la barre M1 suivante, on cherche le
   premier tick *après* signal.formed_at où le prix croise signal.entry_price.
   Sur XAUUSD, cela donne typiquement un fill 2-5 USD/oz plus proche du
   mss_close → SL-distance réduite → position plus grosse → plus de $ par R.

2. **SL tick** (refine_sl)
   On examine les ticks bid dans la fenêtre de la Spring bar (Wyckoff).
   Un percentile filtré (défaut: 5e percentile) élimine les spikes ponctuels
   et retourne un SL plus haut (pour les longs), donc plus court, donc un
   meilleur RR.  On garde toujours le SL le plus serré entre version tick et
   version M1.

Usage typique
-------------
    from zeus.backtest.tick_loader import parse_mt5_ticks, resample_ticks
    from zeus.backtest.tick_optimizer import optimize_signals_with_ticks
    from zeus.backtest.sd_simulation import simulate_all

    tick_df   = parse_mt5_ticks("data/ticks/xauusd/XAUUSD_2024_ticks.csv")
    ohlcv_1s  = resample_ticks(tick_df, "1s")

    refined = optimize_signals_with_ticks(signals, tick_df, m1_df)
    results, n_exp = simulate_all(
        refined, m1_df,
        tick_df          = ohlcv_1s,   # SL/TP scanning à la seconde
        use_signal_entry = True,       # utilise refined.entry_price comme fill
        spread           = 0.0,        # spread déjà inclus dans le prix tick
        **other_kwargs,
    )
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# ── entrée tick ───────────────────────────────────────────────────────────────

def find_tick_entry(
    signal:           Any,          # SDSignal (ou tout Tradeable)
    tick_df:          pd.DataFrame, # colonnes bid, ask, mid
    max_wait_seconds: int = 300,
) -> tuple[float, pd.Timestamp] | None:
    """
    Cherche le premier tick après signal.formed_at où le prix franchit
    signal.entry_price dans la bonne direction.

    - Long  : premier tick où ask >= entry_price
    - Short : premier tick où bid <= entry_price

    Returns
    -------
    (fill_price, fill_timestamp) ou None si aucun tick trouvé dans la fenêtre.
    """
    entry_ts = signal.formed_at
    end_ts   = entry_ts + pd.Timedelta(seconds=max_wait_seconds)

    try:
        window = tick_df.loc[entry_ts:end_ts]
    except KeyError:
        return None

    if window.empty:
        return None

    if signal.direction == "long":
        mask = window["ask"] >= signal.entry_price
        col  = "ask"
    else:
        mask = window["bid"] <= signal.entry_price
        col  = "bid"

    hits = window.loc[mask]
    if hits.empty:
        return None

    first = hits.iloc[0]
    return float(first[col]), first.name


# ── SL tick ───────────────────────────────────────────────────────────────────

def find_tick_sl(
    signal:       Any,
    tick_df:      pd.DataFrame,
    m1_df:        pd.DataFrame,
    lookback_m1:  int   = 3,
    percentile:   float = 5.0,
    buffer_usd:   float = 0.10,
    max_tighten:  float = 0.0,   # plafond de resserrement en USD (0 = illimité)
) -> float:
    """
    Calcule un SL affiné à partir des ticks bid/ask autour de la Spring bar.

    Pour les longs :
      - prend le percentile `percentile` des prix bid dans la fenêtre Spring
      - soustrait buffer_usd  →  SL potentiel
      - retourne le MAX entre tick_sl et signal.stop_loss (le plus serré)
      - si max_tighten > 0, annule le resserrement quand il dépasse ce seuil

    Pour les shorts : logique symétrique (percentile élevé, ask).

    Si la fenêtre ticks est vide ou trop petite (<10 ticks), retourne
    signal.stop_loss inchangé.
    """
    wy = getattr(signal, "wyckoff", None)
    if wy is None:
        return signal.stop_loss

    spring_idx = wy.manip_bar
    start_idx  = max(0, spring_idx - lookback_m1)
    end_idx    = min(spring_idx + 1, len(m1_df) - 1)

    spring_start_ts = m1_df.index[start_idx]
    spring_end_ts   = m1_df.index[end_idx]

    try:
        if signal.direction == "long":
            prices = tick_df.loc[spring_start_ts:spring_end_ts, "bid"]
        else:
            prices = tick_df.loc[spring_start_ts:spring_end_ts, "ask"]
    except KeyError:
        return signal.stop_loss

    if len(prices) < 10:
        return signal.stop_loss

    if signal.direction == "long":
        tick_extreme = float(prices.quantile(percentile / 100.0))
        tick_sl      = tick_extreme - buffer_usd
        candidate    = max(tick_sl, signal.stop_loss)
        tightening   = candidate - signal.stop_loss
        if max_tighten > 0.0 and tightening > max_tighten:
            return signal.stop_loss   # resserrement trop fort → SL M1 conservé
        return candidate
    else:
        tick_extreme = float(prices.quantile(1.0 - percentile / 100.0))
        tick_sl      = tick_extreme + buffer_usd
        candidate    = min(tick_sl, signal.stop_loss)
        tightening   = signal.stop_loss - candidate
        if max_tighten > 0.0 and tightening > max_tighten:
            return signal.stop_loss
        return candidate


# ── pipeline complet ──────────────────────────────────────────────────────────

def optimize_signals_with_ticks(
    signals:          list[Any],
    tick_df:          pd.DataFrame,
    m1_df:            pd.DataFrame,
    refine_entry:     bool  = True,
    refine_sl:        bool  = True,
    sl_percentile:    float = 5.0,
    sl_buffer_usd:    float = 0.10,
    sl_lookback_m1:   int   = 3,
    sl_max_tighten:   float = 0.0,   # plafond de resserrement SL (0 = illimité)
    max_entry_wait_s: int   = 300,
) -> list[Any]:
    """
    Applique les deux raffinements tick sur toute la liste de signaux.

    Retourne une nouvelle liste de signaux (dataclasses.replace) avec :
      - entry_price  mis à jour (tick ask/bid au premier franchissement)
      - stop_loss    mis à jour si plus serré qu'en M1

    Les signaux pour lesquels le tick est introuvable (données manquantes,
    ou prix non atteint) sont retournés inchangés.

    Statistiques de raffinement affichées en fin de traitement.
    """
    refined: list[Any] = []
    n_entry_improved = 0
    n_sl_improved    = 0
    entry_deltas: list[float] = []
    sl_deltas:    list[float] = []

    for sig in signals:
        new_entry = sig.entry_price
        new_sl    = sig.stop_loss

        # ── 1. Entrée tick ────────────────────────────────────────────────────
        if refine_entry:
            result = find_tick_entry(sig, tick_df, max_wait_seconds=max_entry_wait_s)
            if result is not None:
                tick_fill, _ = result
                # Toujours utiliser le fill tick réel (ask pour long, bid pour short).
                # tick_fill >= entry_price pour un long (premier ask qui franchit le niveau).
                # Cela corrige le bug où new_entry restait à entry_price (bid/mid)
                # au lieu du vrai ask, provoquant un fill sans spread appliqué.
                delta = tick_fill - sig.entry_price  # >0 = slippage, ~0 = fill au niveau
                entry_deltas.append(delta)
                new_entry = tick_fill
                n_entry_improved += 1

        # ── 2. SL tick ────────────────────────────────────────────────────────
        if refine_sl:
            tick_sl_price = find_tick_sl(
                sig, tick_df, m1_df,
                lookback_m1 = sl_lookback_m1,
                percentile  = sl_percentile,
                buffer_usd  = sl_buffer_usd,
                max_tighten = sl_max_tighten,
            )
            if sig.direction == "long" and tick_sl_price > sig.stop_loss:
                delta = tick_sl_price - sig.stop_loss
                sl_deltas.append(delta)
                new_sl = tick_sl_price
                n_sl_improved += 1
            elif sig.direction == "short" and tick_sl_price < sig.stop_loss:
                delta = sig.stop_loss - tick_sl_price
                sl_deltas.append(delta)
                new_sl = tick_sl_price
                n_sl_improved += 1

        # ── applique les mises à jour ─────────────────────────────────────────
        refined_sig = dataclasses.replace(
            sig,
            entry_price = round(new_entry, 6),
            stop_loss   = round(new_sl,    6),
        )
        refined.append(refined_sig)

    # ── rapport ───────────────────────────────────────────────────────────────
    n = len(signals)
    log.info(
        "TickOptimizer: %d signaux  |  entrées améliorées %d/%d (moy %.2f USD)  |  "
        "SL affinés %d/%d (moy %.2f USD)",
        n,
        n_entry_improved, n,
        (sum(entry_deltas) / len(entry_deltas)) if entry_deltas else 0.0,
        n_sl_improved, n,
        (sum(sl_deltas)    / len(sl_deltas))    if sl_deltas    else 0.0,
    )

    _print_summary(n, n_entry_improved, entry_deltas, n_sl_improved, sl_deltas)

    return refined


def _print_summary(
    n: int,
    n_entry: int, entry_d: list[float],
    n_sl:    int, sl_d:    list[float],
) -> None:
    avg_e = sum(entry_d) / len(entry_d) if entry_d else 0.0
    avg_s = sum(sl_d)    / len(sl_d)    if sl_d    else 0.0
    pct_e = n_entry / n * 100 if n else 0.0
    pct_s = n_sl    / n * 100 if n else 0.0

    print(
        f"\n  ── Tick Optimizer — {n} signaux ──────────────────────────────\n"
        f"  Fills tick trouvés : {n_entry:>3} / {n}  ({pct_e:5.1f}%)  "
        f"  glissement moy = {avg_e:+.3f} USD/oz\n"
        f"  SL affinés        : {n_sl:>3} / {n}  ({pct_s:5.1f}%)  "
        f"  tightening moy = {avg_s:+.3f} USD/oz\n"
        f"  ─────────────────────────────────────────────────────────────"
    )
