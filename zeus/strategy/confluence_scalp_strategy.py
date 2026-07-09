"""
ConfluenceScalpStrategy — port de la méthode "9-10 confirmations" (setups
A+ / A++) documentée par l'utilisateur, en réutilisant intégralement
l'infrastructure SMC déjà construite et testée (mtf_strategy.py +
zeus/strategy/smc/*). Aucune nouvelle logique de détection n'est ajoutée
ici — uniquement un preset qui active simultanément les gates déjà
existants correspondant à chaque confirmation documentée :

 1. Market structure         → swing_bias / internal_bias (MTFSMCStrategy)
 2. Retracement Fibonacci OTE → require_ote
 3. Zone Supply/Demand (OB)
    + volume profile          → require_poc_zone_confluence
 4. Liquidité externe
    (weekly high/low, session) → require_weekly_bias + require_session_sweep
 5. Imbalance / FVG           → require_entry_fvg
 6. POC                       → require_poc_zone_confluence (avec #3)
 7. Sessions (Asian → Londres) → killzone_only + require_asian_sweep
 8. Setup d'entrée
    (sweep puis retournement)  → require_ltf_sweep + require_entry_pattern
                                  + require_choch_candle
 9. Confluence structure+zone  → min_grade (score de confluence global)
10. 50 % du mouvement (Fibonacci) → require_pd_filter (zone premium/discount)

Cascade : 1H (zones HTF) → 15M (MSS, optionnel) → 1M (déclencheur
d'entrée) — reprise du cascade déjà validé de ScalpSMCStrategy. Le signal
se décide à la clôture de la bougie M1 ; l'exécution "quelques secondes"
décrite par l'utilisateur relève du tick, hors de portée d'un backtest en
barres OHLC — voir la mise en garde dans le rapport de validation.
"""
from __future__ import annotations

import pandas as pd

from zeus.strategy.base import Signal, SignalType
from zeus.strategy.confluence import PatternGrade
from zeus.strategy.mtf_strategy import MTFSMCStrategy


class ConfluenceScalpStrategy(MTFSMCStrategy):
    """
    Toutes les confirmations documentées activées simultanément (setup
    "A++"). Voir le docstring du module pour la correspondance complète
    confirmation → gate.

    Args
    ----
    df_htf_1h : OHLCV 1H (zones HTF : OB / FVG / OTE).
    df_mtf_15m : OHLCV 15M optionnel (gate MSS supplémentaire).
    df_daily : OHLCV journalier (biais journalier + hebdomadaire).
    min_htf_score / min_grade : seuil de score et de grade minimum pour la
        confluence structure+zone (confirmation #9).
    """

    def __init__(
        self,
        df_htf_1h:               pd.DataFrame,
        df_mtf_15m:              pd.DataFrame | None = None,
        df_daily:                pd.DataFrame | None = None,
        sl_pips:                 float = 20.0,
        max_sl_pips:             float = 30.0,
        pip_value:               float = 1.0,
        min_htf_score:           float = 6.0,
        min_grade:               PatternGrade = PatternGrade.B,
        swing_length:            int   = 20,
        internal_length:         int   = 5,
        atr_period:              int   = 100,
        ltf_lookback:            int   = 15,
        mss_lookback:            int   = 20,
        min_wick_ratio:          float = 0.60,
        max_daily_signals:       int   = 3,
        max_signals_per_session: int   = 1,
    ) -> None:
        super().__init__(
            df_htf=df_htf_1h,
            df_mtf=df_mtf_15m,
            df_daily=df_daily,
            sl_pips=sl_pips,
            max_sl_pips=max_sl_pips,
            pip_value=pip_value,
            min_htf_score=min_htf_score,
            min_grade=min_grade,
            swing_length=swing_length,
            internal_length=internal_length,
            atr_period=atr_period,
            ltf_lookback=ltf_lookback,
            zone_tolerance_pct=0.003,
            sweep_zone_tol_pct=0.005,
            mss_lookback=mss_lookback,
            killzone_only=True,                # #7 sessions (Asian → Londres)
            require_entry_fvg=True,             # #5 imbalance / FVG
            require_asian_sweep=True,           # #7 liquidité de session
            require_weekly_bias=True,           # #4 liquidité externe (semaine)
            require_choch_candle=True,          # #8 bougie d'entrée dirigée
            require_pd_filter=True,             # #10 50 % du mouvement
            require_htf_internal_align=False,   # entrées en pullback (comme ScalpSMCStrategy)
            require_clean_approach=True,        # approche propre vers la zone
            require_poc_zone_confluence=True,   # #3 / #6 POC dans la zone
            require_session_sweep=True,         # #4 liquidité de session précédente
            session_sweep_lookback=30,
            require_ltf_sweep=True,             # #8 sweep puis retournement à l'entrée
            ltf_sweep_lookback=3,
            require_ote=True,                   # #2 zone OTE Fibonacci
            require_daily_bias=True,            # #1 / #4 biais journalier
            require_entry_pattern=True,         # #8 pattern de bougie d'entrée
            min_wick_ratio=min_wick_ratio,
            max_daily_signals=max_daily_signals,
            max_signals_per_session=max_signals_per_session,
        )

    def generate_signal(self, df: pd.DataFrame, bar_index: int) -> Signal:
        sig = super().generate_signal(df, bar_index)
        if sig.type != SignalType.NONE:
            reason = sig.reason.replace("MTF grade=", "CONF9 grade=")
            return Signal(sig.type, sig.confidence, reason, sig.bar_index)
        return sig
