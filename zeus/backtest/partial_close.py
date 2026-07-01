"""
Progressive partial-close system.

Models a "laddered exit" where a position is reduced at several R-multiple
targets rather than fully closed at a single take-profit level.

Available profiles
------------------
PartialCloseConfig.default()   — original SMC sniper profile:
  3R  → close 75 % of position; SL to breakeven
  5R  → close 40 % of remaining (85 % total); no SL move
  5R+ → trail 5 R behind highest; hard close at 25 R

PartialCloseConfig.mtf_smc()   — 4-TF cascade scalp profile:
  1R  → SL to breakeven only (no partial close)
  3R  → close 60 % (40 % remaining)
  5R  → close 62.5 % of remaining (= 85 % of original; 15 % remaining)
  10R → close 100 % of remaining (final exit)

Pip convention for XAUUSD:
  1 pip = $1 in quoted price (e.g. 2600.00 → 2601.00 = 1 pip).
  This matches the MT4/MT5 "point" convention for gold.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────────────────
# Configuration dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PartialCloseLevel:
    """One rung on the progressive exit ladder."""
    r_multiple:     float           # trigger when price reaches this R multiple
    close_fraction: float           # fraction of the REMAINING position to close (0–1)
    sl_to_r:        float | None    # move SL to this R multiple after close; None = no move


@dataclass(frozen=True)
class TrailingConfig:
    """
    Trailing stop for the final runner after all discrete levels are hit.

    activate_at_r: start trailing once this R is reached
    trail_r:       keep SL this many R behind the peak price reached
    hard_close_r:  force-close if price reaches this R (regardless of trail)
    """
    activate_at_r: float = 5.0
    trail_r:       float = 5.0
    hard_close_r:  float = 25.0


@dataclass(frozen=True)
class PartialCloseConfig:
    """
    Full progressive exit configuration.

    levels:   ordered list of (R multiple → partial close) steps, ascending R
    trailing: trailing stop config applied after all levels are hit
    """
    levels:   list[PartialCloseLevel]
    trailing: TrailingConfig

    @staticmethod
    def default() -> PartialCloseConfig:
        """
        Default SMC sniper profile:
          3R  → close 75 % of position; SL to breakeven
          5R  → close 40 % of remaining (= 10 % of original; 85 % total)
          5R+ → trail 5 R behind highest; hard close at 25 R
        """
        return PartialCloseConfig(
            levels=[
                PartialCloseLevel(r_multiple=3.0,  close_fraction=0.75, sl_to_r=0.0),
                PartialCloseLevel(r_multiple=5.0,  close_fraction=0.40, sl_to_r=None),
            ],
            trailing=TrailingConfig(activate_at_r=5.0, trail_r=5.0, hard_close_r=25.0),
        )

    @staticmethod
    def mtf_smc() -> PartialCloseConfig:
        """
        4-TF SMC scalp profile — SL stays 20-30 pips; no trailing stop.

          1R  → SL to breakeven only (no partial close)
          3R  → close 60 % of original (40 % remaining)
          5R  → close 62.5 % of remaining (= 25 % of original; 85 % total; 15 % remaining)
          10R → close 100 % of remaining (final exit; 0 % remaining)

        close_fraction values are always relative to the REMAINING quantity at that moment.
        A close_fraction of 0.0 means "move SL only — do not close any quantity."
        """
        return PartialCloseConfig(
            levels=[
                PartialCloseLevel(r_multiple=1.0,   close_fraction=0.0,    sl_to_r=0.0),
                PartialCloseLevel(r_multiple=3.0,   close_fraction=0.60,   sl_to_r=None),
                PartialCloseLevel(r_multiple=5.0,   close_fraction=0.625,  sl_to_r=None),
                PartialCloseLevel(r_multiple=10.0,  close_fraction=1.0,    sl_to_r=None),
            ],
            # Trailing and hard-close disabled — the 10R level is the final exit.
            trailing=TrailingConfig(activate_at_r=999.0, trail_r=5.0, hard_close_r=999.0),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Runtime state (one instance per open position)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PartialCloseState:
    """
    Tracks partial close progress for a single open position.

    All price-level computations are in absolute price units (not fractions).

    Important: sl_price is mutable (moved to breakeven, then trailed), but R-multiple
    calculations ALWAYS use the original SL distance captured at trade open.  This
    prevents move_sl_to_r(0) from collapsing sl_distance to zero and making subsequent
    R levels unreachable.
    """
    entry_price:  float
    sl_price:     float     # current effective SL (may be moved to breakeven)
    is_long:      bool
    original_qty: float

    # Mutable runtime fields (initialised in __post_init__)
    remaining_qty:     float = field(init=False)
    _original_sl_dist: float = field(init=False, repr=False)

    levels_hit:  int         = 0
    peak_r:      float       = 0.0
    trailing_sl: float | None = None
    partial_pnl: float       = 0.0

    def __post_init__(self) -> None:
        self.remaining_qty = self.original_qty
        # Capture the original SL distance once — never changes after open
        if self.is_long:
            self._original_sl_dist = max(self.entry_price - self.sl_price, 1e-10)
        else:
            self._original_sl_dist = max(self.sl_price - self.entry_price, 1e-10)

    # ------------------------------------------------------------------ #
    # Price / R helpers                                                    #
    # ------------------------------------------------------------------ #

    @property
    def sl_distance(self) -> float:
        """
        ORIGINAL SL distance in absolute price units (always positive).

        Used for all R-multiple arithmetic.  Moving sl_price does not change this
        value — it is captured once at construction time.
        """
        return self._original_sl_dist

    def current_r(self, price: float) -> float:
        """
        R multiple at *price* relative to the original SL distance.

        Negative values mean the trade is currently in loss.
        """
        d = self.sl_distance
        if d <= 0.0:
            return 0.0
        if self.is_long:
            return (price - self.entry_price) / d
        return (self.entry_price - price) / d

    def price_at_r(self, r: float) -> float:
        """Absolute price corresponding to R multiple *r*."""
        d = self.sl_distance
        if self.is_long:
            return self.entry_price + r * d
        return self.entry_price - r * d

    def effective_sl(self) -> float:
        """Active SL price: trailing if armed, else the current base SL."""
        return self.trailing_sl if self.trailing_sl is not None else self.sl_price

    # ------------------------------------------------------------------ #
    # Mutation helpers                                                     #
    # ------------------------------------------------------------------ #

    def close_partial(self, price: float, fraction: float) -> float:
        """
        Close *fraction* of the remaining quantity at *price*.

        Returns the gross PnL contribution of this partial close.
        Mutates remaining_qty and partial_pnl.
        """
        qty_to_close = self.remaining_qty * fraction
        if self.is_long:
            pnl = (price - self.entry_price) * qty_to_close
        else:
            pnl = (self.entry_price - price) * qty_to_close
        self.remaining_qty -= qty_to_close
        self.partial_pnl   += pnl
        return pnl

    def move_sl_to_r(self, r: float) -> None:
        """Move the base SL to the price corresponding to R multiple *r*."""
        self.sl_price = self.price_at_r(r)

    def update_trailing(self, price: float, trail_r: float) -> None:
        """
        Advance the trailing stop to (current_price - trail_r × sl_distance)
        for longs, or (current_price + trail_r × sl_distance) for shorts.

        Only moves in the favourable direction (never tightens against position).
        """
        trail_distance = trail_r * self.sl_distance
        if self.is_long:
            new_trail = price - trail_distance
            if self.trailing_sl is None or new_trail > self.trailing_sl:
                self.trailing_sl = new_trail
        else:
            new_trail = price + trail_distance
            if self.trailing_sl is None or new_trail < self.trailing_sl:
                self.trailing_sl = new_trail

    # ------------------------------------------------------------------ #
    # Bar-level update — called once per OHLCV bar                        #
    # ------------------------------------------------------------------ #

    def process_bar(
        self,
        bar_high: float,
        bar_low:  float,
        config:   PartialCloseConfig,
    ) -> list[tuple[str, float, float]]:
        """
        Process one OHLCV bar and apply any triggered partial closes or SL moves.

        Returns a list of (event_type, price, qty) tuples for the events that
        fired during this bar:
          - ("partial", price, qty)  — a partial close level triggered
          - ("sl",      price, qty)  — the remaining position was stopped out
          - ("trail",   price, qty)  — trailing stop triggered final close

        The caller is responsible for recording PnL and updating equity.
        """
        events: list[tuple[str, float, float]] = []
        levels = config.levels
        trailing = config.trailing

        # Determine bar price extremes relative to direction
        favourable = bar_high if self.is_long else bar_low
        unfavourable = bar_low if self.is_long else bar_high

        current_r = self.current_r(favourable)
        if current_r > self.peak_r:
            self.peak_r = current_r

        # ── 1. Check partial close levels ────────────────────────────────
        while self.levels_hit < len(levels):
            level = levels[self.levels_hit]
            if current_r < level.r_multiple:
                break

            trigger_price = self.price_at_r(level.r_multiple)

            if level.close_fraction > 0.0:
                qty = self.remaining_qty * level.close_fraction
                self.close_partial(trigger_price, level.close_fraction)
                events.append(("partial", trigger_price, qty))
            else:
                # close_fraction == 0.0: SL-move only (e.g. breakeven at 1R)
                events.append(("be", trigger_price, 0.0))

            if level.sl_to_r is not None:
                self.move_sl_to_r(level.sl_to_r)

            self.levels_hit += 1

        # ── 2. Update trailing stop ───────────────────────────────────────
        if self.peak_r >= trailing.activate_at_r:
            self.update_trailing(favourable, trailing.trail_r)

        # ── 3. Hard close at max R ────────────────────────────────────────
        if current_r >= trailing.hard_close_r and self.remaining_qty > 0:
            trigger_price = self.price_at_r(trailing.hard_close_r)
            qty = self.remaining_qty
            self.close_partial(trigger_price, 1.0)
            events.append(("partial", trigger_price, qty))
            return events

        # ── 4. Check SL / trailing SL hit ────────────────────────────────
        effective = self.effective_sl()
        sl_hit = (
            (self.is_long  and unfavourable <= effective)
            or (not self.is_long and unfavourable >= effective)
        )
        if sl_hit and self.remaining_qty > 0:
            # Fill at the stop price (not the bar extreme); realistic for 5M
            # intraday data where price trades continuously through the session.
            trigger_price = effective
            qty = self.remaining_qty
            if self.is_long:
                pnl = (trigger_price - self.entry_price) * qty
            else:
                pnl = (self.entry_price - trigger_price) * qty
            self.partial_pnl += pnl
            self.remaining_qty = 0.0
            event_type = "trail" if self.trailing_sl is not None else "sl"
            events.append((event_type, trigger_price, qty))

        return events
