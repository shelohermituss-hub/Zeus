//+------------------------------------------------------------------+
//| ZeusZones.mqh — port fidèle de zeus/strategy/supply_demand/      |
//| zone_detector.py.                                                 |
//|                                                                  |
//| Convention d'indexation : tous les tableaux OHLC sont ordonnés   |
//| du plus ANCIEN (index 0) au plus RÉCENT (index n-1), comme le    |
//| DataFrame pandas de référence.  Utiliser ArraySetAsSeries(false) |
//| après CopyRates.                                                 |
//+------------------------------------------------------------------+
#property strict
#include "ZeusPivot.mqh"

#define ZZ_EPS 1e-8

// ── Paramètres (défauts identiques à zone_detector.py) ───────────────
struct ZeusZoneParams
  {
   int               swing_lookback;        // 5
   int               bos_lookback;          // 20
   int               bos_max_bars;          // 40  (BOS haussier)
   int               bos_max_bars_supply;   // 80  (BOS baissier)
   int               max_base_candles;      // 10
   double            min_pivot_score;       // 4.0
   double            min_zone_score;        // 4.0
   int               atr_period;            // 14
  };

void ZeusZoneParamsDefaults(ZeusZoneParams &p)
  {
   p.swing_lookback      = 5;
   p.bos_lookback        = 20;
   p.bos_max_bars        = 40;
   p.bos_max_bars_supply = 80;
   p.max_base_candles    = 10;
   p.min_pivot_score     = 4.0;
   p.min_zone_score      = 4.0;
   p.atr_period          = 14;
  }

// ── Zone S&D ─────────────────────────────────────────────────────────
struct ZeusSDZone
  {
   ENUM_PIVOT_SIDE   side;
   double            zone_top;       // body_high du pivot
   double            zone_bottom;    // body_low du pivot
   double            wick_extreme;   // SL serré (wick_low demand / wick_high supply)
   int               pivot_bar;
   datetime          formed_at;      // timestamp de la barre de BOS
   int               bos_bar;
   double            bos_level;
   int               base_candles;
   // score 5 critères (0–2 chacun)
   double            s_bos, s_impulse, s_time, s_fresh, s_sweep;
   bool              is_mitigated;
   int               touch_count;
  };

double ZeusZoneTotal(const ZeusSDZone &z)
  {
   return NormalizeDouble(z.s_bos + z.s_impulse + z.s_time + z.s_fresh + z.s_sweep, 2);
  }

double ZeusZoneMidpoint(const ZeusSDZone &z) { return (z.zone_top + z.zone_bottom) / 2.0; }
double ZeusZoneHeight(const ZeusSDZone &z)   { return z.zone_top - z.zone_bottom; }

bool ZeusPriceInZone(const ZeusSDZone &z, const double lo, const double hi)
  {
   return (lo <= z.zone_top && hi >= z.zone_bottom);
  }

bool ZeusClosedThrough(const ZeusSDZone &z, const double close)
  {
   if(z.side == PIVOT_DEMAND)
      return (close < z.wick_extreme);
   return (close > z.wick_extreme);
  }

// ── ATR : True Range moyenné en rolling mean, min_periods=1 ─────────
// (pandas .rolling(period, min_periods=1).mean() — PAS l'ATR Wilder de MT5)
void ZeusComputeATR(const double &high[], const double &low[],
                    const double &close[], const int n,
                    const int period, double &atr[])
  {
   ArrayResize(atr, n);
   double tr[];
   ArrayResize(tr, n);
   for(int i = 0; i < n; i++)
     {
      double hl = high[i] - low[i];
      if(i == 0)
         tr[i] = hl;                             // prev_close = NaN → max = h-l
      else
        {
         double hc = MathAbs(high[i] - close[i-1]);
         double lc = MathAbs(low[i]  - close[i-1]);
         tr[i] = MathMax(hl, MathMax(hc, lc));
        }
     }
   double sum = 0.0;
   for(int i = 0; i < n; i++)
     {
      sum += tr[i];
      if(i >= period)
         sum -= tr[i - period];
      int cnt = (i < period) ? (i + 1) : period;
      atr[i] = sum / cnt;
     }
  }

// ── Quantile linéaire (pandas Series.quantile, interpolation=linear) ─
double ZeusQuantile(double &values[], const int n, const double q)
  {
   if(n <= 0) return 0.0;
   double tmp[];
   ArrayResize(tmp, n);
   ArrayCopy(tmp, values, 0, 0, n);
   ArraySort(tmp);
   double pos = q * (n - 1);
   int    lo  = (int)MathFloor(pos);
   int    hi  = (int)MathCeil(pos);
   if(lo == hi) return tmp[lo];
   double frac = pos - lo;
   return tmp[lo] + (tmp[hi] - tmp[lo]) * frac;
  }

// ── Swing detection ──────────────────────────────────────────────────
bool ZeusIsSwingLow(const double &low[], const int n, const int i, const int lb)
  {
   double lo_i = low[i];
   int start = MathMax(0, i - lb);
   int end   = MathMin(n, i + lb + 1);
   double mn = DBL_MAX;
   for(int k = start; k < end; k++)
      if(low[k] < mn) mn = low[k];
   return (lo_i <= mn);
  }

bool ZeusIsSwingHigh(const double &high[], const int n, const int i, const int lb)
  {
   double hi_i = high[i];
   int start = MathMax(0, i - lb);
   int end   = MathMin(n, i + lb + 1);
   double mx = -DBL_MAX;
   for(int k = start; k < end; k++)
      if(high[k] > mx) mx = high[k];
   return (hi_i >= mx);
  }

// ── BOS detection ────────────────────────────────────────────────────
bool ZeusFindBullishBOS(const double &high[], const double &close[],
                        const int n, const int pivot_bar,
                        const ZeusZoneParams &p,
                        int &bos_bar, double &bos_level)
  {
   int lb_start = MathMax(0, pivot_bar - p.bos_lookback);
   double lvl = -DBL_MAX;
   for(int k = lb_start; k < pivot_bar; k++)
      if(high[k] > lvl) lvl = high[k];
   if(lvl == -DBL_MAX) return false;

   int end = MathMin(n, pivot_bar + p.bos_max_bars + 1);
   for(int j = pivot_bar + 1; j < end; j++)
      if(close[j] > lvl)
        { bos_bar = j; bos_level = lvl; return true; }
   return false;
  }

bool ZeusFindBearishBOS(const double &low[], const double &close[],
                        const int n, const int pivot_bar,
                        const ZeusZoneParams &p,
                        int &bos_bar, double &bos_level)
  {
   int lb_start = MathMax(0, pivot_bar - p.bos_lookback);
   double lvl = DBL_MAX;
   for(int k = lb_start; k < pivot_bar; k++)
      if(low[k] < lvl) lvl = low[k];
   if(lvl == DBL_MAX) return false;

   int end = MathMin(n, pivot_bar + p.bos_max_bars_supply + 1);
   for(int j = pivot_bar + 1; j < end; j++)
      if(close[j] < lvl)
        { bos_bar = j; bos_level = lvl; return true; }
   return false;
  }

// ── Scoring ──────────────────────────────────────────────────────────
double ZeusScoreBOS(const double &close[], const double &atr[],
                    const int bos_bar, const double bos_level,
                    const ENUM_PIVOT_SIDE side)
  {
   double c     = close[bos_bar];
   double atr_v = atr[bos_bar];
   if(atr_v < ZZ_EPS)
      return 1.0;
   double dist  = (side == PIVOT_DEMAND) ? (c - bos_level) : (bos_level - c);
   double ratio = dist / atr_v;
   if(ratio >= 0.50) return 2.0;
   if(ratio >= 0.30) return 1.5;
   if(ratio >= 0.10) return 1.0;
   return 0.5;
  }

double ZeusScoreImpulse(const double &open[], const double &high[],
                        const double &low[], const double &close[],
                        const int pivot_bar, const int bos_bar)
  {
   // impulse = barres [pivot_bar+1 .. bos_bar] inclus
   int start = pivot_bar + 1;
   int end   = bos_bar;          // inclus
   if(end < start)
      return 1.0;
   double sum = 0.0;
   int    cnt = 0;
   for(int i = start; i <= end; i++)
     {
      double rng = high[i] - low[i];
      if(rng < ZZ_EPS)
         continue;                              // masque rng >= eps
      sum += MathAbs(close[i] - open[i]) / rng;
      cnt++;
     }
   if(cnt == 0)
      return 1.0;
   double avg = sum / cnt;
   if(avg >= 0.70) return 2.0;
   if(avg >= 0.55) return 1.5;
   if(avg >= 0.40) return 1.0;
   return 0.5;
  }

double ZeusScoreTime(const int base_candles)
  {
   if(base_candles <= 2) return 2.0;
   if(base_candles <= 4) return 1.5;
   if(base_candles <= 6) return 1.0;
   if(base_candles <= 8) return 0.5;
   return 0.0;
  }

double ZeusScoreSweep(const double &high[], const double &low[],
                      const double &close[],
                      const int pivot_bar, const ENUM_PIVOT_SIDE side,
                      const int lookback = 20)
  {
   int start = MathMax(0, pivot_bar - lookback);
   int cnt   = pivot_bar - start;                // fenêtre [start, pivot_bar)
   if(cnt < 3)
      return 0.0;

   if(side == PIVOT_DEMAND)
     {
      double lows[];
      ArrayResize(lows, cnt);
      for(int k = 0; k < cnt; k++) lows[k] = low[start + k];
      double ref = ZeusQuantile(lows, cnt, 0.25);
      for(int k = start; k < pivot_bar; k++)
         if(low[k] < ref && close[k] > ref)
            return 2.0;
     }
   else
     {
      double highs[];
      ArrayResize(highs, cnt);
      for(int k = 0; k < cnt; k++) highs[k] = high[start + k];
      double ref = ZeusQuantile(highs, cnt, 0.75);
      for(int k = start; k < pivot_bar; k++)
         if(high[k] > ref && close[k] < ref)
            return 2.0;
     }
   return 0.0;
  }

// ── Base start (_find_base_start) ────────────────────────────────────
int ZeusFindBaseStart(const double &high[], const double &low[],
                      const double &atr[], const int pivot_bar,
                      const ZeusZoneParams &p)
  {
   double avg_atr   = atr[pivot_bar];
   double threshold = (avg_atr > ZZ_EPS) ? avg_atr * 1.5 : DBL_MAX;

   int limit = MathMax(-1, pivot_bar - p.max_base_candles);
   for(int i = pivot_bar - 1; i > limit; i--)
     {
      double rng = high[i] - low[i];
      if(rng > threshold)
         return i + 1;
     }
   return MathMax(0, pivot_bar - p.max_base_candles);
  }

// ── Déduplication (_is_duplicate) ────────────────────────────────────
bool ZeusIsDuplicate(const ZeusSDZone &nz, const ZeusSDZone &zones[], const int n_zones)
  {
   for(int k = 0; k < n_zones; k++)
     {
      if(zones[k].side != nz.side || zones[k].is_mitigated)
         continue;
      double dist = MathAbs(ZeusZoneMidpoint(zones[k]) - ZeusZoneMidpoint(nz));
      double ref  = MathMin(ZeusZoneHeight(zones[k]), ZeusZoneHeight(nz));
      if(ref < ZZ_EPS)
         continue;
      if(dist < ref * 0.5)
         return true;
     }
   return false;
  }

// ── detect_zones : scan complet, tri par formed_at ───────────────────
// time[]/open[]/... : barres du timeframe de zones (M15), ordre ancien→récent.
// Retourne le nombre de zones écrites dans out[].
int ZeusDetectZones(const datetime &time[],
                    const double &open[], const double &high[],
                    const double &low[],  const double &close[],
                    const int n, const ZeusZoneParams &p,
                    ZeusSDZone &out[])
  {
   ArrayResize(out, 0);
   if(n < p.swing_lookback * 2 + 2)
      return 0;

   double atr[];
   ZeusComputeATR(high, low, close, n, p.atr_period, atr);

   int lo     = p.swing_lookback;
   int hi_lim = n - 1;
   if(hi_lim <= lo)
      return 0;

   int n_zones = 0;
   for(int i = lo; i < hi_lim; i++)
     {
      ZeusPivotCandle pc;
      ZeusAnalyzeCandle(time[i], open[i], high[i], low[i], close[i], pc);
      if(pc.score < p.min_pivot_score)
         continue;

      ENUM_PIVOT_SIDE side;
      int    bos_bar   = -1;
      double bos_level = 0.0;
      bool   found     = false;

      if(pc.side == PIVOT_DEMAND && ZeusIsSwingLow(low, n, i, p.swing_lookback))
        {
         found = ZeusFindBullishBOS(high, close, n, i, p, bos_bar, bos_level);
         side  = PIVOT_DEMAND;
        }
      else if(pc.side == PIVOT_SUPPLY && ZeusIsSwingHigh(high, n, i, p.swing_lookback))
        {
         found = ZeusFindBearishBOS(low, close, n, i, p, bos_bar, bos_level);
         side  = PIVOT_SUPPLY;
        }
      else
         continue;

      if(!found)
         continue;
      if(pc.body_high <= pc.body_low)             // _build_zone : garde
         continue;

      ZeusSDZone z;
      z.side         = side;
      z.zone_top     = pc.body_high;
      z.zone_bottom  = pc.body_low;
      z.wick_extreme = (side == PIVOT_DEMAND) ? pc.wick_low : pc.wick_high;
      z.pivot_bar    = i;
      z.formed_at    = time[bos_bar];
      z.bos_bar      = bos_bar;
      z.bos_level    = bos_level;

      int base_start  = ZeusFindBaseStart(high, low, atr, i, p);
      z.base_candles  = i - base_start + 1;

      z.s_bos     = ZeusScoreBOS(close, atr, bos_bar, bos_level, side);
      z.s_impulse = ZeusScoreImpulse(open, high, low, close, i, bos_bar);
      z.s_time    = ZeusScoreTime(z.base_candles);
      z.s_fresh   = 2.0;
      z.s_sweep   = ZeusScoreSweep(high, low, close, i, side);
      z.is_mitigated = false;
      z.touch_count  = 0;

      if(ZeusZoneTotal(z) < p.min_zone_score)
         continue;
      if(ZeusIsDuplicate(z, out, n_zones))
         continue;

      ArrayResize(out, n_zones + 1);
      out[n_zones] = z;
      n_zones++;
     }

   // tri par formed_at (stable : insertion)
   for(int a = 1; a < n_zones; a++)
     {
      ZeusSDZone key = out[a];
      int b = a - 1;
      while(b >= 0 && out[b].formed_at > key.formed_at)
        { out[b + 1] = out[b]; b--; }
      out[b + 1] = key;
     }
   return n_zones;
  }
//+------------------------------------------------------------------+
