//+------------------------------------------------------------------+
//| ZeusWyckoff.mqh — port fidèle de zeus/strategy/supply_demand/    |
//| wyckoff.py (scan _scan_demand/_scan_supply + _check_accum +      |
//| _score ; résultats identiques au chemin "fast" du backtest).     |
//|                                                                  |
//| Le pattern : Accumulation (base serrée) → Manipulation           |
//| (Spring/Upthrust : mèche au-delà de la base, close dedans) →     |
//| MSS (close à travers la borne opposée, sur la barre courante).   |
//+------------------------------------------------------------------+
#property strict
#include "ZeusPivot.mqh"

#define ZW_MIN_RANGE 1e-8

// ── Paramètres (défauts production _WY) ─────────────────────────────
struct ZeusWyckoffParams
  {
   int               lookback;                // 200
   int               min_accum_bars;          // 3
   int               max_accum_bars;          // 20
   double            accum_range_mult;        // 6.0
   int               mss_lookback;            // 60
   double            min_spring_sweep_pct;    // 0.05 (or) / 0.10 (forex)
   double            min_mss_strength_pct;    // 0.03
  };

void ZeusWyckoffDefaultsXau(ZeusWyckoffParams &p)
  {
   p.lookback             = 200;
   p.min_accum_bars       = 3;
   p.max_accum_bars       = 20;
   p.accum_range_mult     = 6.0;
   p.mss_lookback         = 60;
   p.min_spring_sweep_pct = 0.05;
   p.min_mss_strength_pct = 0.03;
  }

void ZeusWyckoffDefaultsForex(ZeusWyckoffParams &p)
  {
   ZeusWyckoffDefaultsXau(p);
   p.min_spring_sweep_pct = 0.10;
  }

// ── Pattern détecté ──────────────────────────────────────────────────
struct ZeusWyckoffPattern
  {
   ENUM_PIVOT_SIDE   side;
   double            accum_high;
   double            accum_low;
   int               accum_bars;
   int               manip_bar;       // index absolu dans le tableau source
   double            manip_extreme;   // wick_low (Spring) / wick_high (Upthrust) = SL
   int               mss_bar;         // index absolu — toujours end_idx-1
   double            mss_close;       // = prix d'entrée suggéré
   double            score;           // 0–10
   datetime          formed_at;
  };

// ── _check_accum ─────────────────────────────────────────────────────
// Fenêtre [max(0, accum_end-max_accum_bars) .. accum_end-1] (slice Python
// highs[start:accum_end] : accum_end EXCLU).
bool ZeusCheckAccum(const double &highs[], const double &lows[],
                    const int accum_end, const ZeusWyckoffParams &p,
                    double &accum_h, double &accum_l)
  {
   int start = MathMax(0, accum_end - p.max_accum_bars);
   accum_h = -DBL_MAX;
   accum_l =  DBL_MAX;
   double sum_rng = 0.0;
   int    cnt     = 0;
   for(int k = start; k < accum_end; k++)
     {
      if(highs[k] > accum_h) accum_h = highs[k];
      if(lows[k]  < accum_l) accum_l = lows[k];
      sum_rng += highs[k] - lows[k];
      cnt++;
     }
   if(cnt == 0)
      return false;
   double accum_rng  = accum_h - accum_l;
   if(accum_rng < ZW_MIN_RANGE)
      return false;
   double avg_candle = sum_rng / cnt;
   if(avg_candle < ZW_MIN_RANGE)
      return false;
   return (accum_rng <= p.accum_range_mult * avg_candle);
  }

// ── _score ───────────────────────────────────────────────────────────
double ZeusWyckoffScore(const ENUM_PIVOT_SIDE side,
                        const double accum_h, const double accum_l,
                        const int accum_cnt,
                        const double &highs[], const double &lows[],
                        const double &closes[], const double &opens[],
                        const int si, const int mi,
                        const ZeusWyckoffParams &p)
  {
   double accum_rng = accum_h - accum_l;
   double score = 0.0;

   // 1. Profondeur d'accumulation
   score += MathMin(2.0, (double)accum_cnt / p.min_accum_bars);

   // 2. Extension du sweep de manipulation
   if(accum_rng > ZW_MIN_RANGE)
     {
      double sweep = (side == PIVOT_DEMAND)
                     ? MathMax(0.0, accum_l - lows[si])
                     : MathMax(0.0, highs[si] - accum_h);
      score += MathMin(2.0, (sweep / accum_rng) * 4.0);
     }

   // 3. Qualité de la bougie de manipulation : longue mèche, petit corps
   double candle_rng = highs[si] - lows[si];
   if(candle_rng > ZW_MIN_RANGE)
     {
      double body_r = MathAbs(closes[si] - opens[si]) / candle_rng;
      double wick_r = (side == PIVOT_DEMAND)
                      ? MathMax(0.0, accum_l - lows[si]) / candle_rng
                      : MathMax(0.0, highs[si] - accum_h) / candle_rng;
      score += MathMin(2.0, wick_r * 3.0 + MathMax(0.0, 0.5 - body_r));
     }

   // 4. Force du close MSS
   if(accum_rng > ZW_MIN_RANGE)
     {
      double strength = (side == PIVOT_DEMAND)
                        ? MathMax(0.0, closes[mi] - accum_h)
                        : MathMax(0.0, accum_l - closes[mi]);
      score += MathMin(2.0, (strength / accum_rng) * 2.0);
     }

   // 5. Vitesse du MSS
   int bars_to_mss = mi - si;
   score += MathMax(0.0, 2.0 - (bars_to_mss - 1) * 0.5);

   return MathMin(10.0, score);
  }

// ── _scan_demand (backward, MSS exigé sur la barre n-1) ─────────────
// Retour : true si pattern trouvé ; indices RELATIFS à la slice.
bool ZeusScanDemand(const double &highs[], const double &lows[],
                    const double &closes[], const int n,
                    const ZeusWyckoffParams &p,
                    double &accum_h, double &accum_l, int &accum_cnt,
                    int &si_out, double &manip_ext, int &mi_out)
  {
   for(int accum_end = n - 2; accum_end >= p.min_accum_bars; accum_end--)
     {
      double ah, al;
      if(!ZeusCheckAccum(highs, lows, accum_end, p, ah, al))
         continue;

      double accum_rng = ah - al;
      double min_sweep = p.min_spring_sweep_pct * accum_rng;
      double min_mss   = p.min_mss_strength_pct * accum_rng;

      int search_end = MathMin(n, accum_end + p.mss_lookback + 2);
      for(int si = accum_end; si < search_end - 1; si++)
        {
         if(lows[si] < al && closes[si] > al)
           {
            if((al - lows[si]) < min_sweep)
               break;                    // spring faible → autre fenêtre accum
            int mss_end = MathMin(n, si + p.mss_lookback + 1);
            for(int mi = si + 1; mi < mss_end; mi++)
              {
               if(closes[mi] > ah)
                 {
                  if(mi == n - 1 && (closes[mi] - ah) >= min_mss)
                    {
                     accum_h   = ah;
                     accum_l   = al;
                     accum_cnt = MathMin(accum_end, p.max_accum_bars);
                     si_out    = si;
                     manip_ext = lows[si];
                     mi_out    = mi;
                     return true;
                    }
                  break;                 // MSS périmé ou trop faible
                 }
              }
            break;                       // un seul candidat spring par fenêtre
           }
        }
     }
   return false;
  }

// ── _scan_supply (Upthrust : continue sur sweep faible) ─────────────
bool ZeusScanSupply(const double &highs[], const double &lows[],
                    const double &closes[], const int n,
                    const ZeusWyckoffParams &p,
                    double &accum_h, double &accum_l, int &accum_cnt,
                    int &si_out, double &manip_ext, int &mi_out)
  {
   for(int accum_end = n - 2; accum_end >= p.min_accum_bars; accum_end--)
     {
      double ah, al;
      if(!ZeusCheckAccum(highs, lows, accum_end, p, ah, al))
         continue;

      double accum_rng = ah - al;
      double min_sweep = p.min_spring_sweep_pct * accum_rng;
      double min_mss   = p.min_mss_strength_pct * accum_rng;

      int search_end = MathMin(n, accum_end + p.mss_lookback + 2);
      for(int si = accum_end; si < search_end - 1; si++)
        {
         if(highs[si] > ah && closes[si] < ah)
           {
            if((highs[si] - ah) < min_sweep)
               continue;                 // upthrust faible → barre suivante
            int mss_end = MathMin(n, si + p.mss_lookback + 1);
            for(int mi = si + 1; mi < mss_end; mi++)
              {
               if(closes[mi] < al)
                 {
                  if(mi == n - 1 && (al - closes[mi]) >= min_mss)
                    {
                     accum_h   = ah;
                     accum_l   = al;
                     accum_cnt = MathMin(accum_end, p.max_accum_bars);
                     si_out    = si;
                     manip_ext = highs[si];
                     mi_out    = mi;
                     return true;
                    }
                  break;
                 }
              }
            break;
           }
        }
     }
   return false;
  }

// ── detect : pattern dont le MSS est la barre end_idx-1 ─────────────
// Tableaux ordonnés ancien→récent ; end_idx EXCLU (comme Python).
bool ZeusWyckoffDetect(const datetime &time[],
                       const double &open[], const double &high[],
                       const double &low[],  const double &close[],
                       const ENUM_PIVOT_SIDE side, const int end_idx,
                       const ZeusWyckoffParams &p,
                       ZeusWyckoffPattern &out)
  {
   if(side == PIVOT_DOJI)
      return false;

   int start   = MathMax(0, end_idx - p.lookback);
   int n_slice = end_idx - start;
   if(n_slice < p.min_accum_bars + 2)
      return false;

   // slices locales (copie — n_slice ≤ 200, coût négligeable)
   double o[], h[], l[], c[];
   ArrayResize(o, n_slice); ArrayResize(h, n_slice);
   ArrayResize(l, n_slice); ArrayResize(c, n_slice);
   for(int k = 0; k < n_slice; k++)
     {
      o[k] = open[start + k];
      h[k] = high[start + k];
      l[k] = low[start + k];
      c[k] = close[start + k];
     }

   double accum_h, accum_l, manip_ext;
   int    accum_cnt, si, mi;
   bool found = (side == PIVOT_DEMAND)
                ? ZeusScanDemand(h, l, c, n_slice, p, accum_h, accum_l, accum_cnt, si, manip_ext, mi)
                : ZeusScanSupply(h, l, c, n_slice, p, accum_h, accum_l, accum_cnt, si, manip_ext, mi);
   if(!found)
      return false;

   double score = ZeusWyckoffScore(side, accum_h, accum_l, accum_cnt,
                                   h, l, c, o, si, mi, p);

   out.side          = side;
   out.accum_high    = NormalizeDouble(accum_h, 6);
   out.accum_low     = NormalizeDouble(accum_l, 6);
   out.accum_bars    = accum_cnt;
   out.manip_bar     = start + si;
   out.manip_extreme = NormalizeDouble(manip_ext, 6);
   out.mss_bar       = start + mi;
   out.mss_close     = NormalizeDouble(c[mi], 6);
   out.score         = NormalizeDouble(score, 2);
   out.formed_at     = time[start + mi];
   return true;
  }
//+------------------------------------------------------------------+
