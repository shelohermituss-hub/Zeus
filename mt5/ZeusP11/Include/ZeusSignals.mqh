//+------------------------------------------------------------------+
//| ZeusSignals.mqh — port de la logique de signal de                |
//| zeus/strategy/supply_demand/sd_strategy.py (config production :  |
//| use_wyckoff_sl=false, fibonacci désactivé, ADX/H4/RSI off).      |
//|                                                                  |
//| Parité avec le Python (audit v2) :                               |
//|   - M15 RESAMPLÉ depuis la fenêtre M1 (jamais CopyRates M15) —   |
//|     mêmes barres, même EMA, même mappage que pandas resample ;   |
//|   - cooldown compté en BARRES M1 (pas en secondes) ;             |
//|   - zones itérées par formed_at croissant (ordre du batch) ;     |
//|   - purge des zones mitigées (pas de saturation silencieuse).    |
//+------------------------------------------------------------------+
#property strict
#include "ZeusZones.mqh"
#include "ZeusWyckoff.mqh"

// ── Paramètres de signaux (sd_strategy) ──────────────────────────────
struct ZeusSignalParams
  {
   double            min_zone_score;
   double            min_composite_score;
   double            min_wyckoff_score_long;
   double            min_wyckoff_score_short;
   int               signal_cooldown;           // en barres M1
   int               trend_slope_lookback;      // barres M15 (longs)
   int               trend_slope_lookback_short;// barres M15 (shorts)
   bool              use_price_above_ema;
   double            ema_atr_tolerance;         // 0 = check binaire
   int               session_start_utc;
   int               session_end_utc;
   int               max_signals_per_day;
   double            min_sl_pips;
   double            pip_size;
  };

void ZeusSignalDefaultsXau(ZeusSignalParams &s)
  {
   s.min_zone_score             = 5.0;
   s.min_composite_score        = 5.0;
   s.min_wyckoff_score_long     = 5.9;
   s.min_wyckoff_score_short    = 8.5;
   s.signal_cooldown            = 10;
   s.trend_slope_lookback       = 3;
   s.trend_slope_lookback_short = 3;
   s.use_price_above_ema        = true;
   s.ema_atr_tolerance          = 0.0;
   s.session_start_utc          = 7;
   s.session_end_utc            = 21;
   s.max_signals_per_day        = 10;
   s.min_sl_pips                = 0.0;
   s.pip_size                   = 0.01;
  }

void ZeusSignalDefaultsForex(ZeusSignalParams &s, const double pip)
  {
   s.min_zone_score             = 4.0;
   s.min_composite_score        = 4.0;
   s.min_wyckoff_score_long     = 7.5;
   s.min_wyckoff_score_short    = 6.5;
   s.signal_cooldown            = 10;
   s.trend_slope_lookback       = 3;
   s.trend_slope_lookback_short = 3;
   s.use_price_above_ema        = true;
   s.ema_atr_tolerance          = 0.5;
   s.session_start_utc          = 7;
   s.session_end_utc            = 17;
   s.max_signals_per_day        = 6;
   s.min_sl_pips                = 5.0;
   s.pip_size                   = pip;
  }

// ── Signal produit ───────────────────────────────────────────────────
struct ZeusSignal
  {
   int               direction;      // +1 long / -1 short
   double            entry_price;    // mss_close
   double            stop_loss;      // zone.wick_extreme
   double            zone_score;
   double            wyckoff_score;
   double            composite;
   datetime          formed_at;
  };

// ── État persistant par symbole ──────────────────────────────────────
#define ZS_MAX_ZONES 512

struct ZeusSymbolSignalState
  {
   ZeusSDZone        zones[ZS_MAX_ZONES];
   long              last_signal_bar[ZS_MAX_ZONES];  // cooldown en barres M1
   int               n_zones;
   int               signals_today;
   int               today_key;                      // AAAAMMJJ (UTC)
   datetime          last_m1_processed;              // géré par l'appelant
   long              bar_counter;                    // barres M1 traitées
  };

void ZeusSignalStateInit(ZeusSymbolSignalState &st)
  {
   st.n_zones           = 0;
   st.signals_today     = 0;
   st.today_key         = 0;
   st.last_m1_processed = 0;
   st.bar_counter       = 0;
   for(int i = 0; i < ZS_MAX_ZONES; i++)
      st.last_signal_bar[i] = -1000000;
  }

// ── Resampling M15 depuis les barres M1 (pandas resample 15min,     ──
//    label=left, closed=left — buckets alignés sur les timestamps)   ──
void ZeusResampleM15(const datetime &m1_time[], const double &m1_open[],
                     const double &m1_high[], const double &m1_low[],
                     const double &m1_close[], const int n1,
                     datetime &t15[], double &o15[], double &h15[],
                     double &l15[], double &c15[], int &n15)
  {
   ArrayResize(t15, 0); ArrayResize(o15, 0); ArrayResize(h15, 0);
   ArrayResize(l15, 0); ArrayResize(c15, 0);
   n15 = 0;
   datetime cur_bucket = 0;
   for(int i = 0; i < n1; i++)
     {
      datetime bucket = m1_time[i] - (datetime)((long)m1_time[i] % 900);
      if(n15 == 0 || bucket != cur_bucket)
        {
         cur_bucket = bucket;
         n15++;
         ArrayResize(t15, n15); ArrayResize(o15, n15); ArrayResize(h15, n15);
         ArrayResize(l15, n15); ArrayResize(c15, n15);
         t15[n15-1] = bucket;
         o15[n15-1] = m1_open[i];
         h15[n15-1] = m1_high[i];
         l15[n15-1] = m1_low[i];
         c15[n15-1] = m1_close[i];
        }
      else
        {
         if(m1_high[i] > h15[n15-1]) h15[n15-1] = m1_high[i];
         if(m1_low[i]  < l15[n15-1]) l15[n15-1] = m1_low[i];
         c15[n15-1] = m1_close[i];
        }
     }
  }

// ── Purge : retire les zones mitigées quand le tableau approche la  ──
//    saturation (les mitigées ne peuvent plus signaler ni dédupliquer)──
void ZeusPruneZones(ZeusSymbolSignalState &st)
  {
   if(st.n_zones < ZS_MAX_ZONES - 16)
      return;
   int w = 0;
   for(int k = 0; k < st.n_zones; k++)
     {
      if(st.zones[k].is_mitigated)
         continue;
      if(w != k)
        {
         st.zones[w]           = st.zones[k];
         st.last_signal_bar[w] = st.last_signal_bar[k];
        }
      w++;
     }
   st.n_zones = w;
  }

// ── Fusion des zones re-détectées (clé = formed_at, side, bottom) ───
void ZeusMergeZones(ZeusSymbolSignalState &st, ZeusSDZone &fresh[], const int n_fresh)
  {
   ZeusPruneZones(st);
   for(int f = 0; f < n_fresh; f++)
     {
      bool known = false;
      for(int k = 0; k < st.n_zones; k++)
        {
         if(st.zones[k].formed_at == fresh[f].formed_at
            && st.zones[k].side == fresh[f].side
            && MathAbs(st.zones[k].zone_bottom - fresh[f].zone_bottom) < 1e-9)
           { known = true; break; }
        }
      if(!known && st.n_zones < ZS_MAX_ZONES)
        {
         st.zones[st.n_zones]           = fresh[f];
         st.last_signal_bar[st.n_zones] = -1000000;
         st.n_zones++;
        }
     }
  }

// ── Update zones sur une barre M1 close (update_zones du Python) ────
void ZeusUpdateZonesOnBar(ZeusSymbolSignalState &st,
                          const double lo, const double hi,
                          const double close, const datetime ts)
  {
   for(int k = 0; k < st.n_zones; k++)
     {
      if(st.zones[k].is_mitigated)          continue;
      if(ts <= st.zones[k].formed_at)       continue;
      if(!ZeusPriceInZone(st.zones[k], lo, hi)) continue;

      st.zones[k].touch_count++;
      if(ZeusClosedThrough(st.zones[k], close))
        {
         st.zones[k].is_mitigated = true;
         st.zones[k].s_fresh      = 0.0;
        }
      else if(st.zones[k].touch_count == 1)
         st.zones[k].s_fresh = 1.0;
     }
  }

// ── EMA (ewm adjust=false, récursif — identique pandas) ─────────────
void ZeusComputeEMA(const double &close[], const int n, const int period, double &ema[])
  {
   ArrayResize(ema, n);
   if(n == 0) return;
   double alpha = 2.0 / (period + 1.0);
   ema[0] = close[0];
   for(int i = 1; i < n; i++)
      ema[i] = alpha * close[i] + (1.0 - alpha) * ema[i - 1];
  }

// ── Détection de signal sur la barre M1 d'index i = n1-1 ────────────
// Pré-conditions gérées par l'appelant :
//   - la barre i est NOUVELLE et close ;
//   - ZeusMergeZones + ZeusUpdateZonesOnBar(barre i) déjà appelés ;
//   - st.bar_counter déjà incrémenté pour la barre i.
// m15_* : résultat de ZeusResampleM15 sur la MÊME fenêtre M1.
bool ZeusDetectSignalOnBar(ZeusSymbolSignalState &st,
                           const datetime &m1_time[], const double &m1_open[],
                           const double &m1_high[], const double &m1_low[],
                           const double &m1_close[], const int n1,
                           const datetime &t15[], const double &c15a[],
                           const double &h15[], const double &l15[], const int n15,
                           const ZeusSignalParams &s,
                           const ZeusWyckoffParams &wp,
                           const int utc_offset_seconds,
                           ZeusSignal &sig)
  {
   if(n1 < 3 || n15 < 60)
      return false;

   int i = n1 - 1;
   datetime ts     = m1_time[i];
   datetime ts_utc = ts - utc_offset_seconds;

   // ── Reset journalier (UTC) ────────────────────────────────────────
   MqlDateTime dt;
   TimeToStruct(ts_utc, dt);
   int day_key = dt.year * 10000 + dt.mon * 100 + dt.day;
   if(day_key != st.today_key)
     {
      st.today_key     = day_key;
      st.signals_today = 0;
     }

   // ── Session + cap journalier ──────────────────────────────────────
   if(dt.hour < s.session_start_utc || dt.hour >= s.session_end_utc)
      return false;
   if(s.max_signals_per_day > 0 && st.signals_today >= s.max_signals_per_day)
      return false;

   // ── Tendance M15 : EMA50 à la barre CONTENANT ts ─────────────────
   // (le resample est construit depuis la même fenêtre M1 → la dernière
   //  barre M15 contient forcément la barre M1 i : j = n15-1, exactement
   //  le searchsorted(side="right")-1 du Python)
   double ema[];
   ZeusComputeEMA(c15a, n15, 50, ema);
   int j   = n15 - 1;
   int lbL = s.trend_slope_lookback;
   int lbS = s.trend_slope_lookback_short;
   bool long_ok  = (j >= lbL) && (ema[j] > ema[j - lbL]);
   bool short_ok = (j >= lbS) && (ema[j] < ema[j - lbS]);
   if(!long_ok && !short_ok)
      return false;

   bool ema_long_ok = true;
   if(s.use_price_above_ema)
     {
      if(s.ema_atr_tolerance > 0)
        {
         double atr15[];
         ZeusComputeATR(h15, l15, c15a, n15, 14, atr15);
         ema_long_ok = (c15a[j] >= ema[j] - atr15[j] * s.ema_atr_tolerance);
        }
      else
         ema_long_ok = (c15a[j] > ema[j]);
     }

   double lo = m1_low[i], hi = m1_high[i];

   // ── Ordre d'itération : zones par formed_at croissant (batch) ────
   int order[];
   ArrayResize(order, st.n_zones);
   for(int k = 0; k < st.n_zones; k++) order[k] = k;
   for(int a = 1; a < st.n_zones; a++)
     {
      int key = order[a];
      int b = a - 1;
      while(b >= 0 && st.zones[order[b]].formed_at > st.zones[key].formed_at)
        { order[b + 1] = order[b]; b--; }
      order[b + 1] = key;
     }

   // Cache Wyckoff par direction
   bool               wy_done[2]  = {false, false};
   bool               wy_found[2] = {false, false};
   ZeusWyckoffPattern wy_pat[2];

   for(int idx = 0; idx < st.n_zones; idx++)
     {
      int k = order[idx];
      if(st.zones[k].is_mitigated)                    continue;
      if(ts <= st.zones[k].formed_at)                 continue;
      if(!ZeusPriceInZone(st.zones[k], lo, hi))       continue;

      bool   is_demand  = (st.zones[k].side == PIVOT_DEMAND);
      double zone_total = ZeusZoneTotal(st.zones[k]);
      if(zone_total < s.min_zone_score)               continue;

      if(is_demand)
        {
         if(!long_ok || !ema_long_ok)                 continue;
        }
      else
        {
         if(!short_ok)                                continue;
        }

      // Cooldown en BARRES M1 (i - last_signal < cooldown du Python)
      if(st.bar_counter - st.last_signal_bar[k] < s.signal_cooldown)
         continue;

      int side_idx = is_demand ? 0 : 1;
      if(!wy_done[side_idx])
        {
         wy_done[side_idx]  = true;
         wy_found[side_idx] = ZeusWyckoffDetect(
            m1_time, m1_open, m1_high, m1_low, m1_close,
            st.zones[k].side, n1, wp, wy_pat[side_idx]);
         if(wy_found[side_idx] && wy_pat[side_idx].mss_bar != i)
            wy_found[side_idx] = false;
        }
      if(!wy_found[side_idx])                         continue;

      double min_wy = is_demand ? s.min_wyckoff_score_long
                                : s.min_wyckoff_score_short;
      if(wy_pat[side_idx].score < min_wy)             continue;

      // ── _build_signal ──────────────────────────────────────────────
      double entry = wy_pat[side_idx].mss_close;
      double slp   = st.zones[k].wick_extreme;
      double risk  = MathAbs(entry - slp);
      if(risk < 1e-8)                                 continue;
      if(s.min_sl_pips > 0 && risk < s.min_sl_pips * s.pip_size)
                                                      continue;

      double composite = NormalizeDouble(zone_total, 2);
      if(composite < s.min_composite_score)           continue;

      sig.direction     = is_demand ? 1 : -1;
      sig.entry_price   = NormalizeDouble(entry, 6);
      sig.stop_loss     = NormalizeDouble(slp, 6);
      sig.zone_score    = NormalizeDouble(zone_total, 2);
      sig.wyckoff_score = wy_pat[side_idx].score;
      sig.composite     = composite;
      sig.formed_at     = ts;

      st.last_signal_bar[k] = st.bar_counter;
      st.signals_today++;
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
