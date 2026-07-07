//+------------------------------------------------------------------+
//| ZeusSignals.mqh — port de la logique de signal de                |
//| zeus/strategy/supply_demand/sd_strategy.py (config production :  |
//| use_wyckoff_sl=false, fibonacci désactivé, ADX/H4/RSI off).      |
//|                                                                  |
//| Fonctionnement live incrémental :                                |
//|   - les zones M15 sont détectées sur la fenêtre d'historique et  |
//|     fusionnées dans un état persistant (touch/mitigation gardés) |
//|   - à chaque barre M1 close : update des zones, filtres, Wyckoff |
//|     sur la barre courante → signal éventuel.                     |
//| Équivalence avec le batch backtest vérifiée par le harnais CSV.  |
//+------------------------------------------------------------------+
#property strict
#include "ZeusZones.mqh"
#include "ZeusWyckoff.mqh"

// ── Paramètres de signaux (sd_strategy) ──────────────────────────────
struct ZeusSignalParams
  {
   double            min_zone_score;            // asymétrique non utilisé : long=short=min
   double            min_composite_score;
   double            min_wyckoff_score_long;
   double            min_wyckoff_score_short;
   int               signal_cooldown;           // barres M1
   int               trend_slope_lookback;      // barres M15 (longs)
   int               trend_slope_lookback_short;// barres M15 (shorts) — défaut = même
   bool              use_price_above_ema;
   double            ema_atr_tolerance;         // 0 = check binaire ; >0 = zone ATR
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
   double            entry_price;    // mss_close (entrée réelle : marché)
   double            stop_loss;      // zone.wick_extreme (production)
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
   int               n_zones;
   datetime          last_signal_at[ZS_MAX_ZONES];   // cooldown par zone (temps barre M1)
   int               signals_today;
   int               today_key;                      // AAAAMMJJ (UTC)
   datetime          last_m1_processed;              // déduplication de barre
  };

void ZeusSignalStateInit(ZeusSymbolSignalState &st)
  {
   st.n_zones           = 0;
   st.signals_today     = 0;
   st.today_key         = 0;
   st.last_m1_processed = 0;
   for(int i = 0; i < ZS_MAX_ZONES; i++)
      st.last_signal_at[i] = 0;
  }

// ── Fusion des zones re-détectées avec l'état persistant ────────────
// Clé d'identité = (formed_at, side, zone_bottom) comme le Python.
void ZeusMergeZones(ZeusSymbolSignalState &st, ZeusSDZone &fresh[], const int n_fresh)
  {
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
         st.zones[st.n_zones] = fresh[f];
         st.last_signal_at[st.n_zones] = 0;
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
      if(st.zones[k].is_mitigated)
         continue;
      if(ts <= st.zones[k].formed_at)          // zone pas encore active
         continue;
      if(!ZeusPriceInZone(st.zones[k], lo, hi))
         continue;

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

// ── EMA50 M15 : série complète (ewm adjust=false → EMA récursive) ───
void ZeusComputeEMA(const double &close[], const int n, const int period, double &ema[])
  {
   ArrayResize(ema, n);
   if(n == 0) return;
   double alpha = 2.0 / (period + 1.0);
   ema[0] = close[0];
   for(int i = 1; i < n; i++)
      ema[i] = alpha * close[i] + (1.0 - alpha) * ema[i - 1];
  }

// ATR M15 rolling-mean 14 (pour ema_atr_tolerance, forex)
void ZeusComputeATR15(const double &high[], const double &low[],
                      const double &close[], const int n, double &atr[])
  {
   ZeusComputeATR(high, low, close, n, 14, atr);
  }

// ── Détection de signal sur la dernière barre M1 close ──────────────
// m1_*  : barres M1 ancien→récent, la DERNIÈRE (n1-1) est close.
// m15_* : barres M15 ancien→récent (closes uniquement, y compris la
//         barre M15 en cours qui contient la barre M1 courante).
// Retour : true si un signal est émis (rempli dans sig).
bool ZeusDetectSignal(ZeusSymbolSignalState &st,
                      const datetime &m1_time[], const double &m1_open[],
                      const double &m1_high[], const double &m1_low[],
                      const double &m1_close[], const int n1,
                      const datetime &m15_time[], const double &m15_open[],
                      const double &m15_high[], const double &m15_low[],
                      const double &m15_close[], const int n15,
                      const ZeusSignalParams &s,
                      const ZeusWyckoffParams &wp,
                      const ZeusZoneParams &zp,
                      const int utc_offset_seconds,
                      ZeusSignal &sig)
  {
   if(n1 < 3 || n15 < 60)
      return false;

   int i = n1 - 1;                                  // dernière barre M1 close
   datetime ts = m1_time[i];
   if(ts == st.last_m1_processed)
      return false;                                 // barre déjà traitée
   st.last_m1_processed = ts;

   // ── Reset journalier (UTC) ────────────────────────────────────────
   datetime ts_utc = ts - utc_offset_seconds;
   MqlDateTime dt;
   TimeToStruct(ts_utc, dt);
   int day_key = dt.year * 10000 + dt.mon * 100 + dt.day;
   if(day_key != st.today_key)
     {
      st.today_key     = day_key;
      st.signals_today = 0;
     }

   // ── Re-détection + merge des zones, puis update sur la barre ─────
   ZeusSDZone fresh[];
   int nf = ZeusDetectZones(m15_time, m15_open, m15_high, m15_low, m15_close,
                            n15, zp, fresh);
   ZeusMergeZones(st, fresh, nf);
   ZeusUpdateZonesOnBar(st, m1_low[i], m1_high[i], m1_close[i], ts);

   // ── Filtre de session (heure UTC) ─────────────────────────────────
   if(dt.hour < s.session_start_utc || dt.hour >= s.session_end_utc)
      return false;
   if(s.max_signals_per_day > 0 && st.signals_today >= s.max_signals_per_day)
      return false;

   // ── Tendance M15 : EMA50, pente + position (à la barre M15 courante)
   double ema[];
   ZeusComputeEMA(m15_close, n15, 50, ema);
   int j = n15 - 1;                                  // barre M15 contenant ts
   int lbL = s.trend_slope_lookback;
   int lbS = s.trend_slope_lookback_short;
   bool long_ok  = (j >= lbL) && (ema[j] > ema[j - lbL]);
   bool short_ok = (j >= lbS) && (ema[j] < ema[j - lbS]);
   if(!long_ok && !short_ok)
      return false;

   bool ema_long_ok  = true;
   if(s.use_price_above_ema)
     {
      if(s.ema_atr_tolerance > 0)
        {
         double atr15[];
         ZeusComputeATR15(m15_high, m15_low, m15_close, n15, atr15);
         ema_long_ok = (m15_close[j] >= ema[j] - atr15[j] * s.ema_atr_tolerance);
        }
      else
         ema_long_ok = (m15_close[j] > ema[j]);
     }
   // shorts : pas de check prix-vs-EMA (use_price_above_ema_for_shorts=false)

   double lo = m1_low[i], hi = m1_high[i];

   // ── Boucle sur les zones candidates ───────────────────────────────
   // Cache Wyckoff par direction (au plus 1 détection par côté et par barre)
   bool               wy_done[2]  = {false, false};
   bool               wy_found[2] = {false, false};
   ZeusWyckoffPattern wy_pat[2];

   for(int k = 0; k < st.n_zones; k++)
     {
      ZeusSDZone z = st.zones[k];
      if(z.is_mitigated)                    continue;
      if(ts <= z.formed_at)                 continue;   // zone pas active
      if(!ZeusPriceInZone(z, lo, hi))       continue;

      bool is_demand = (z.side == PIVOT_DEMAND);
      double zone_total = ZeusZoneTotal(z);
      if(zone_total < s.min_zone_score)     continue;

      // Filtre de tendance directionnel
      if(is_demand)
        {
         if(!long_ok)                       continue;
         if(!ema_long_ok)                   continue;
        }
      else
        {
         if(!short_ok)                      continue;
        }

      // Cooldown par zone (signal_cooldown barres M1 = minutes)
      if(st.last_signal_at[k] != 0
         && (ts - st.last_signal_at[k]) < s.signal_cooldown * 60)
         continue;

      // Wyckoff sur la barre courante (end_idx = n1, MSS = barre i)
      int side_idx = is_demand ? 0 : 1;
      if(!wy_done[side_idx])
        {
         wy_done[side_idx]  = true;
         wy_found[side_idx] = ZeusWyckoffDetect(
            m1_time, m1_open, m1_high, m1_low, m1_close,
            z.side, n1, wp, wy_pat[side_idx]);
         if(wy_found[side_idx] && wy_pat[side_idx].mss_bar != i)
            wy_found[side_idx] = false;               // MSS pas sur la barre courante
        }
      if(!wy_found[side_idx])               continue;

      double min_wy = is_demand ? s.min_wyckoff_score_long
                                : s.min_wyckoff_score_short;
      if(wy_pat[side_idx].score < min_wy)   continue;

      // ── _build_signal (production : SL = wick_extreme de la zone) ──
      double entry = wy_pat[side_idx].mss_close;
      double slp   = z.wick_extreme;
      double risk  = MathAbs(entry - slp);
      if(risk < 1e-8)                       continue;
      if(s.min_sl_pips > 0 && risk < s.min_sl_pips * s.pip_size)
                                            continue;

      double composite = NormalizeDouble(zone_total, 2);   // fib désactivé
      if(composite < s.min_composite_score) continue;

      // Signal accepté
      sig.direction     = is_demand ? 1 : -1;
      sig.entry_price   = NormalizeDouble(entry, 6);
      sig.stop_loss     = NormalizeDouble(slp, 6);
      sig.zone_score    = NormalizeDouble(zone_total, 2);
      sig.wyckoff_score = wy_pat[side_idx].score;
      sig.composite     = composite;
      sig.formed_at     = ts;

      st.last_signal_at[k] = ts;
      st.signals_today++;
      return true;
     }
   return false;
  }
//+------------------------------------------------------------------+
