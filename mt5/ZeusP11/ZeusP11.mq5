//+------------------------------------------------------------------+
//|                                                     ZeusP11.mq5  |
//|  Expert Advisor — portefeuille propfirm P11-V4                   |
//|                                                                  |
//|  Port MQL5 de la stratégie Zeus S&D/Wyckoff validée :            |
//|    - XAUUSD  : V4 production (WS 5.9/8.5, 4T Runner@20R)         |
//|    - 10 paires : WS7.5 forex (TIERED-5R, Runner@10R)             |
//|    - Guard propfirm : daily 3% · DD total 6% · 3 pertes/jour ·   |
//|      flat 21h UTC · risque fixe (pas de compounding)             |
//|                                                                  |
//|  À attacher sur UN SEUL graphique (n'importe lequel, ex. XAUUSD  |
//|  M1) — l'EA gère tous les symboles de InpSymbols via timer.      |
//|                                                                  |
//|  AVANT TOUT TRADING RÉEL :                                       |
//|    1. InpSignalLogMode=true sur données historiques →            |
//|       comparer signals_*.csv avec le Python                      |
//|       (zeus/backtest/compare_ea_signals.py) : exigence 100%.     |
//|    2. Compte démo plusieurs semaines.                            |
//|  Le trading réel sans ces validations viole les règles du projet.|
//+------------------------------------------------------------------+
#property copyright "Zeus"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include "Include\ZeusZones.mqh"
#include "Include\ZeusWyckoff.mqh"
#include "Include\ZeusSignals.mqh"
#include "Include\ZeusGuard.mqh"
#include "Include\ZeusExits.mqh"

// ═════════════════════════ INPUTS ═══════════════════════════════════
input group "── Compte & risque ──"
input double InpAccountBalance     = 100000.0;  // Solde initial de référence (USD)
input double InpRiskPerTradePct    = 0.006;     // Risque par trade (0.006 = 0.6%)
input double InpMaxDailyLossPct    = 0.03;      // Arrêt journalier (3%)
input double InpMaxTotalDDPct      = 0.06;      // Arrêt définitif (6%)
input int    InpMaxLossesPerDay    = 3;         // Stop après N pertes/jour
input int    InpFlatHourUTC        = 21;        // Aucune entrée après (UTC)

input group "── Symboles ──"
input string InpSymbols            = "XAUUSD,GBPUSD,EURUSD,GBPAUD,EURNZD,USDCHF,GER40,CADJPY,NZDUSD,EURJPY,XAGUSD";
input string InpXauSymbol          = "XAUUSD";  // Symbole traité en config V4 or
input string InpPipOverrides       = "GER40=1.0;XAGUSD=0.01"; // pip par symbole (défaut: JPY=0.01, autres=0.0001)

input group "── Seuils de signaux (défauts = valeurs validées) ──"
input double InpXauWSLong          = 5.9;       // Or : score Wyckoff min (longs)
input double InpXauWSShort         = 8.5;       // Or : score Wyckoff min (shorts)
input double InpFxWSLong           = 7.5;       // Forex : score Wyckoff min (longs)
input double InpFxWSShort          = 6.5;       // Forex : score Wyckoff min (shorts)

input group "── Sorties (RR) ──"
input double InpXauRunnerRR        = 20.0;      // Or : runner final (R)
input double InpFxRunnerRR         = 10.0;      // Forex : runner final (R)

input group "── Exécution ──"
input long   InpMagic              = 20260707;  // Magic number
input int    InpTimerSeconds       = 15;        // Fréquence de scan
input int    InpSlippagePoints     = 20;        // Déviation max (points)
input int    InpServerUTCOffsetH   = -99;       // Décalage serveur→UTC (heures), -99 = auto
input bool   InpSignalLogMode      = false;     // true = log CSV, AUCUN ordre
input bool   InpTradingEnabled     = true;      // false = observation seule

// ═════════════════════════ ÉTAT ═════════════════════════════════════
#define ZP11_MAX_SYMBOLS 16

string                 g_symbols[ZP11_MAX_SYMBOLS];
int                    g_n_symbols = 0;
double                 g_pip[ZP11_MAX_SYMBOLS];
bool                   g_is_xau[ZP11_MAX_SYMBOLS];

ZeusSymbolSignalState  g_sig_state[ZP11_MAX_SYMBOLS];
ZeusSignalParams       g_sig_params[ZP11_MAX_SYMBOLS];
ZeusWyckoffParams      g_wy_params[ZP11_MAX_SYMBOLS];
ZeusZoneParams         g_zone_params[ZP11_MAX_SYMBOLS];
ZeusExitParams         g_exit_params[ZP11_MAX_SYMBOLS];
ZeusOpenTrade          g_trades[ZP11_MAX_SYMBOLS];
// caps par symbole/direction (jour et mois — clés AAAAMMJJ / AAAAMM)
int                    g_day_loss_key[ZP11_MAX_SYMBOLS][2];
int                    g_day_loss_cnt[ZP11_MAX_SYMBOLS][2];
int                    g_mon_loss_key[ZP11_MAX_SYMBOLS][2];
int                    g_mon_loss_cnt[ZP11_MAX_SYMBOLS][2];

ZeusGuardConfig        g_guard_cfg;
ZeusGuardState         g_guard;
CTrade                 g_trade;
int                    g_utc_offset_sec = 0;
int                    g_log_handle     = INVALID_HANDLE;

// ═════════════════════════ HELPERS ══════════════════════════════════
double PipForSymbol(const string sym)
  {
   // overrides explicites "SYM=pip;SYM=pip"
   string parts[];
   int n = StringSplit(InpPipOverrides, ';', parts);
   for(int i = 0; i < n; i++)
     {
      string kv[];
      if(StringSplit(parts[i], '=', kv) == 2 && kv[0] == sym)
         return StringToDouble(kv[1]);
     }
   if(StringFind(sym, "JPY") >= 0) return 0.01;
   return 0.0001;
  }

datetime ToUTC(const datetime server_time)
  {
   return server_time - g_utc_offset_sec;
  }

double LotsForRisk(const string sym, const double sl_dist, const double risk_usd)
  {
   double tick_size  = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
   if(tick_size <= 0 || tick_value <= 0)
      return 0.0;
   double loss_per_lot = (sl_dist / tick_size) * tick_value;
   if(loss_per_lot <= 0)
      return 0.0;
   double lots = risk_usd / loss_per_lot;
   double step = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);
   if(step > 0)
      lots = MathRound(lots / step) * step;
   return MathMax(vmin, MathMin(vmax, lots));
  }

// ═════════════════════════ INIT ═════════════════════════════════════
int OnInit()
  {
   // décalage serveur → UTC
   if(InpServerUTCOffsetH == -99)
      g_utc_offset_sec = (int)(TimeCurrent() - TimeGMT());
   else
      g_utc_offset_sec = InpServerUTCOffsetH * 3600;

   // guard
   ZeusGuardDefaults(g_guard_cfg);
   g_guard_cfg.max_daily_loss_pct = InpMaxDailyLossPct;
   g_guard_cfg.max_total_dd_pct   = InpMaxTotalDDPct;
   g_guard_cfg.risk_per_trade_pct = InpRiskPerTradePct;
   g_guard_cfg.max_losses_per_day = InpMaxLossesPerDay;
   g_guard_cfg.flat_hour_utc      = InpFlatHourUTC;
   if(!ZeusGuardConfigValid(g_guard_cfg))
     {
      Print("ZeusP11: configuration de risque INVALIDE — refus de démarrer (fail closed)");
      return INIT_PARAMETERS_INCORRECT;
     }
   ZeusGuardInit(g_guard, InpAccountBalance);

   // symboles
   string parts[];
   g_n_symbols = StringSplit(InpSymbols, ',', parts);
   if(g_n_symbols > ZP11_MAX_SYMBOLS) g_n_symbols = ZP11_MAX_SYMBOLS;
   for(int i = 0; i < g_n_symbols; i++)
     {
      StringTrimLeft(parts[i]);
      StringTrimRight(parts[i]);
      g_symbols[i] = parts[i];
      if(!SymbolSelect(g_symbols[i], true))
        {
         PrintFormat("ZeusP11: symbole %s introuvable chez le broker — refus de démarrer",
                     g_symbols[i]);
         return INIT_PARAMETERS_INCORRECT;
        }
      g_pip[i]    = PipForSymbol(g_symbols[i]);
      g_is_xau[i] = (g_symbols[i] == InpXauSymbol);

      ZeusSignalStateInit(g_sig_state[i]);
      ZeusZoneParamsDefaults(g_zone_params[i]);
      if(g_is_xau[i])
        {
         ZeusSignalDefaultsXau(g_sig_params[i]);
         g_sig_params[i].min_wyckoff_score_long  = InpXauWSLong;
         g_sig_params[i].min_wyckoff_score_short = InpXauWSShort;
         g_sig_params[i].pip_size = g_pip[i];
         ZeusWyckoffDefaultsXau(g_wy_params[i]);
         ZeusExitsXauV4(g_exit_params[i]);
         g_exit_params[i].runner_rr = InpXauRunnerRR;
        }
      else
        {
         ZeusSignalDefaultsForex(g_sig_params[i], g_pip[i]);
         g_sig_params[i].min_wyckoff_score_long  = InpFxWSLong;
         g_sig_params[i].min_wyckoff_score_short = InpFxWSShort;
         ZeusWyckoffDefaultsForex(g_wy_params[i]);
         ZeusExitsForexTiered(g_exit_params[i]);
         g_exit_params[i].runner_rr = InpFxRunnerRR;
        }
      ZeusTradeReset(g_trades[i]);
      for(int d = 0; d < 2; d++)
        {
         g_day_loss_key[i][d] = 0; g_day_loss_cnt[i][d] = 0;
         g_mon_loss_key[i][d] = 0; g_mon_loss_cnt[i][d] = 0;
        }
     }

   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(InpSlippagePoints);

   if(InpSignalLogMode)
     {
      string fname = StringFormat("ZeusP11_signals_%s.csv",
                                  TimeToString(TimeCurrent(), TIME_DATE));
      StringReplace(fname, ".", "-");
      g_log_handle = FileOpen(fname, FILE_WRITE | FILE_CSV | FILE_ANSI, ';');
      if(g_log_handle != INVALID_HANDLE)
         FileWrite(g_log_handle, "formed_at_utc", "symbol", "direction",
                   "entry", "sl", "zone_score", "wyckoff_score");
      Print("ZeusP11: MODE LOG SIGNAUX — aucun ordre ne sera envoyé");
     }

   EventSetTimer(InpTimerSeconds);
   PrintFormat("ZeusP11 démarré : %d symboles · risque %.2f%% (%.0f$) · "
               "guard daily %.0f%%/DD %.0f%% · offset UTC %+d h",
               g_n_symbols, InpRiskPerTradePct * 100,
               ZeusGuardRiskUSD(g_guard, g_guard_cfg),
               InpMaxDailyLossPct * 100, InpMaxTotalDDPct * 100,
               g_utc_offset_sec / 3600);
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   if(g_log_handle != INVALID_HANDLE)
      FileClose(g_log_handle);
  }

// ═════════════════════════ BOUCLE ═══════════════════════════════════
void OnTimer()
  {
   for(int i = 0; i < g_n_symbols; i++)
      ProcessSymbol(i);
  }

void ProcessSymbol(const int i)
  {
   string sym = g_symbols[i];

   // ── Données M1 (ancien→récent) — on exclut la barre en formation ──
   MqlRates m1[];
   ArraySetAsSeries(m1, false);
   int copied = CopyRates(sym, PERIOD_M1, 1, 3000, m1);   // start=1 : barres closes
   if(copied < 300)
      return;

   datetime m1_time[];  double m1_open[], m1_high[], m1_low[], m1_close[];
   ArrayResize(m1_time, copied);  ArrayResize(m1_open, copied);
   ArrayResize(m1_high, copied);  ArrayResize(m1_low, copied);
   ArrayResize(m1_close, copied);
   for(int k = 0; k < copied; k++)
     {
      m1_time[k]  = m1[k].time;
      m1_open[k]  = m1[k].open;
      m1_high[k]  = m1[k].high;
      m1_low[k]   = m1[k].low;
      m1_close[k] = m1[k].close;
     }
   int n1 = copied;
   datetime last_bar = m1_time[n1 - 1];
   if(last_bar == g_sig_state[i].last_m1_processed)
      return;                                    // pas de nouvelle barre close

   // ── Sorties du trade ouvert ───────────────────────────────────────
   if(g_trades[i].active)
     {
      // Reconciliation : position fermée côté broker (SL touché) ?
      bool pos_alive = PositionSelectByTicket(g_trades[i].ticket);
      double pnl_r; string reason;
      if(ZeusManageExits(g_trade, sym, g_trades[i], g_exit_params[i],
                         m1_high[n1 - 1], m1_low[n1 - 1], pnl_r, reason)
         || (!pos_alive && !InpSignalLogMode))
        {
         if(!pos_alive && g_trades[i].active && g_trades[i].remaining_frac > 0)
           {
            // fermée par le SL broker entre deux ticks — même résultat que SL/BE
            double r_exit = g_trades[i].direction
                            * (g_trades[i].sl - g_trades[i].entry_price)
                            / g_trades[i].sl_dist;
            pnl_r  = g_trades[i].realized_r + g_trades[i].remaining_frac * r_exit;
            reason = "SL/BE(broker)";
           }
         OnTradeFinished(i, last_bar, pnl_r, reason);
        }
     }

   // ── Données M15 pour les zones ────────────────────────────────────
   MqlRates m15[];
   ArraySetAsSeries(m15, false);
   int c15 = CopyRates(sym, PERIOD_M15, 0, 400, m15);
   if(c15 < 60)
      return;
   datetime t15[];  double o15[], h15[], l15[], c15a[];
   ArrayResize(t15, c15);  ArrayResize(o15, c15);  ArrayResize(h15, c15);
   ArrayResize(l15, c15);  ArrayResize(c15a, c15);
   for(int k = 0; k < c15; k++)
     {
      t15[k] = m15[k].time;  o15[k] = m15[k].open;  h15[k] = m15[k].high;
      l15[k] = m15[k].low;   c15a[k] = m15[k].close;
     }

   // ── Détection de signal ───────────────────────────────────────────
   ZeusSignal sig;
   bool has_sig = ZeusDetectSignal(g_sig_state[i],
                                   m1_time, m1_open, m1_high, m1_low, m1_close, n1,
                                   t15, o15, h15, l15, c15a, c15,
                                   g_sig_params[i], g_wy_params[i], g_zone_params[i],
                                   g_utc_offset_sec, sig);
   if(!has_sig)
      return;

   datetime ts_utc = ToUTC(sig.formed_at);

   if(InpSignalLogMode)
     {
      if(g_log_handle != INVALID_HANDLE)
        {
         FileWrite(g_log_handle,
                   TimeToString(ts_utc, TIME_DATE | TIME_MINUTES), sym,
                   (sig.direction > 0 ? "long" : "short"),
                   DoubleToString(sig.entry_price, 6),
                   DoubleToString(sig.stop_loss, 6),
                   DoubleToString(sig.zone_score, 2),
                   DoubleToString(sig.wyckoff_score, 2));
         FileFlush(g_log_handle);
        }
      return;                                    // jamais d'ordre en mode log
     }

   // ── Filtres d'entrée (identiques au moteur Python) ────────────────
   if(!InpTradingEnabled)                        return;
   if(g_trades[i].active)                        return;  // 1 trade/symbole

   int dir_idx = (sig.direction > 0) ? 0 : 1;
   MqlDateTime dt;  TimeToStruct(ts_utc, dt);
   int dkey = dt.year * 10000 + dt.mon * 100 + dt.day;
   int mkey = dt.year * 100 + dt.mon;
   int cap_d = g_is_xau[i] ? 1 : 1;              // caps V4 : 1 perte/jour/direction
   int cap_m = 4;
   if(g_day_loss_key[i][dir_idx] == dkey && g_day_loss_cnt[i][dir_idx] >= cap_d)
      return;
   if(g_mon_loss_key[i][dir_idx] == mkey && g_mon_loss_cnt[i][dir_idx] >= cap_m)
      return;

   if(!ZeusGuardCanTrade(g_guard, g_guard_cfg, ts_utc))
      return;

   // ── Entrée marché avec SL attaché ─────────────────────────────────
   double risk_usd = ZeusGuardRiskUSD(g_guard, g_guard_cfg);
   MqlTick tick;
   if(!SymbolInfoTick(sym, tick))
      return;
   double price   = (sig.direction > 0) ? tick.ask : tick.bid;
   double sl_dist = MathAbs(price - sig.stop_loss);
   if(sl_dist <= 0)
      return;
   if((sig.direction > 0) != (sig.stop_loss < price))
      return;                                    // SL du mauvais côté → refus

   double lots = LotsForRisk(sym, sl_dist, risk_usd);
   if(lots <= 0)
      return;

   bool ok = (sig.direction > 0)
             ? g_trade.Buy(lots, sym, 0.0, sig.stop_loss, 0.0, "zeus-p11")
             : g_trade.Sell(lots, sym, 0.0, sig.stop_loss, 0.0, "zeus-p11");
   if(!ok || g_trade.ResultRetcode() != TRADE_RETCODE_DONE)
     {
      PrintFormat("ZeusP11 %s: ordre rejeté (retcode=%d) — pas de trade",
                  sym, g_trade.ResultRetcode());
      return;
     }

   double fill = g_trade.ResultPrice();
   if(fill <= 0) fill = price;

   ZeusOpenTrade t;
   ZeusTradeReset(t);
   t.active       = true;
   t.ticket       = g_trade.ResultOrder();
   t.direction    = sig.direction;
   t.entry_price  = fill;
   t.sl           = sig.stop_loss;
   t.sl_dist      = MathAbs(fill - sig.stop_loss);
   t.lots_initial = g_trade.ResultVolume();
   t.remaining_frac = 1.0;
   t.opened_at    = sig.formed_at;
   g_trades[i]    = t;

   PrintFormat("ZeusP11 %s: ENTRÉE %s %.2f lots @ %.5f SL %.5f (WS=%.2f zone=%.2f)",
               sym, (sig.direction > 0 ? "LONG" : "SHORT"),
               t.lots_initial, fill, sig.stop_loss,
               sig.wyckoff_score, sig.zone_score);
  }

// ── Fin de trade : guard + caps ──────────────────────────────────────
void OnTradeFinished(const int i, const datetime bar_time,
                     const double pnl_r, const string reason)
  {
   datetime ts_utc  = ToUTC(bar_time);
   double   pnl_usd = pnl_r * ZeusGuardRiskUSD(g_guard, g_guard_cfg);
   ZeusGuardOnTradeClosed(g_guard, g_guard_cfg, ts_utc, pnl_usd);

   if(pnl_r < 0)
     {
      int dir_idx = (g_trades[i].direction > 0) ? 0 : 1;
      MqlDateTime dt;
      datetime opened_utc = ToUTC(g_trades[i].opened_at);
      TimeToStruct(opened_utc, dt);
      int dkey = dt.year * 10000 + dt.mon * 100 + dt.day;
      int mkey = dt.year * 100 + dt.mon;
      if(g_day_loss_key[i][dir_idx] != dkey)
        { g_day_loss_key[i][dir_idx] = dkey; g_day_loss_cnt[i][dir_idx] = 0; }
      g_day_loss_cnt[i][dir_idx]++;
      if(g_mon_loss_key[i][dir_idx] != mkey)
        { g_mon_loss_key[i][dir_idx] = mkey; g_mon_loss_cnt[i][dir_idx] = 0; }
      g_mon_loss_cnt[i][dir_idx]++;
     }

   PrintFormat("ZeusP11 %s: SORTIE [%s] %+.2fR (%+.0f$) · équity guard %.0f$ · "
               "halt_jour=%s halt_def=%s",
               g_symbols[i], reason, pnl_r, pnl_usd, g_guard.equity,
               (g_guard.halted_for_day ? "OUI" : "non"),
               (g_guard.halted_permanently ? "OUI ⛔" : "non"));

   ZeusTradeReset(g_trades[i]);
  }
//+------------------------------------------------------------------+
