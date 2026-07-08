//+------------------------------------------------------------------+
//|                                                     ZeusP11.mq5  |
//|  Expert Advisor — portefeuille propfirm P11-V4  (audit v2)       |
//|                                                                  |
//|  Port MQL5 de la stratégie Zeus S&D/Wyckoff validée :            |
//|    - XAUUSD  : V4 production (WS 5.9/8.5, 4T Runner@20R)         |
//|    - 10 paires : WS7.5 forex (TIERED-5R, Runner@10R)             |
//|    - Guard propfirm : daily 3% · DD total 6% · 3 pertes/jour ·   |
//|      flat 21h UTC · risque fixe (pas de compounding)             |
//|                                                                  |
//|  À attacher sur UN SEUL graphique — l'EA gère tous les symboles  |
//|  de InpSymbols via timer.  Le M15 est resamplé depuis les M1     |
//|  (jamais CopyRates M15) pour une parité exacte avec le Python.   |
//|                                                                  |
//|  AVANT TOUT TRADING RÉEL : harnais d'équivalence (100%) puis     |
//|  démo — voir README.md.                                          |
//+------------------------------------------------------------------+
#property copyright "Zeus"
#property version   "2.00"
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
input string InpSymbols            = "XAUUSD,GBPUSD,EURUSD,GBPAUD,EURNZD,USDCHF,DAX,CADJPY,NZDUSD,EURJPY,XAGUSD";
input string InpXauSymbol          = "XAUUSD";  // Symbole traité en config V4 or
input string InpPipOverrides       = "DAX=1.0;XAGUSD=0.01"; // pip par symbole (défaut: JPY=0.01, autres=0.0001)

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
#define ZP11_M1_WINDOW   3000     // = M1_LIMIT du moteur Python

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
// caps par symbole/direction : [i][0]=long, [i][1]=short
int                    g_day_loss_key[ZP11_MAX_SYMBOLS][2];
int                    g_day_loss_cnt[ZP11_MAX_SYMBOLS][2];
int                    g_mon_loss_key[ZP11_MAX_SYMBOLS][2];
int                    g_mon_loss_cnt[ZP11_MAX_SYMBOLS][2];
bool                   g_warmed_up[ZP11_MAX_SYMBOLS];

ZeusGuardConfig        g_guard_cfg;
ZeusGuardState         g_guard;
CTrade                 g_trade;
int                    g_utc_offset_sec = 0;
int                    g_log_handle     = INVALID_HANDLE;

// ═════════════════════════ HELPERS ══════════════════════════════════
double PipForSymbol(const string sym)
  {
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
      lots = MathFloor(lots / step) * step;   // floor : jamais > risque cible
   if(lots < vmin)
      return 0.0;                             // risque min > cible → refus
   return MathMin(vmax, lots);
  }

// Ticket de la position ouverte pour (symbole, magic) — compatible hedging
ulong FindPositionTicket(const string sym)
  {
   for(int p = PositionsTotal() - 1; p >= 0; p--)
     {
      ulong tk = PositionGetTicket(p);
      if(tk == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) == sym
         && PositionGetInteger(POSITION_MAGIC) == InpMagic)
         return tk;
     }
   return 0;
  }

// ═════════════════════════ INIT ═════════════════════════════════════
int OnInit()
  {
   if(InpServerUTCOffsetH == -99)
      g_utc_offset_sec = (int)(TimeCurrent() - TimeGMT());
   else
      g_utc_offset_sec = InpServerUTCOffsetH * 3600;

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
      g_warmed_up[i] = false;
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
      MqlDateTime now;
      TimeToStruct(TimeCurrent(), now);
      string fname = StringFormat("ZeusP11_signals_%04d%02d%02d.csv",
                                  now.year, now.mon, now.day);
      g_log_handle = FileOpen(fname, FILE_WRITE | FILE_CSV | FILE_ANSI, ';');
      if(g_log_handle != INVALID_HANDLE)
         FileWrite(g_log_handle, "formed_at_utc", "symbol", "direction",
                   "entry", "sl", "zone_score", "wyckoff_score");
      Print("ZeusP11: MODE LOG SIGNAUX — aucun ordre ne sera envoyé");
     }

   EventSetTimer(InpTimerSeconds);
   PrintFormat("ZeusP11 v2 démarré : %d symboles · risque %.2f%% (%.0f$) · "
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

// Le testeur de stratégie n'appelle pas toujours OnTimer : OnTick relaie.
void OnTick()
  {
   static datetime last_tick_run = 0;
   if(TimeCurrent() - last_tick_run < InpTimerSeconds)
      return;
   last_tick_run = TimeCurrent();
   for(int i = 0; i < g_n_symbols; i++)
      ProcessSymbol(i);
  }

void ProcessSymbol(const int i)
  {
   string sym = g_symbols[i];

   // ── Fenêtre M1 close (start=1 : exclut la barre en formation) ────
   MqlRates m1[];
   ArraySetAsSeries(m1, false);
   int copied = CopyRates(sym, PERIOD_M1, 1, ZP11_M1_WINDOW, m1);
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

   if(m1_time[n1 - 1] == g_sig_state[i].last_m1_processed)
      return;                                    // pas de nouvelle barre close

   // ── M15 resamplé depuis la MÊME fenêtre M1 (parité Python) ───────
   datetime t15[];  double o15[], h15[], l15[], c15a[];
   int n15 = 0;
   ZeusResampleM15(m1_time, m1_open, m1_high, m1_low, m1_close, n1,
                   t15, o15, h15, l15, c15a, n15);
   if(n15 < 60)
      return;

   // ── Zones : re-détection + merge (une fois par nouvelle barre) ───
   ZeusSDZone fresh[];
   int nf = ZeusDetectZones(t15, o15, h15, l15, c15a, n15, g_zone_params[i], fresh);
   ZeusMergeZones(g_sig_state[i], fresh, nf);

   // ── Rattrapage : chaque barre close non traitée, dans l'ordre ────
   int first_new = 0;
   if(g_sig_state[i].last_m1_processed != 0)
     {
      first_new = n1;                             // défaut : rien de nouveau
      for(int k = n1 - 1; k >= 0; k--)
        {
         if(m1_time[k] <= g_sig_state[i].last_m1_processed)
           { first_new = k + 1; break; }
         if(k == 0)
            first_new = 0;                        // tout est nouveau
        }
     }

   bool first_run = (g_sig_state[i].last_m1_processed == 0);

   for(int b = first_new; b < n1; b++)
     {
      // 1. Sorties du trade ouvert sur CETTE barre
      if(g_trades[i].active)
        {
         double pnl_r = 0.0; string reason = "";
         bool finished = ZeusManageExits(g_trade, sym, g_trades[i], g_exit_params[i],
                                         m1_high[b], m1_low[b], pnl_r, reason);
         if(!finished && !InpSignalLogMode)
           {
            // Réconciliation : SL broker a fermé la position entre 2 ticks
            if(!PositionSelectByTicket(g_trades[i].ticket))
              {
               double r_exit = g_trades[i].direction
                               * (g_trades[i].sl - g_trades[i].entry_price)
                               / g_trades[i].sl_dist;
               pnl_r    = g_trades[i].realized_r
                          + g_trades[i].remaining_frac * r_exit;
               reason   = "SL/BE(broker)";
               finished = true;
              }
           }
         if(finished)
            OnTradeFinished(i, m1_time[b], pnl_r, reason);
        }

      // 2. État des zones sur CETTE barre
      ZeusUpdateZonesOnBar(g_sig_state[i], m1_low[b], m1_high[b], m1_close[b],
                           m1_time[b]);
      g_sig_state[i].bar_counter++;
     }
   g_sig_state[i].last_m1_processed = m1_time[n1 - 1];

   // Premier passage : warm-up de l'état uniquement, pas de signal
   // (évite d'entrer sur un signal périmé au moment de l'attache)
   if(first_run)
     {
      g_warmed_up[i] = true;
      return;
     }

   // ── Signal sur la dernière barre close uniquement ─────────────────
   ZeusSignal sig;
   bool has_sig = ZeusDetectSignalOnBar(g_sig_state[i],
                                        m1_time, m1_open, m1_high, m1_low,
                                        m1_close, n1,
                                        t15, c15a, h15, l15, n15,
                                        g_sig_params[i], g_wy_params[i],
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
      return;
     }

   // ── Filtres d'entrée (moteur Python) ──────────────────────────────
   if(!InpTradingEnabled)                        return;
   if(g_trades[i].active)                        return;

   int dir_idx = (sig.direction > 0) ? 0 : 1;
   MqlDateTime dt;  TimeToStruct(ts_utc, dt);
   int dkey = dt.year * 10000 + dt.mon * 100 + dt.day;
   int mkey = dt.year * 100 + dt.mon;
   if(g_day_loss_key[i][dir_idx] == dkey && g_day_loss_cnt[i][dir_idx] >= 1)
      return;                                    // cap V4 : 1 perte/jour/direction
   if(g_mon_loss_key[i][dir_idx] == mkey && g_mon_loss_cnt[i][dir_idx] >= 4)
      return;                                    // cap V4 : 4 pertes/mois/direction

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
      return;

   double lots = LotsForRisk(sym, sl_dist, risk_usd);
   if(lots <= 0)
     {
      PrintFormat("ZeusP11 %s: taille min broker > risque cible — trade refusé", sym);
      return;
     }

   bool ok = (sig.direction > 0)
             ? g_trade.Buy(lots, sym, 0.0, sig.stop_loss, 0.0, "zeus-p11")
             : g_trade.Sell(lots, sym, 0.0, sig.stop_loss, 0.0, "zeus-p11");
   if(!ok || g_trade.ResultRetcode() != TRADE_RETCODE_DONE)
     {
      PrintFormat("ZeusP11 %s: ordre rejeté (retcode=%d) — pas de trade",
                  sym, g_trade.ResultRetcode());
      return;
     }

   ulong ticket = FindPositionTicket(sym);        // fiable netting ET hedging
   if(ticket == 0)
     {
      PrintFormat("ZeusP11 %s: position introuvable après fill — VÉRIFIER MANUELLEMENT", sym);
      return;
     }

   double fill = g_trade.ResultPrice();
   if(fill <= 0) fill = price;

   ZeusOpenTrade t;
   ZeusTradeReset(t);
   t.active         = true;
   t.ticket         = ticket;
   t.direction      = sig.direction;
   t.entry_price    = fill;
   t.sl             = sig.stop_loss;
   t.sl_dist        = MathAbs(fill - sig.stop_loss);
   t.lots_initial   = g_trade.ResultVolume();
   t.remaining_frac = 1.0;
   t.opened_at      = sig.formed_at;
   g_trades[i]      = t;

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
      TimeToStruct(ToUTC(g_trades[i].opened_at), dt);
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
