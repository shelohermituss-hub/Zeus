//+------------------------------------------------------------------+
//|                                            ZeusOpenMomentum.mq5   |
//|  Expert Advisor — continuation de momentum sur l'ouverture US     |
//|                                                                   |
//|  Stratégie :                                                      |
//|    1. Mesure le range des InpORMinutes premières minutes après    |
//|       l'ouverture du marché actions US (9h30 ET, DST US géré      |
//|       correctement — voir Include/ZeusOpenRange.mqh).             |
//|    2. Détermine le momentum (sens du close de fin de fenêtre vs   |
//|       le prix d'ouverture) — si le mouvement est trop faible,     |
//|       aucun trade n'est pris ce jour-là.                          |
//|    3. Entre dans le sens du momentum : immédiatement à la clôture |
//|       de la fenêtre (InpRequireBreakout=false, défaut), ou sur    |
//|       cassure confirmée du range (InpRequireBreakout=true).       |
//|    4. SL = extrémité opposée du range, plafonné à InpMaxSLPips.   |
//|    5. Sorties gérées par l'échelle progressive ZeusExits.mqh      |
//|       (même moteur que ZeusP11 — TP1 BE, TP2 partiel, runner).    |
//|    6. Un seul trade par jour, par symbole. Guard propfirm partagé |
//|       avec ZeusP11 (ZeusGuard.mqh, mêmes garanties fail-closed).  |
//|                                                                   |
//|  ⚠️ STRATÉGIE NOUVELLE, NON VALIDÉE : contrairement à ZeusP11      |
//|  (porté depuis un backtest Python déjà audité), cette logique n'a |
//|  PAS de référence de backtest établie. InpTradingEnabled=false et |
//|  InpSignalLogMode=true par défaut — NE PAS activer le trading     |
//|  réel avant validation démo, conformément au protocole du projet  |
//|  (voir README.md).                                                |
//+------------------------------------------------------------------+
#property copyright "Zeus"
#property version   "1.00"
#property strict

#include <Trade\Trade.mqh>
#include "Include\ZeusOpenRange.mqh"
#include "Include\ZeusGuard.mqh"
#include "Include\ZeusExits.mqh"

// ═════════════════════════ INPUTS ═══════════════════════════════════
input group "── Compte & risque ──"
input double InpAccountBalance     = 100000.0;  // Solde initial de référence (USD)
input double InpRiskPerTradePct    = 0.005;     // Risque par trade (0.005 = 0.5%)
input double InpMaxDailyLossPct    = 0.03;      // Arrêt journalier (3%)
input double InpMaxTotalDDPct      = 0.06;      // Arrêt définitif (6%)
input int    InpMaxLossesPerDay    = 2;         // Stop après N pertes/jour
input int    InpFlatHourUTC        = 20;        // Aucune entrée après / clôture forcée (UTC)

input group "── Symboles ──"
input string InpSymbols            = "XAUUSD";  // Liste séparée par virgules (noms broker exacts)
input string InpPipOverrides       = "";        // pip par symbole, ex "XAGUSD=0.01" (défaut: JPY=0.01, autres=0.0001, or=0.1)

input group "── Range d'ouverture US ──"
input int    InpORMinutes          = 15;        // Durée de la fenêtre d'ouverture (minutes)
input double InpMinMomentumPips    = 15.0;      // Mouvement minimum pour valider un momentum
input bool   InpRequireBreakout    = false;     // true = attend la cassure du range avant d'entrer
input double InpMaxSLPips          = 30.0;      // SL maximum accepté (rejet sinon)
input double InpMinSLPips          = 5.0;       // SL minimum accepté (range trop étroit = rejet)

input group "── Sorties (RR) ──"
input double InpTP1R               = 1.0;       // TP1 : passage à BE (R)
input double InpTP1ClosePct        = 0.0;       // fraction fermée à TP1 (0 = BE seul)
input double InpTP2R               = 2.0;       // TP2 (R)
input double InpTP2CumulativePct   = 0.60;      // fraction cumulée fermée après TP2
input double InpRunnerRR           = 4.0;       // Runner final (R)

input group "── Exécution ──"
input long   InpMagic              = 20260810;  // Magic number
input int    InpTimerSeconds       = 15;        // Fréquence de scan
input int    InpSlippagePoints     = 20;        // Déviation max (points)
input int    InpServerUTCOffsetH   = -99;       // Décalage serveur→UTC (heures), -99 = auto
input bool   InpSignalLogMode      = true;      // true = log CSV, AUCUN ordre (défaut : validation)
input bool   InpTradingEnabled     = false;     // false = observation seule (défaut : sécurité)

// ═════════════════════════ ÉTAT ═════════════════════════════════════
#define ZOM_MAX_SYMBOLS 8
#define ZOM_M1_WINDOW   120       // largement suffisant : la fenêtre d'ouverture ne dépasse jamais ~30 barres

string                 g_symbols[ZOM_MAX_SYMBOLS];
int                    g_n_symbols = 0;
double                 g_pip[ZOM_MAX_SYMBOLS];
ZeusORState            g_or_state[ZOM_MAX_SYMBOLS];
ZeusExitParams         g_exit_params[ZOM_MAX_SYMBOLS];
ZeusOpenTrade          g_trades[ZOM_MAX_SYMBOLS];
datetime               g_last_m1_processed[ZOM_MAX_SYMBOLS];
bool                   g_warmed_up[ZOM_MAX_SYMBOLS];

ZeusGuardConfig        g_guard_cfg;
ZeusGuardState         g_guard;
CTrade                 g_trade;
int                    g_log_handle = INVALID_HANDLE;

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
   if(StringFind(sym, "XAU") >= 0) return 0.1;    // or : 1 pip = 0.1$ (convention retenue ce projet)
   if(StringFind(sym, "JPY") >= 0) return 0.01;
   return 0.0001;
  }

// Décalage serveur→UTC recalculé À CHAQUE APPEL (suit le DST du broker) —
// distinct du DST US utilisé par ZeusOpenRange.mqh pour l'heure d'ouverture
// elle-même (ce sont deux fuseaux/règles différents).
int CurrentUTCOffsetSec()
  {
   if(InpServerUTCOffsetH != -99)
      return InpServerUTCOffsetH * 3600;
   return (int)(TimeCurrent() - TimeGMT());
  }

datetime ToUTC(const datetime server_time)
  {
   return server_time - CurrentUTCOffsetSec();
  }

// Dimensionnement par risque réel (voir ZeusP11.mq5 pour le détail du
// garde-fou anti-tick_value corrompu — même logique, reprise à l'identique).
double LotsForRisk(const string sym, const double sl_dist, const double risk_usd)
  {
   double tick_size  = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE_LOSS);
   if(tick_value <= 0)
      tick_value = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
   if(tick_size <= 0 || tick_value <= 0)
      return 0.0;
   double loss_per_lot = (sl_dist / tick_size) * tick_value;
   if(loss_per_lot <= 0)
      return 0.0;

   if(SymbolInfoString(sym, SYMBOL_CURRENCY_PROFIT) == AccountInfoString(ACCOUNT_CURRENCY))
     {
      double contract_size = SymbolInfoDouble(sym, SYMBOL_TRADE_CONTRACT_SIZE);
      if(contract_size > 0)
        {
         double loss_per_lot_direct = sl_dist * contract_size;
         if(loss_per_lot_direct > 0
            && MathAbs(loss_per_lot - loss_per_lot_direct) > loss_per_lot_direct * 0.05)
           {
            PrintFormat("ZeusOpenMomentum %s: tick_value_loss incohérent (perte/lot=%.2f$ vs "
                        "%.2f$ attendu via contract_size=%.2f) — calcul direct utilisé",
                        sym, loss_per_lot, loss_per_lot_direct, contract_size);
            loss_per_lot = loss_per_lot_direct;
           }
        }
     }
   double lots = risk_usd / loss_per_lot;
   double step = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);
   if(step > 0)
      lots = MathFloor(lots / step) * step;   // floor : jamais > risque cible
   if(lots < vmin)
      return 0.0;                             // risque min > cible → refus

   // Garde-fou a priori : le pire cas réel (lot arrondi au minimum broker)
   // ne doit pas dépasser ~1.5× le risque visé (marge pour l'arrondi au pas
   // de volume minimum, qui peut légèrement gonfler le risque réel).
   double worst_case_loss = vmin * loss_per_lot;
   if(worst_case_loss > risk_usd * 1.5)
     {
      PrintFormat("ZeusOpenMomentum %s: REFUS — même au lot minimum (%.4f), la perte "
                  "au SL (%.2f$) dépasse largement le risque visé (%.2f$). "
                  "Vérifier tick_value/contract_size de ce symbole chez le broker.",
                  sym, vmin, worst_case_loss, risk_usd);
      return 0.0;
     }
   return MathMin(vmax, lots);
  }

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

double ZeusPositionRealizedUSD(const ulong ticket)
  {
   if(!HistorySelectByPosition((long)ticket))
      return 0.0;
   double total = 0.0;
   int n = HistoryDealsTotal();
   for(int k = 0; k < n; k++)
     {
      ulong deal = HistoryDealGetTicket(k);
      if(deal == 0) continue;
      long entry = HistoryDealGetInteger(deal, DEAL_ENTRY);
      if(entry != DEAL_ENTRY_OUT && entry != DEAL_ENTRY_OUT_BY)
         continue;
      total += HistoryDealGetDouble(deal, DEAL_PROFIT)
             + HistoryDealGetDouble(deal, DEAL_SWAP)
             + HistoryDealGetDouble(deal, DEAL_COMMISSION);
     }
   return total;
  }

// ═════════════════════════ INIT ═════════════════════════════════════
int OnInit()
  {
   ZeusGuardDefaults(g_guard_cfg);
   g_guard_cfg.max_daily_loss_pct = InpMaxDailyLossPct;
   g_guard_cfg.max_total_dd_pct   = InpMaxTotalDDPct;
   g_guard_cfg.risk_per_trade_pct = InpRiskPerTradePct;
   g_guard_cfg.max_losses_per_day = InpMaxLossesPerDay;
   g_guard_cfg.flat_hour_utc      = InpFlatHourUTC;
   if(!ZeusGuardConfigValid(g_guard_cfg))
     {
      Print("ZeusOpenMomentum: configuration de risque INVALIDE — refus de démarrer (fail closed)");
      return INIT_PARAMETERS_INCORRECT;
     }
   ZeusGuardInit(g_guard, InpAccountBalance);

   if(InpORMinutes < 1 || InpORMinutes > 120)
     {
      Print("ZeusOpenMomentum: InpORMinutes hors plage raisonnable — refus de démarrer");
      return INIT_PARAMETERS_INCORRECT;
     }
   if(InpMaxSLPips <= InpMinSLPips)
     {
      Print("ZeusOpenMomentum: InpMaxSLPips <= InpMinSLPips — configuration incohérente");
      return INIT_PARAMETERS_INCORRECT;
     }

   string parts[];
   g_n_symbols = StringSplit(InpSymbols, ',', parts);
   if(g_n_symbols > ZOM_MAX_SYMBOLS) g_n_symbols = ZOM_MAX_SYMBOLS;
   for(int i = 0; i < g_n_symbols; i++)
     {
      StringTrimLeft(parts[i]);
      StringTrimRight(parts[i]);
      g_symbols[i] = parts[i];
      if(!SymbolSelect(g_symbols[i], true))
        {
         PrintFormat("ZeusOpenMomentum: symbole %s introuvable chez le broker — refus de démarrer",
                     g_symbols[i]);
         return INIT_PARAMETERS_INCORRECT;
        }
      g_pip[i] = PipForSymbol(g_symbols[i]);
      ZeusORStateInit(g_or_state[i]);
      ZeusTradeReset(g_trades[i]);
      g_exit_params[i].tp1_r = InpTP1R;
      g_exit_params[i].tp1_close_pct = InpTP1ClosePct;
      g_exit_params[i].tp2_r = InpTP2R;
      g_exit_params[i].tp2_cumulative_pct = InpTP2CumulativePct;
      g_exit_params[i].tp3_r = 0.0;
      g_exit_params[i].tp3_cumulative_pct = 0.0;
      g_exit_params[i].runner_rr = InpRunnerRR;
      g_last_m1_processed[i] = 0;
      g_warmed_up[i] = false;
     }

   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(InpSlippagePoints);

   if(InpSignalLogMode)
     {
      string fname = StringFormat("ZeusOpenMomentum_signals_%s.csv", TimeToString(TimeCurrent(), TIME_DATE));
      g_log_handle = FileOpen(fname, FILE_WRITE | FILE_CSV | FILE_ANSI, ';');
      if(g_log_handle != INVALID_HANDLE)
         FileWrite(g_log_handle, "time_utc", "symbol", "direction", "or_open", "or_high", "or_low",
                   "sl_pips", "entry_price");
     }

   if(!InpTradingEnabled)
      Print("ZeusOpenMomentum: InpTradingEnabled=false — mode OBSERVATION, aucun ordre envoyé.");
   if(InpSignalLogMode)
      Print("ZeusOpenMomentum: InpSignalLogMode=true — signaux journalisés en CSV, aucun ordre envoyé.");

   EventSetTimer(InpTimerSeconds);
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   if(g_log_handle != INVALID_HANDLE)
      FileClose(g_log_handle);
  }

void OnTimer()
  {
   for(int i = 0; i < g_n_symbols; i++)
      ProcessSymbol(i);
  }

void OnTick()
  {
   // Les sorties (SL/BE/TP/runner) sont vérifiées à chaque tick pour une
   // résolution intrabar fine ; les nouveaux signaux d'entrée, eux, ne
   // s'évaluent qu'à la clôture d'une barre M1 (dans ProcessSymbol, via
   // le timer) — inutile de les tester à chaque tick.
   for(int i = 0; i < g_n_symbols; i++)
      ManageOpenTrade(i);
  }

// ═════════════════════════ CŒUR ═════════════════════════════════════
void ProcessSymbol(const int i)
  {
   string sym = g_symbols[i];

   MqlRates m1[];
   ArraySetAsSeries(m1, false);
   int copied = CopyRates(sym, PERIOD_M1, 1, ZOM_M1_WINDOW, m1);
   if(copied < 5)
      return;

   datetime last_bar_time = m1[copied - 1].time;
   if(last_bar_time == g_last_m1_processed[i])
      return;                                       // pas de nouvelle barre M1 close

   // Rattrapage : traite chaque barre close non encore vue, dans l'ordre
   // (comme ZeusP11 — une barre manquée doit quand même être vue, sinon
   // la fenêtre d'ouverture pourrait être mal construite après une
   // coupure réseau/redémarrage du terminal).
   int first_new = 0;
   if(g_last_m1_processed[i] != 0)
     {
      first_new = copied;
      for(int k = copied - 1; k >= 0; k--)
        {
         if(m1[k].time <= g_last_m1_processed[i]) { first_new = k + 1; break; }
         if(k == 0) first_new = 0;
        }
     }
   else
     {
      // Premier passage : ne rattrape pas tout l'historique (fenêtre
      // d'ouverture potentiellement déjà entamée/finie) — ne considère
      // que la dernière barre, pour ne pas générer un signal basé sur
      // un range partiel construit rétroactivement.
      first_new = copied - 1;
      g_warmed_up[i] = false;
     }

   for(int k = first_new; k < copied; k++)
     {
      datetime ts_utc = ToUTC(m1[k].time);
      int direction = 0;
      double sl_price = 0.0;
      bool fired = ZeusOpenRangeOnBar(g_or_state[i], ts_utc,
                                       m1[k].open, m1[k].high, m1[k].low, m1[k].close,
                                       InpORMinutes, InpMinMomentumPips, g_pip[i],
                                       InpRequireBreakout, direction, sl_price);
      if(fired)
         TryOpenTrade(i, sym, direction, sl_price, ts_utc, m1[k].close,
                      g_or_state[i].or_open, g_or_state[i].or_high, g_or_state[i].or_low);
     }

   g_last_m1_processed[i] = last_bar_time;
  }

void TryOpenTrade(const int i, const string sym, const int direction, const double sl_ref,
                  const datetime ts_utc, const double bar_close,
                  const double or_open, const double or_high, const double or_low)
  {
   if(g_trades[i].active)
     {
      PrintFormat("ZeusOpenMomentum %s: signal ignoré — position déjà ouverte sur ce symbole", sym);
      return;
     }
   if(!ZeusGuardCanTrade(g_guard, g_guard_cfg, ts_utc))
      return;

   MqlTick tick;
   if(!SymbolInfoTick(sym, tick))
      return;
   double price   = (direction > 0) ? tick.ask : tick.bid;
   double pv      = g_pip[i];
   double sl_dist_raw = MathAbs(price - sl_ref);
   double sl_pips = sl_dist_raw / pv;

   if(sl_pips > InpMaxSLPips)
     {
      PrintFormat("ZeusOpenMomentum %s: SL trop large (%.1f pips > max %.1f) — signal refusé",
                  sym, sl_pips, InpMaxSLPips);
      return;
     }
   if(sl_pips < InpMinSLPips)
     {
      PrintFormat("ZeusOpenMomentum %s: SL trop étroit (%.1f pips < min %.1f, range d'ouverture "
                  "anormalement plat) — signal refusé", sym, sl_pips, InpMinSLPips);
      return;
     }

   double sl_price = (direction > 0) ? price - sl_dist_raw : price + sl_dist_raw;

   if(InpSignalLogMode && g_log_handle != INVALID_HANDLE)
     {
      FileWrite(g_log_handle, TimeToString(ts_utc, TIME_DATE | TIME_SECONDS), sym,
                (direction > 0 ? "long" : "short"),
                DoubleToString(or_open, 5), DoubleToString(or_high, 5), DoubleToString(or_low, 5),
                DoubleToString(sl_pips, 1), DoubleToString(price, 5));
      FileFlush(g_log_handle);
     }

   if(!InpTradingEnabled || InpSignalLogMode)
      return;                                        // observation / log seul : pas d'ordre

   double risk_usd = ZeusGuardRiskUSD(g_guard, g_guard_cfg);
   double lots = LotsForRisk(sym, sl_dist_raw, risk_usd);
   if(lots <= 0)
     {
      PrintFormat("ZeusOpenMomentum %s: taille min broker > risque cible — trade refusé", sym);
      return;
     }

   bool ok = (direction > 0)
             ? g_trade.Buy(lots, sym, 0.0, sl_price, 0.0, "zeus-open-momentum")
             : g_trade.Sell(lots, sym, 0.0, sl_price, 0.0, "zeus-open-momentum");
   if(!ok || g_trade.ResultRetcode() != TRADE_RETCODE_DONE)
     {
      PrintFormat("ZeusOpenMomentum %s: ordre rejeté (retcode=%d) — pas de trade",
                  sym, g_trade.ResultRetcode());
      return;
     }

   ulong ticket = FindPositionTicket(sym);
   if(ticket == 0)
     {
      PrintFormat("ZeusOpenMomentum %s: position introuvable après fill — VÉRIFIER MANUELLEMENT", sym);
      return;
     }

   double fill = g_trade.ResultPrice();
   if(fill <= 0) fill = price;

   ZeusTradeReset(g_trades[i]);
   g_trades[i].active      = true;
   g_trades[i].ticket      = ticket;
   g_trades[i].direction   = direction;
   g_trades[i].entry_price = fill;
   g_trades[i].sl          = sl_price;
   g_trades[i].sl_dist     = MathAbs(fill - sl_price);
   g_trades[i].lots_initial = lots;
   g_trades[i].remaining_frac = 1.0;
   g_trades[i].opened_at   = TimeCurrent();

   PrintFormat("ZeusOpenMomentum %s: ouverture %s lots=%.2f prix=%.5f SL=%.5f (%.1f pips)",
               sym, (direction > 0 ? "LONG" : "SHORT"), lots, fill, sl_price, sl_pips);
  }

void ManageOpenTrade(const int i)
  {
   if(!g_trades[i].active)
      return;
   string sym = g_symbols[i];

   MqlTick tick;
   if(!SymbolInfoTick(sym, tick))
      return;
   double bar_high = MathMax(tick.bid, tick.ask);
   double bar_low  = MathMin(tick.bid, tick.ask);

   double pnl_r = 0.0;
   string reason = "";
   bool closed = ZeusManageExits(g_trade, sym, g_trades[i], g_exit_params[i],
                                 bar_high, bar_low, pnl_r, reason);

   // Clôture forcée à l'heure de guard (flat_hour_utc) même sans avoir
   // touché SL/TP/runner — cohérent avec ZeusGuardCanTrade qui interdit
   // déjà toute NOUVELLE entrée après cette heure.
   datetime ts_utc = ToUTC(TimeCurrent());
   MqlDateTime dt; TimeToStruct(ts_utc, dt);
   if(!closed && g_trades[i].active && dt.hour >= InpFlatHourUTC)
     {
      if(ZeusFullClose(g_trade, sym, g_trades[i].ticket))
        {
         double r_exit = g_trades[i].direction *
                         (((g_trades[i].direction > 0) ? tick.bid : tick.ask) - g_trades[i].entry_price)
                         / g_trades[i].sl_dist;
         g_trades[i].realized_r += g_trades[i].remaining_frac * r_exit;
         g_trades[i].remaining_frac = 0.0;
         pnl_r  = g_trades[i].realized_r;
         reason = "FLAT_HOUR";
         closed = true;
        }
     }

   if(!closed)
      return;

   double pnl_usd = ZeusPositionRealizedUSD(g_trades[i].ticket);
   ZeusGuardOnTradeClosed(g_guard, g_guard_cfg, ts_utc, pnl_usd);
   PrintFormat("ZeusOpenMomentum %s: position fermée (%s) R=%.2f PnL=%.2f$ équité=%.2f$",
               sym, reason, pnl_r, pnl_usd, g_guard.equity);

   ZeusTradeReset(g_trades[i]);
  }
//+------------------------------------------------------------------+
