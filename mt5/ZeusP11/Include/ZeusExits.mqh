//+------------------------------------------------------------------+
//| ZeusExits.mqh — échelle de sorties générique (port de la logique |
//| _manage_exits de zeus/live/p11_engine.py).                       |
//|                                                                  |
//| XAU V4     : TP1@1R (0% + SL→BE) · TP2@3R (60%) · TP3@8R (85%)   |
//|              · Runner@20R                                        |
//| Forex TIER : TP1@1.5R (33% + BE) · TP2@5R (cum 70%) · Runner@10R |
//|                                                                  |
//| Résolution pessimiste : le SL/BE est vérifié AVANT les TP sur    |
//| chaque barre, comme le backtest.                                 |
//|                                                                  |
//| CORRECTIF (bug fermetures partielles) : remaining_frac/realized_r|
//| ne sont désormais mis à jour QUE si le broker confirme la        |
//| fermeture (aucune comptabilité optimiste). Le mode de remplissage|
//| est fixé explicitement avant chaque ordre de clôture — plusieurs |
//| brokers rejettent sinon silencieusement les fermetures partielles|
//| (retcode 10030 "Unsupported filling mode"), ce qui laissait      |
//| la position entière courir jusqu'au SL ou au runner en un seul   |
//| bloc au lieu de suivre l'échelle de sorties.                     |
//+------------------------------------------------------------------+
#property strict
#include <Trade\Trade.mqh>

struct ZeusExitParams
  {
   double            tp1_r;
   double            tp1_close_pct;        // fraction fermée à TP1 (0 = BE seul)
   double            tp2_r;
   double            tp2_cumulative_pct;   // fraction CUMULÉE fermée après TP2
   double            tp3_r;                // 0 = pas de TP3
   double            tp3_cumulative_pct;
   double            runner_rr;
  };

void ZeusExitsXauV4(ZeusExitParams &e)
  {
   e.tp1_r = 1.0;  e.tp1_close_pct = 0.0;
   e.tp2_r = 3.0;  e.tp2_cumulative_pct = 0.60;
   e.tp3_r = 8.0;  e.tp3_cumulative_pct = 0.85;
   e.runner_rr = 20.0;
  }

void ZeusExitsForexTiered(ZeusExitParams &e)
  {
   e.tp1_r = 1.5;  e.tp1_close_pct = 0.33;
   e.tp2_r = 5.0;  e.tp2_cumulative_pct = 0.70;
   e.tp3_r = 0.0;  e.tp3_cumulative_pct = 0.0;
   e.runner_rr = 10.0;
  }

// ── Trade ouvert géré par l'EA ───────────────────────────────────────
struct ZeusOpenTrade
  {
   bool              active;
   ulong             ticket;         // ticket de position MT5
   int               direction;      // +1 long / -1 short
   double            entry_price;
   double            sl;             // stop courant (monte à BE)
   double            sl_dist;        // 1R
   double            lots_initial;
   double            remaining_frac; // fraction restante RÉELLE côté broker (1.0 → 0.0)
   double            realized_r;
   bool              tp1_done, tp2_done, tp3_done;  // niveau déjà TENTÉ (pas forcément réussi)
   datetime          opened_at;
  };

void ZeusTradeReset(ZeusOpenTrade &t)
  {
   t.active = false;
   t.ticket = 0;
   t.direction = 0;
   t.entry_price = 0; t.sl = 0; t.sl_dist = 0;
   t.lots_initial = 0; t.remaining_frac = 0; t.realized_r = 0;
   t.tp1_done = false; t.tp2_done = false; t.tp3_done = false;
   t.opened_at = 0;
  }

double ZeusTradeLevel(const ZeusOpenTrade &t, const double r)
  {
   return t.entry_price + t.direction * r * t.sl_dist;
  }

// ── Fermeture partielle réelle (normalisée au step du symbole) ──────
// Retour : true SEULEMENT si le broker a réellement réduit la position.
// L'APPELANT ne doit créditer remaining_frac/realized_r QUE si ce
// retour est true — sinon la position reste pleine côté broker et la
// comptabilité interne doit rester fidèle à cette réalité (pas de
// double comptage, pas de position "fantôme" qu'on croit réduite).
bool ZeusPartialClose(CTrade &trade, const string symbol,
                      const ZeusOpenTrade &t, const double frac)
  {
   if(frac <= 0.0)
      return true;                       // rien à fermer : pas un échec

   double step = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double lots = t.lots_initial * frac;
   if(step > 0)
      lots = MathFloor(lots / step) * step;

   if(lots < vmin)
     {
      PrintFormat("ZeusP11 %s: partiel visé %.4f lots < minimum broker %.4f — "
                  "REPORTÉ (position gardée pleine, rattrapée au palier suivant)",
                  symbol, t.lots_initial * frac, vmin);
      return false;                       // pas fermé — l'appelant ne décompte rien
     }

   trade.SetTypeFillingBySymbol(symbol);   // évite le rejet "Unsupported filling mode"
   bool ok = trade.PositionClosePartial(t.ticket, lots);
   if(!ok)
     {
      PrintFormat("ZeusP11 %s: fermeture partielle REJETÉE lots=%.4f "
                  "(retcode=%d, %s) — position gardée pleine, retentera au prochain tick",
                  symbol, lots, trade.ResultRetcode(), trade.ResultComment());
     }
   return ok;
  }

// ── Fermeture totale du reliquat (niveau runner) ─────────────────────
// Même principe : ne rapporte succès que si le broker confirme.
bool ZeusFullClose(CTrade &trade, const string symbol, const ulong ticket)
  {
   trade.SetTypeFillingBySymbol(symbol);
   bool ok = trade.PositionClose(ticket);
   if(!ok)
     {
      PrintFormat("ZeusP11 %s: fermeture finale REJETÉE ticket=%I64u "
                  "(retcode=%d, %s) — position reste ouverte, retentera au prochain tick",
                  symbol, ticket, trade.ResultRetcode(), trade.ResultComment());
     }
   return ok;
  }

// ── Gestion des sorties sur une barre M1 close ───────────────────────
// Retour : true si le trade est RÉELLEMENT terminé côté broker
// (pnl_r_out rempli). reason_out : "SL/BE" ou "RUNNER".
bool ZeusManageExits(CTrade &trade, const string symbol,
                     ZeusOpenTrade &t, const ZeusExitParams &e,
                     const double bar_high, const double bar_low,
                     double &pnl_r_out, string &reason_out)
  {
   if(!t.active)
      return false;

   bool is_long = (t.direction > 0);

   // SL / BE d'abord (pessimiste, comme le backtest) — le SL est un
   // ordre attaché côté broker : s'il est touché, la position (le
   // reliquat réel, = remaining_frac) est déjà fermée par le broker.
   bool slhit = is_long ? (bar_low <= t.sl) : (bar_high >= t.sl);
   if(slhit)
     {
      double r_exit = t.direction * (t.sl - t.entry_price) / t.sl_dist;
      t.realized_r += t.remaining_frac * r_exit;
      t.remaining_frac = 0.0;
      pnl_r_out  = t.realized_r;
      reason_out = "SL/BE";
      return true;
     }

   #define ZE_HIT(lvl) (is_long ? (bar_high >= (lvl)) : (bar_low <= (lvl)))

   if(!t.tp1_done && ZE_HIT(ZeusTradeLevel(t, e.tp1_r)))
     {
      t.tp1_done = true;
      if(e.tp1_close_pct > 0)
        {
         if(ZeusPartialClose(trade, symbol, t, e.tp1_close_pct))
           {
            t.realized_r    += e.tp1_close_pct * e.tp1_r;
            t.remaining_frac -= e.tp1_close_pct;
           }
         // échec : remaining_frac inchangé — le palier suivant rattrapera
         // la fraction visée via son propre calcul cumulatif.
        }
      t.sl = t.entry_price;                          // break-even
      trade.PositionModify(t.ticket, t.sl, 0.0);
     }

   if(!t.tp2_done && ZE_HIT(ZeusTradeLevel(t, e.tp2_r)))
     {
      t.tp2_done = true;
      double closed = MathMax(0.0, e.tp2_cumulative_pct - (1.0 - t.remaining_frac));
      if(closed > 0 && ZeusPartialClose(trade, symbol, t, closed))
        {
         t.realized_r    += closed * e.tp2_r;
         t.remaining_frac -= closed;
        }
      // NB (audit) : zeus/live/p11_engine.py::_manage_exits ne remonte PAS le
      // SL après TP2/TP3 (reste au BE fixé à TP1) — voir tr.sl à la ligne 280
      // de ce fichier, jamais réassigné ensuite. C'est la RÉFÉRENCE déclarée
      // de ce port (voir en-tête de fichier) ; le comportement ci-dessous
      // matche donc le moteur live à l'identique. zeus/backtest/sd_simulation.py
      // (utilisé pour valider les performances du portefeuille), lui, FAIT
      // remonter active_sl à tp1 puis tp2 après chaque palier — les deux
      // moteurs Python divergent entre eux sur ce point précis, en amont de
      // tout port MQL5. Ne pas "corriger" ce fichier pour matcher le backtest
      // sans d'abord trancher lequel des deux moteurs Python est la vérité.
     }

   if(e.tp3_r > 0 && !t.tp3_done && ZE_HIT(ZeusTradeLevel(t, e.tp3_r)))
     {
      t.tp3_done = true;
      double closed = MathMax(0.0, e.tp3_cumulative_pct - (1.0 - t.remaining_frac));
      if(closed > 0 && ZeusPartialClose(trade, symbol, t, closed))
        {
         t.realized_r    += closed * e.tp3_r;
         t.remaining_frac -= closed;
        }
     }

   if(ZE_HIT(ZeusTradeLevel(t, e.runner_rr)))
     {
      if(t.remaining_frac <= 0.0)
        {
         // tout avait déjà été fermé par les paliers précédents
         pnl_r_out  = t.realized_r;
         reason_out = "RUNNER";
         return true;
        }
      if(!ZeusFullClose(trade, symbol, t.ticket))
         return false;                    // rejet broker : on retente au prochain tick
      t.realized_r += t.remaining_frac * e.runner_rr;
      t.remaining_frac = 0.0;
      pnl_r_out  = t.realized_r;
      reason_out = "RUNNER";
      return true;
     }

   #undef ZE_HIT
   return false;
  }
//+------------------------------------------------------------------+
