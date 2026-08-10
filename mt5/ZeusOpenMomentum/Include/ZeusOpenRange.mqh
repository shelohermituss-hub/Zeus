//+------------------------------------------------------------------+
//| ZeusOpenRange.mqh — détection "Momentum de l'Open US"             |
//|                                                                    |
//| US Open = 9h30 ET (ouverture actions US / futures), converti en   |
//| UTC selon la règle DST DES ÉTATS-UNIS (2e dimanche de mars ->     |
//| 1er dimanche de novembre) — PAS la même règle que l'UE/UK, donc   |
//| PAS le simple décalage serveur→UTC utilisé ailleurs dans ce repo   |
//| (CurrentUTCOffsetSec() gère le DST du BROKER, pas celui du marché |
//| US ciblé par cette stratégie).                                    |
//|                                                                    |
//| Logique (une seule tentative par jour, par symbole) :              |
//|  1. Range d'ouverture = [Open US, Open US + InpORMinutes]          |
//|     — haut/bas/prix d'ouverture de cette fenêtre.                  |
//|  2. Momentum figé UNE FOIS à la clôture de la fenêtre : sens du    |
//|     close vs le prix d'ouverture de la fenêtre, si le mouvement    |
//|     dépasse un seuil minimum (sinon : pas de trade aujourd'hui).   |
//|  3. Entrée :                                                       |
//|     - InpRequireBreakout=false (défaut, "continuation immédiate") |
//|       → entrée dès la clôture de la fenêtre, dans le sens du       |
//|         momentum déjà établi.                                     |
//|     - InpRequireBreakout=true → attend que le prix casse le bord   |
//|       du range dans CE MÊME sens avant d'entrer (confirmation      |
//|       plus stricte, moins de faux départs, entrée plus tardive).   |
//|  4. SL = extrémité opposée du range (l'appelant applique un        |
//|     plafond InpMaxSLPips).                                         |
//+------------------------------------------------------------------+
#property strict

enum ENUM_OR_PHASE
  {
   OR_PHASE_WAITING  = 0,   // avant l'ouverture US du jour
   OR_PHASE_BUILDING = 1,   // dans la fenêtre d'ouverture, range en construction
   OR_PHASE_ARMED    = 2,   // range terminé, momentum figé, attend la cassure (si requise)
   OR_PHASE_DONE     = 3    // trade pris (ou aucun momentum net) — rien de plus aujourd'hui
  };

struct ZeusORState
  {
   int               phase;
   int               today_key;      // AAAAMMJJ (UTC) du jour US courant
   datetime          or_start;       // Open US (UTC) du jour
   datetime          or_end;         // or_start + InpORMinutes*60
   double            or_open;        // prix au tout début de la fenêtre
   double            or_high;
   double            or_low;
   int               momentum;       // 0=indéterminé, +1=haussier, -1=baissier (figé à or_end)
  };

void ZeusORStateInit(ZeusORState &st)
  {
   st.phase     = OR_PHASE_WAITING;
   st.today_key = 0;
   st.or_start  = 0; st.or_end = 0;
   st.or_open   = 0; st.or_high = 0; st.or_low = 0;
   st.momentum  = 0;
  }

// ── Règle DST US (en vigueur depuis 2007, Energy Policy Act 2005) ────
// 2e dimanche de mars 07:00 UTC (= 02:00 EST, avant le changement) ->
// 1er dimanche de novembre 06:00 UTC (= 02:00 EDT, avant le changement).
datetime ZeusNthSundayUTC(const int year, const int month, const int n, const int hour_utc)
  {
   MqlDateTime dt;
   dt.year = year; dt.mon = month; dt.day = 1;
   dt.hour = 0; dt.min = 0; dt.sec = 0;
   datetime first = StructToTime(dt);
   MqlDateTime fdt;
   TimeToStruct(first, fdt);
   int dow          = fdt.day_of_week;             // 0=dimanche ... 6=samedi
   int to_first_sun = (dow == 0) ? 0 : (7 - dow);
   dt.day  = 1 + to_first_sun + (n - 1) * 7;
   dt.hour = hour_utc;
   return StructToTime(dt);
  }

bool ZeusIsUSDaylightTime(const datetime utc_time)
  {
   MqlDateTime dt;
   TimeToStruct(utc_time, dt);
   datetime dst_start = ZeusNthSundayUTC(dt.year, 3, 2, 7);
   datetime dst_end   = ZeusNthSundayUTC(dt.year, 11, 1, 6);
   return (utc_time >= dst_start && utc_time < dst_end);
  }

// Heure d'ouverture US (9h30 ET) en UTC, pour le jour UTC de ts_utc.
// Le test DST est fait à MIDI UTC de ce jour-là (pas à 00:00) pour éviter
// toute ambiguïté sur le jour même du changement d'heure — la séance US
// (13h30/14h30 UTC) a de toute façon lieu après ce changement.
datetime ZeusUSOpenTimeUTC(const datetime ts_utc)
  {
   MqlDateTime dt;
   TimeToStruct(ts_utc, dt);
   dt.hour = 12; dt.min = 0; dt.sec = 0;
   datetime midday = StructToTime(dt);
   bool edt = ZeusIsUSDaylightTime(midday);
   dt.hour = edt ? 13 : 14;
   dt.min  = 30;
   return StructToTime(dt);
  }

int ZeusORDayKeyUTC(const datetime ts_utc)
  {
   MqlDateTime dt;
   TimeToStruct(ts_utc, dt);
   return dt.year * 10000 + dt.mon * 100 + dt.day;
  }

// ── Mise à jour sur une barre M1 UTC close ───────────────────────────
// Retourne true si un NOUVEAU signal d'entrée doit être émis sur CETTE
// barre (le range ne redéclenche jamais deux fois le même jour).
bool ZeusOpenRangeOnBar(ZeusORState &st, const datetime ts_utc,
                        const double bar_open, const double bar_high,
                        const double bar_low, const double bar_close,
                        const int or_minutes, const double min_momentum_pips,
                        const double pip_size, const bool require_breakout,
                        int &direction_out, double &sl_out)
  {
   MqlDateTime dt;
   TimeToStruct(ts_utc, dt);
   if(dt.day_of_week == 0 || dt.day_of_week == 6)
      return false;                                // week-end : pas de séance US

   int day = ZeusORDayKeyUTC(ts_utc);
   if(day != st.today_key)
     {
      // Nouveau jour : réinitialise et recalcule l'heure d'ouverture US
      // (peut différer d'hier si on traverse un changement DST US).
      st.today_key = day;
      st.or_start  = ZeusUSOpenTimeUTC(ts_utc);
      st.or_end    = st.or_start + or_minutes * 60;
      st.or_open   = 0; st.or_high = 0; st.or_low = 0;
      st.momentum  = 0;
      st.phase     = OR_PHASE_WAITING;
     }

   if(st.phase == OR_PHASE_DONE)
      return false;                                 // déjà traité aujourd'hui

   if(ts_utc < st.or_start)
      return false;                                 // pas encore l'heure

   if(ts_utc < st.or_end)
     {
      // Dans la fenêtre : construit le range.
      if(st.phase == OR_PHASE_WAITING)
        {
         st.phase   = OR_PHASE_BUILDING;
         st.or_open = bar_open;
         st.or_high = bar_high;
         st.or_low  = bar_low;
        }
      else
        {
         if(bar_high > st.or_high) st.or_high = bar_high;
         if(bar_low  < st.or_low)  st.or_low  = bar_low;
        }
      return false;
     }

   // ts_utc >= or_end : la fenêtre vient de se terminer (ou est déjà finie).
   if(st.phase == OR_PHASE_BUILDING)
     {
      if(st.or_open <= 0)
        {
         // L'EA a démarré APRÈS le début de la fenêtre : pas de vrai
         // or_open capturé, le range serait faux — abandon propre du jour
         // plutôt qu'un signal basé sur des données incomplètes.
         st.phase = OR_PHASE_DONE;
         return false;
        }
      double move_pips = (bar_close - st.or_open) / pip_size;
      if(MathAbs(move_pips) < min_momentum_pips)
        {
         st.phase = OR_PHASE_DONE;                  // pas de momentum net : rien aujourd'hui
         return false;
        }
      st.momentum = (move_pips > 0) ? 1 : -1;
      st.phase    = OR_PHASE_ARMED;

      if(!require_breakout)
        {
         direction_out = st.momentum;
         sl_out        = (st.momentum > 0) ? st.or_low : st.or_high;
         st.phase      = OR_PHASE_DONE;
         return true;
        }
      return false;
     }

   if(st.phase == OR_PHASE_ARMED)
     {
      bool broke = (st.momentum > 0) ? (bar_close > st.or_high) : (bar_close < st.or_low);
      if(broke)
        {
         direction_out = st.momentum;
         sl_out        = (st.momentum > 0) ? st.or_low : st.or_high;
         st.phase      = OR_PHASE_DONE;
         return true;
        }
      return false;                                 // pas encore cassé — le guard
                                                      // (flat_hour_utc) borne l'attente
     }

   return false;
  }
//+------------------------------------------------------------------+
