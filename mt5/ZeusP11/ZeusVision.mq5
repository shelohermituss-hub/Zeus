//+------------------------------------------------------------------+
//| ZeusVision.mq5 — indicateur visuel Zones Supply/Demand + Wyckoff  |
//|                                                                    |
//| Réutilise SANS AUCUNE MODIFICATION la détection déjà validée de   |
//| ZeusP11 (Include/ZeusSignals.mqh -> ZeusZones.mqh + ZeusWyckoff.mqh)|
//| pour tracer sur le graphique exactement ce que la stratégie V4    |
//| voit :                                                             |
//|   - Zones Supply/Demand M15 (resamplées depuis M1, comme en prod, |
//|     jamais CopyRates M15 direct — parité stricte avec ZeusP11) —  |
//|     PRIORITÉ VISUELLE : gros mots "DEMANDE"/"OFFRE" collés au     |
//|     prix actuel, peu de zones affichées, seulement les valides.   |
//|   - Patterns Wyckoff Accumulation -> Manipulation -> MSS détectés |
//|     sur M1 (Spring/Upthrust + confirmation) — désactivés par      |
//|     défaut (InpShowWyckoff=false), à activer une fois les zones   |
//|     bien comprises.                                                |
//|                                                                    |
//| Multi-timeframe : la détection tourne toujours sur le couple      |
//| M1/M15 que le bot trade réellement, mais l'indicateur peut être   |
//| posé sur N'IMPORTE QUEL graphique (M1, M5, M15, H1, H4, ...) —    |
//| les objets sont positionnés par prix/temps absolus, donc visibles |
//| quelle que soit la résolution d'affichage choisie.                |
//|                                                                    |
//| Ne modifie AUCUN fichier de la stratégie de production — lecture  |
//| seule sur ZeusZones.mqh / ZeusWyckoff.mqh / ZeusSignals.mqh.      |
//+------------------------------------------------------------------+
#property copyright "Zeus"
#property strict
#property indicator_chart_window
#property indicator_buffers 1
#property indicator_plots   0

#include "Include/ZeusSignals.mqh"

// ── Inputs ────────────────────────────────────────────────────────
enum ENUM_ZV_ASSET { ZV_AUTO, ZV_XAUUSD, ZV_FOREX };

input group "Général"
input ENUM_ZV_ASSET InpAssetClass        = ZV_AUTO;  // Auto = détecte XAUUSD/GOLD dans le nom du symbole
input int           InpM1Window          = 3000;     // Barres M1 chargées (= fenêtre production ZP11_M1_WINDOW)
input int           InpWyckoffScanBars   = 1500;     // Profondeur de scan Wyckoff (barres M1)

input group "Zones Supply/Demand (M15, resamplé depuis M1)"
input bool          InpShowZones            = true;
input double        InpMinZoneScoreToShow   = 4.0;   // Ne montre que les zones déjà correctes (0 = toutes)
input bool          InpShowMitigatedZones   = false;  // false = ne garde que les zones encore valides
input int           InpMaxZonesShown        = 12;     // Peu de zones affichées = lecture immédiate
input double        InpMinZoneHeightPct     = 0.05;   // Hauteur visuelle mini (% du prix) — évite les zones "fil de fer"
input color         InpDemandColor          = clrLightSkyBlue;   // remplissage zone de demande (achat)
input color         InpSupplyColor          = clrLightPink;       // remplissage zone d'offre (vente)
input color         InpDemandLabelColor     = clrNavy;            // texte "DEMANDE" (contraste sur fond clair)
input color         InpSupplyLabelColor     = clrDarkRed;         // texte "OFFRE"
input color         InpMitigatedColor       = clrGainsboro;

input group "Wyckoff (M1) — désactivé par défaut, active une fois les zones bien comprises"
input bool          InpShowWyckoff           = false;
input double        InpMinWyckoffScoreToShow = 0.0;   // 0 = tous les patterns détectés (score 0-10)
input int           InpMaxWyckoffShown       = 40;
input color         InpDemandWyckoffColor    = clrDodgerBlue;
input color         InpSupplyWyckoffColor    = clrOrangeRed;
input color         InpAccumBoxColor         = clrGoldenrod;

input group "Seuils de signal production (surlignage [SIGNAL])"
input double        InpXauWSLong  = 5.9;   // ZeusSignalDefaultsXau.min_wyckoff_score_long
input double        InpXauWSShort = 8.5;   // ZeusSignalDefaultsXau.min_wyckoff_score_short
input double        InpFxWSLong   = 7.5;   // ZeusSignalDefaultsForex.min_wyckoff_score_long
input double        InpFxWSShort  = 6.5;   // ZeusSignalDefaultsForex.min_wyckoff_score_short

// ── État ──────────────────────────────────────────────────────────
#define ZV_PREFIX_ZONE "ZV_Zone_"
#define ZV_PREFIX_WY   "ZV_Wy_"

double                DummyBuffer[];
ZeusSymbolSignalState g_state;
ZeusZoneParams        g_zone_params;
ZeusWyckoffParams     g_wy_params;
bool                  g_is_xau;
double                g_ws_long, g_ws_short;
datetime              g_last_processed = 0;

//+------------------------------------------------------------------+
int OnInit()
  {
   SetIndexBuffer(0, DummyBuffer, INDICATOR_CALCULATIONS);
   ArraySetAsSeries(DummyBuffer, false);

   ZeusSignalStateInit(g_state);
   ZeusZoneParamsDefaults(g_zone_params);

   g_is_xau = (InpAssetClass == ZV_XAUUSD) ||
              (InpAssetClass == ZV_AUTO &&
               (StringFind(_Symbol, "XAU") >= 0 || StringFind(_Symbol, "GOLD") >= 0));

   if(g_is_xau)
     {
      ZeusWyckoffDefaultsXau(g_wy_params);
      g_ws_long  = InpXauWSLong;
      g_ws_short = InpXauWSShort;
     }
   else
     {
      ZeusWyckoffDefaultsForex(g_wy_params);
      g_ws_long  = InpFxWSLong;
      g_ws_short = InpFxWSShort;
     }

   IndicatorSetString(INDICATOR_SHORTNAME, "ZeusVision (" + (g_is_xau ? "XAU" : "Forex") + ")");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   ObjectsDeleteAll(0, ZV_PREFIX_ZONE);
   ObjectsDeleteAll(0, ZV_PREFIX_WY);
  }

//+------------------------------------------------------------------+
int OnCalculate(const int rates_total, const int prev_calculated,
                const datetime &time[], const double &open[],
                const double &high[], const double &low[], const double &close[],
                const long &tick_volume[], const long &volume[], const int &spread[])
  {
   // Fenêtre M1 close, chargée indépendamment du graphique hôte —
   // même logique que ZeusP11.mq5::ProcessSymbol (fidélité de calcul).
   MqlRates m1[];
   ArraySetAsSeries(m1, false);
   int copied = CopyRates(_Symbol, PERIOD_M1, 1, InpM1Window, m1);
   if(copied < 300)
      return(rates_total);

   datetime m1_time[]; double m1_open[], m1_high[], m1_low[], m1_close[];
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

   if(m1_time[n1 - 1] == g_last_processed)
      return(rates_total);            // pas de nouvelle barre M1 close

   // M15 resamplé depuis la MÊME fenêtre M1 (jamais CopyRates M15 —
   // parité stricte avec ZeusP11 / le moteur Python de référence).
   datetime t15[]; double o15[], h15[], l15[], c15[];
   int n15 = 0;
   ZeusResampleM15(m1_time, m1_open, m1_high, m1_low, m1_close, n1, t15, o15, h15, l15, c15, n15);
   if(n15 < 60)
      return(rates_total);

   ZeusSDZone fresh[];
   int nf = ZeusDetectZones(t15, o15, h15, l15, c15, n15, g_zone_params, fresh);
   ZeusMergeZones(g_state, fresh, nf);

   // Rattrapage touches/mitigation barre par barre, dans l'ordre —
   // identique au chemin de production (l'ordre affecte le résultat).
   int first_new = 0;
   if(g_last_processed != 0)
     {
      first_new = n1;
      for(int k = n1 - 1; k >= 0; k--)
        {
         if(m1_time[k] <= g_last_processed) { first_new = k + 1; break; }
         if(k == 0) first_new = 0;
        }
     }
   for(int k = first_new; k < n1; k++)
      ZeusUpdateZonesOnBar(g_state, m1_low[k], m1_high[k], m1_close[k], m1_time[k]);

   g_last_processed = m1_time[n1 - 1];

   if(InpShowZones)
      DrawZones();
   if(InpShowWyckoff)
      DrawWyckoff(m1_time, m1_open, m1_high, m1_low, m1_close, n1);

   ChartRedraw(0);
   return(rates_total);
  }

//+------------------------------------------------------------------+
//| Zones Supply/Demand — rectangles + étiquette de score             |
//+------------------------------------------------------------------+
void DrawZones()
  {
   ObjectsDeleteAll(0, ZV_PREFIX_ZONE);

   // Tri par formed_at décroissant (les plus récentes en premier) —
   // insertion, n_zones reste petit (<= ZS_MAX_ZONES).
   int idx[];
   ArrayResize(idx, g_state.n_zones);
   for(int k = 0; k < g_state.n_zones; k++)
      idx[k] = k;
   for(int a = 1; a < g_state.n_zones; a++)
     {
      int key = idx[a];
      int b   = a - 1;
      while(b >= 0 && g_state.zones[idx[b]].formed_at < g_state.zones[key].formed_at)
        { idx[b + 1] = idx[b]; b--; }
      idx[b + 1] = key;
     }

   // g_last_processed (dernière barre M1 close) plutôt que TimeCurrent() :
   // évite d'étendre les zones/étiquettes dans une zone sans bougies
   // (week-end, hors session) où elles seraient invisibles à l'écran.
   datetime now_time = g_last_processed;
   int shown = 0;
   for(int a = 0; a < g_state.n_zones && shown < InpMaxZonesShown; a++)
     {
      ZeusSDZone z = g_state.zones[idx[a]];
      if(z.is_mitigated && !InpShowMitigatedZones)
         continue;
      double total = ZeusZoneTotal(z);
      if(total < InpMinZoneScoreToShow)
         continue;

      bool   is_demand = (z.side == PIVOT_DEMAND);
      color  col       = z.is_mitigated ? InpMitigatedColor : (is_demand ? InpDemandColor : InpSupplyColor);
      color  lbl_col   = z.is_mitigated ? InpMitigatedColor : (is_demand ? InpDemandLabelColor : InpSupplyLabelColor);
      string name      = ZV_PREFIX_ZONE + IntegerToString((int)z.formed_at) + "_" + (is_demand ? "D" : "S");

      // Hauteur visuelle minimum : une zone dont le corps de bougie pivot
      // est très fin par rapport à l'échelle du graphique s'afficherait
      // sinon comme un simple trait — le padding est purement visuel,
      // le tooltip et le score gardent les vraies bornes de la zone.
      double disp_top = z.zone_top, disp_bottom = z.zone_bottom;
      double mid_price0 = (disp_top + disp_bottom) / 2.0;
      double min_height = mid_price0 * (InpMinZoneHeightPct / 100.0);
      if(InpMinZoneHeightPct > 0 && (disp_top - disp_bottom) < min_height)
        {
         double pad = (min_height - (disp_top - disp_bottom)) / 2.0;
         disp_top    += pad;
         disp_bottom -= pad;
        }

      // Rectangle : grand aplat de couleur claire, bord net, toujours
      // derrière les bougies (BACK=true) pour ne jamais masquer le prix.
      ObjectCreate(0, name, OBJ_RECTANGLE, 0, z.formed_at, disp_top, now_time, disp_bottom);
      ObjectSetInteger(0, name, OBJPROP_COLOR, col);
      ObjectSetInteger(0, name, OBJPROP_FILL, !z.is_mitigated);
      ObjectSetInteger(0, name, OBJPROP_STYLE, z.is_mitigated ? STYLE_DOT : STYLE_SOLID);
      ObjectSetInteger(0, name, OBJPROP_WIDTH, z.is_mitigated ? 1 : 2);
      ObjectSetInteger(0, name, OBJPROP_BACK, true);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      string tip = StringFormat(
         "%s — score %.1f/10 (BOS %.1f Impulsion %.1f Temps %.1f Fraîcheur %.1f Sweep %.1f)\ntouchée %d fois%s",
         is_demand ? "ZONE DE DEMANDE" : "ZONE D'OFFRE", total,
         z.s_bos, z.s_impulse, z.s_time, z.s_fresh, z.s_sweep,
         z.touch_count, z.is_mitigated ? " — mitigée (déjà traversée)" : " — encore valide");
      ObjectSetString(0, name, OBJPROP_TOOLTIP, tip);

      // Étiquette : mot complet, gros, positionnée au bord DROIT de la
      // zone (côté prix actuel) pour être visible sans avoir à remonter
      // dans l'historique — c'est ce qu'on regarde en premier sur le graphique.
      string lbl = name + "_lbl";
      double mid_price = (z.zone_top + z.zone_bottom) / 2.0;
      ObjectCreate(0, lbl, OBJ_TEXT, 0, now_time, mid_price);
      ObjectSetString(0, lbl, OBJPROP_TEXT,
                       StringFormat(" %s (%.0f/10)", is_demand ? "DEMANDE" : "OFFRE", total));
      ObjectSetInteger(0, lbl, OBJPROP_COLOR, lbl_col);
      ObjectSetInteger(0, lbl, OBJPROP_FONTSIZE, 10);
      ObjectSetString(0, lbl, OBJPROP_FONT, "Arial Bold");
      ObjectSetInteger(0, lbl, OBJPROP_ANCHOR, ANCHOR_RIGHT);
      ObjectSetInteger(0, lbl, OBJPROP_SELECTABLE, false);

      shown++;
     }
  }

//+------------------------------------------------------------------+
//| Wyckoff — scan historique + dessin d'un pattern par MSS détecté   |
//+------------------------------------------------------------------+
void DrawWyckoff(const datetime &m1_time[], const double &m1_open[],
                 const double &m1_high[], const double &m1_low[],
                 const double &m1_close[], const int n1)
  {
   ObjectsDeleteAll(0, ZV_PREFIX_WY);

   int start = MathMax(1, n1 - InpWyckoffScanBars);
   int shown = 0;

   // ZeusWyckoffDetect exige que le MSS soit exactement la dernière
   // barre de la fenêtre (end_idx-1) : un pattern historique donné
   // n'est donc détecté qu'à UN SEUL end_idx — pas de doublons.
   for(int end_idx = n1; end_idx >= start && shown < InpMaxWyckoffShown; end_idx--)
     {
      for(int side_i = 0; side_i < 2 && shown < InpMaxWyckoffShown; side_i++)
        {
         ENUM_PIVOT_SIDE side = (side_i == 0) ? PIVOT_DEMAND : PIVOT_SUPPLY;
         ZeusWyckoffPattern pat;
         if(!ZeusWyckoffDetect(m1_time, m1_open, m1_high, m1_low, m1_close, side, end_idx, g_wy_params, pat))
            continue;
         if(pat.score < InpMinWyckoffScoreToShow)
            continue;
         DrawOneWyckoffPattern(pat, m1_time);
         shown++;
        }
     }
  }

void DrawOneWyckoffPattern(const ZeusWyckoffPattern &pat, const datetime &m1_time[])
  {
   bool   is_demand = (pat.side == PIVOT_DEMAND);
   color  col       = is_demand ? InpDemandWyckoffColor : InpSupplyWyckoffColor;
   string base      = ZV_PREFIX_WY + IntegerToString((int)pat.formed_at) + "_" + (is_demand ? "D" : "S");

   // Boîte d'accumulation — approximation visuelle (manip_bar - accum_bars
   // à manip_bar) : le scan réel autorise un écart entre la fin de la
   // fenêtre d'accumulation et la barre de manipulation elle-même, non
   // reconstitué ici puisque non exposé par ZeusWyckoffPattern. Les
   // points manip/MSS ci-dessous restent, eux, exacts (issus du struct).
   int      accum_start_bar  = (int)MathMax(0, pat.manip_bar - pat.accum_bars);
   datetime accum_start_time = m1_time[accum_start_bar];
   string   box = base + "_accum";
   ObjectCreate(0, box, OBJ_RECTANGLE, 0, accum_start_time, pat.accum_high,
                m1_time[pat.manip_bar], pat.accum_low);
   ObjectSetInteger(0, box, OBJPROP_COLOR, InpAccumBoxColor);
   ObjectSetInteger(0, box, OBJPROP_FILL, false);
   ObjectSetInteger(0, box, OBJPROP_STYLE, STYLE_DASH);
   ObjectSetInteger(0, box, OBJPROP_BACK, true);
   ObjectSetInteger(0, box, OBJPROP_SELECTABLE, false);
   ObjectSetString(0, box, OBJPROP_TOOLTIP, StringFormat("Accumulation (%d barres, approx.)", pat.accum_bars));

   // Flèche de manipulation (Spring pour demande / Upthrust pour offre)
   string arr1 = base + "_manip";
   ObjectCreate(0, arr1, OBJ_ARROW, 0, m1_time[pat.manip_bar], pat.manip_extreme);
   ObjectSetInteger(0, arr1, OBJPROP_ARROWCODE, is_demand ? 233 : 234);   // Wingdings flèche haut/bas
   ObjectSetInteger(0, arr1, OBJPROP_COLOR, col);
   ObjectSetInteger(0, arr1, OBJPROP_WIDTH, 1);
   ObjectSetInteger(0, arr1, OBJPROP_SELECTABLE, false);
   ObjectSetString(0, arr1, OBJPROP_TOOLTIP, is_demand ? "Spring (manipulation)" : "Upthrust (manipulation)");

   // Flèche de confirmation MSS (Market Structure Shift)
   string arr2 = base + "_mss";
   ObjectCreate(0, arr2, OBJ_ARROW, 0, m1_time[pat.mss_bar], pat.mss_close);
   ObjectSetInteger(0, arr2, OBJPROP_ARROWCODE, 108);   // Wingdings drapeau
   ObjectSetInteger(0, arr2, OBJPROP_COLOR, col);
   ObjectSetInteger(0, arr2, OBJPROP_WIDTH, 2);
   ObjectSetInteger(0, arr2, OBJPROP_SELECTABLE, false);

   double signal_threshold = is_demand ? g_ws_long : g_ws_short;
   string grade            = (pat.score >= signal_threshold) ? " [SIGNAL]" : "";
   ObjectSetString(0, arr2, OBJPROP_TOOLTIP,
                   StringFormat("MSS %s score=%.1f (seuil prod=%.1f)%s",
                                is_demand ? "haussier" : "baissier", pat.score, signal_threshold, grade));

   string lbl = base + "_lbl";
   ObjectCreate(0, lbl, OBJ_TEXT, 0, m1_time[pat.mss_bar], pat.mss_close);
   ObjectSetString(0, lbl, OBJPROP_TEXT, StringFormat("MSS %.1f%s", pat.score, grade));
   ObjectSetInteger(0, lbl, OBJPROP_COLOR, col);
   ObjectSetInteger(0, lbl, OBJPROP_FONTSIZE, 8);
   ObjectSetInteger(0, lbl, OBJPROP_ANCHOR, is_demand ? ANCHOR_LEFT_LOWER : ANCHOR_LEFT_UPPER);
   ObjectSetInteger(0, lbl, OBJPROP_SELECTABLE, false);
  }
//+------------------------------------------------------------------+
