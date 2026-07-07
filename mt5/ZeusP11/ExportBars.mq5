//+------------------------------------------------------------------+
//|                                                  ExportBars.mq5  |
//|  Script utilitaire — exporte l'historique M1 du symbole du       |
//|  graphique en CSV (UTC), pour le harnais de comparaison Python.  |
//|                                                                  |
//|  Usage : glisser le script sur un graphique → le fichier         |
//|  ZeusBars_<SYMBOLE>.csv apparaît dans MQL5/Files.                |
//|  Format : datetime_utc;open;high;low;close;volume                |
//|  (compatible parse_m1_csv Format B après remplacement du ';')    |
//+------------------------------------------------------------------+
#property script_show_inputs
#property strict

input int InpBars            = 400000;   // barres M1 max à exporter
input int InpServerUTCOffsetH = -99;     // décalage serveur→UTC (h), -99 = auto

void OnStart()
  {
   int offset_sec = (InpServerUTCOffsetH == -99)
                    ? (int)(TimeCurrent() - TimeGMT())
                    : InpServerUTCOffsetH * 3600;

   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   int copied = CopyRates(_Symbol, PERIOD_M1, 0, InpBars, rates);
   if(copied <= 0)
     {
      Print("ExportBars: aucune barre copiée — augmenter l'historique (F2)");
      return;
     }

   string fname = StringFormat("ZeusBars_%s.csv", _Symbol);
   int h = FileOpen(fname, FILE_WRITE | FILE_CSV | FILE_ANSI, ';');
   if(h == INVALID_HANDLE)
     {
      Print("ExportBars: impossible d'ouvrir ", fname);
      return;
     }

   for(int k = 0; k < copied; k++)
     {
      datetime utc = rates[k].time - offset_sec;
      FileWrite(h,
                TimeToString(utc, TIME_DATE | TIME_MINUTES),
                DoubleToString(rates[k].open, 8),
                DoubleToString(rates[k].high, 8),
                DoubleToString(rates[k].low, 8),
                DoubleToString(rates[k].close, 8),
                IntegerToString((int)rates[k].tick_volume));
     }
   FileClose(h);
   PrintFormat("ExportBars: %d barres M1 de %s exportées vers %s (UTC, offset %+dh)",
               copied, _Symbol, fname, offset_sec / 3600);
  }
//+------------------------------------------------------------------+
