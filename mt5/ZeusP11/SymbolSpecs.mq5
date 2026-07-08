//+------------------------------------------------------------------+
//|                                                 SymbolSpecs.mq5   |
//|  Script utilitaire — exporte les spécifications broker de chaque |
//|  symbole de InpSymbols, pour vérifier leur cohérence avec les    |
//|  hypothèses du backtest Python (contract size, spread, swap).    |
//|                                                                  |
//|  Usage : glisser le script sur UN graphique (n'importe lequel).  |
//|  Sortie : SymbolSpecs.csv dans MQL5/Files + journal Experts.      |
//+------------------------------------------------------------------+
#property script_show_inputs
#property strict

input string InpSymbols = "XAUUSD,GBPUSD,EURUSD,GBPAUD,EURNZD,USDCHF,CADJPY,NZDUSD,EURJPY,XAGUSD"; // liste à vérifier (+ DAX/UKXGBP si dispo)

string SwapModeToStr(const int mode)
  {
   switch(mode)
     {
      case SYMBOL_SWAP_MODE_DISABLED:        return "disabled";
      case SYMBOL_SWAP_MODE_POINTS:          return "points";
      case SYMBOL_SWAP_MODE_CURRENCY_SYMBOL: return "currency_symbol";
      case SYMBOL_SWAP_MODE_CURRENCY_MARGIN: return "currency_margin";
      case SYMBOL_SWAP_MODE_CURRENCY_DEPOSIT:return "currency_deposit";
      case SYMBOL_SWAP_MODE_INTEREST_CURRENT:return "interest_current";
      case SYMBOL_SWAP_MODE_INTEREST_OPEN:   return "interest_open";
      case SYMBOL_SWAP_MODE_REOPEN_CURRENT:  return "reopen_current";
      case SYMBOL_SWAP_MODE_REOPEN_BID:      return "reopen_bid";
      default:                                return "unknown";
     }
  }

string TradeModeToStr(const int mode)
  {
   switch(mode)
     {
      case SYMBOL_TRADE_MODE_DISABLED:  return "DISABLED";
      case SYMBOL_TRADE_MODE_LONGONLY:  return "long_only";
      case SYMBOL_TRADE_MODE_SHORTONLY: return "short_only";
      case SYMBOL_TRADE_MODE_CLOSEONLY: return "close_only";
      case SYMBOL_TRADE_MODE_FULL:      return "full";
      default:                          return "unknown";
     }
  }

void OnStart()
  {
   string parts[];
   int n = StringSplit(InpSymbols, ',', parts);

   int h = FileOpen("SymbolSpecs.csv", FILE_WRITE | FILE_CSV | FILE_ANSI, ';');
   if(h == INVALID_HANDLE)
     {
      Print("SymbolSpecs: impossible d'ouvrir SymbolSpecs.csv");
      return;
     }
   FileWrite(h, "symbol", "exists", "digits", "point", "tick_size", "tick_value",
             "tick_value_profit", "tick_value_loss", "contract_size",
             "volume_min", "volume_max", "volume_step",
             "currency_base", "currency_profit", "currency_margin",
             "spread_points", "swap_long", "swap_short", "swap_mode",
             "trade_mode", "trade_calc_mode");

   string acct_ccy = AccountInfoString(ACCOUNT_CURRENCY);
   PrintFormat("SymbolSpecs: devise du compte = %s", acct_ccy);

   for(int i = 0; i < n; i++)
     {
      string sym = parts[i];
      StringTrimLeft(sym);
      StringTrimRight(sym);
      if(sym == "") continue;

      bool exists = SymbolSelect(sym, true);
      if(!exists)
        {
         PrintFormat("SymbolSpecs: %s INTROUVABLE chez ce broker", sym);
         FileWrite(h, sym, "NO", "", "", "", "", "", "", "", "", "", "",
                   "", "", "", "", "", "", "", "", "");
         continue;
        }

      int    digits      = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
      double point        = SymbolInfoDouble(sym, SYMBOL_POINT);
      double tick_size    = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
      double tick_value   = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
      double tick_val_p   = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE_PROFIT);
      double tick_val_l   = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE_LOSS);
      double contract     = SymbolInfoDouble(sym, SYMBOL_TRADE_CONTRACT_SIZE);
      double vmin         = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
      double vmax         = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);
      double vstep        = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
      string ccy_base     = SymbolInfoString(sym, SYMBOL_CURRENCY_BASE);
      string ccy_profit   = SymbolInfoString(sym, SYMBOL_CURRENCY_PROFIT);
      string ccy_margin   = SymbolInfoString(sym, SYMBOL_CURRENCY_MARGIN);
      long   spread_pts   = SymbolInfoInteger(sym, SYMBOL_SPREAD);
      double swap_long    = SymbolInfoDouble(sym, SYMBOL_SWAP_LONG);
      double swap_short   = SymbolInfoDouble(sym, SYMBOL_SWAP_SHORT);
      int    swap_mode    = (int)SymbolInfoInteger(sym, SYMBOL_SWAP_MODE);
      int    trade_mode   = (int)SymbolInfoInteger(sym, SYMBOL_TRADE_MODE);
      int    calc_mode    = (int)SymbolInfoInteger(sym, SYMBOL_TRADE_CALC_MODE);

      string flag = (ccy_profit == acct_ccy) ? "  [pas de conversion]"
                                              : "  [CONVERSION requise]";
      PrintFormat("SymbolSpecs: %-8s contract=%.2f tick_size=%.5f tick_value_loss=%.5f "
                  "profit_ccy=%s%s spread=%d pts swap L/S=%.2f/%.2f",
                  sym, contract, tick_size, tick_val_l, ccy_profit, flag,
                  spread_pts, swap_long, swap_short);

      FileWrite(h, sym, "YES",
                IntegerToString(digits),
                DoubleToString(point, 8),
                DoubleToString(tick_size, 8),
                DoubleToString(tick_value, 5),
                DoubleToString(tick_val_p, 5),
                DoubleToString(tick_val_l, 5),
                DoubleToString(contract, 4),
                DoubleToString(vmin, 4),
                DoubleToString(vmax, 4),
                DoubleToString(vstep, 4),
                ccy_base, ccy_profit, ccy_margin,
                IntegerToString((int)spread_pts),
                DoubleToString(swap_long, 4),
                DoubleToString(swap_short, 4),
                SwapModeToStr(swap_mode),
                TradeModeToStr(trade_mode),
                IntegerToString(calc_mode));
     }

   FileClose(h);
   Print("SymbolSpecs: terminé — voir SymbolSpecs.csv (MQL5/Files) et le journal ci-dessus");
  }
//+------------------------------------------------------------------+
