//+------------------------------------------------------------------+
//| ZeusGuard.mqh — port fidèle de zeus/risk/propfirm.py             |
//| (PropFirmConfig + PropFirmGuard).                                |
//|                                                                  |
//| Fail closed : état incohérent → can_trade = false.               |
//| Toutes les heures sont UTC (l'appelant convertit depuis l'heure  |
//| serveur broker).                                                 |
//+------------------------------------------------------------------+
#property strict

struct ZeusGuardConfig
  {
   double            max_daily_loss_pct;    // 0.03
   double            max_total_dd_pct;      // 0.06
   double            risk_per_trade_pct;    // 0.006
   int               max_losses_per_day;    // 3
   int               flat_hour_utc;         // 21
   double            profit_target_pct;     // 0.10
   int               min_trading_days;      // 4
  };

void ZeusGuardDefaults(ZeusGuardConfig &c)
  {
   c.max_daily_loss_pct = 0.03;
   c.max_total_dd_pct   = 0.06;
   c.risk_per_trade_pct = 0.006;
   c.max_losses_per_day = 3;
   c.flat_hour_utc      = 21;
   c.profit_target_pct  = 0.10;
   c.min_trading_days   = 4;
  }

bool ZeusGuardConfigValid(const ZeusGuardConfig &c)
  {
   if(!(c.max_daily_loss_pct > 0 && c.max_daily_loss_pct < 1)) return false;
   if(!(c.max_total_dd_pct > 0 && c.max_total_dd_pct < 1))     return false;
   if(!(c.risk_per_trade_pct > 0
        && c.risk_per_trade_pct <= c.max_daily_loss_pct))      return false;
   if(c.max_losses_per_day < 1)                                return false;
   if(c.flat_hour_utc < 0 || c.flat_hour_utc > 23)             return false;
   return true;
  }

struct ZeusGuardState
  {
   double            initial_balance;
   double            equity;
   double            day_start_equity;
   int               current_day;         // AAAAMMJJ UTC, 0 = non initialisé
   int               losses_today;
   bool              halted_for_day;
   bool              halted_permanently;
   int               trading_days_count;  // approximation du set Python
   int               last_trading_day;    // dernier jour compté
  };

void ZeusGuardInit(ZeusGuardState &st, const double initial_balance)
  {
   st.initial_balance    = initial_balance;
   st.equity             = initial_balance;
   st.day_start_equity   = initial_balance;
   st.current_day        = 0;
   st.losses_today       = 0;
   st.halted_for_day     = false;
   st.halted_permanently = false;
   st.trading_days_count = 0;
   st.last_trading_day   = 0;
  }

int ZeusDayKeyUTC(const datetime ts_utc)
  {
   MqlDateTime dt;
   TimeToStruct(ts_utc, dt);
   return dt.year * 10000 + dt.mon * 100 + dt.day;
  }

// _roll_day
void ZeusGuardRollDay(ZeusGuardState &st, const datetime ts_utc)
  {
   int day = ZeusDayKeyUTC(ts_utc);
   if(st.current_day == 0 || day > st.current_day)
     {
      st.current_day      = day;
      st.day_start_equity = st.equity;
      st.losses_today     = 0;
      st.halted_for_day   = false;
     }
  }

// can_trade — ts_utc = timestamp UTC de la barre/du tick
bool ZeusGuardCanTrade(ZeusGuardState &st, const ZeusGuardConfig &c,
                       const datetime ts_utc)
  {
   if(st.initial_balance <= 0)
      return false;                       // fail closed
   if(st.halted_permanently)
      return false;

   ZeusGuardRollDay(st, ts_utc);

   if(st.halted_for_day)
      return false;

   MqlDateTime dt;
   TimeToStruct(ts_utc, dt);
   if(dt.hour >= c.flat_hour_utc)
      return false;
   return true;
  }

// on_trade_closed
void ZeusGuardOnTradeClosed(ZeusGuardState &st, const ZeusGuardConfig &c,
                            const datetime ts_utc, const double pnl_usd)
  {
   ZeusGuardRollDay(st, ts_utc);

   st.equity += pnl_usd;
   int day = ZeusDayKeyUTC(ts_utc);
   if(day != st.last_trading_day)
     {
      st.last_trading_day = day;
      st.trading_days_count++;
     }
   if(pnl_usd < 0)
      st.losses_today++;

   // 1. Drawdown total interne (depuis le solde initial — convention FTMO)
   double total_dd = (st.initial_balance - st.equity) / st.initial_balance;
   if(total_dd >= c.max_total_dd_pct)
     {
      st.halted_permanently = true;
      return;
     }

   // 2. Perte journalière interne
   double daily_loss = (st.day_start_equity - st.equity) / st.initial_balance;
   if(daily_loss >= c.max_daily_loss_pct)
     {
      st.halted_for_day = true;
      return;
     }

   // 3. Série de pertes journalière
   if(st.losses_today >= c.max_losses_per_day)
      st.halted_for_day = true;
  }

double ZeusGuardRiskUSD(const ZeusGuardState &st, const ZeusGuardConfig &c)
  {
   return st.initial_balance * c.risk_per_trade_pct;   // fixe, pas de compounding
  }

bool ZeusGuardChallengePassed(const ZeusGuardState &st, const ZeusGuardConfig &c)
  {
   double profit_pct = (st.equity - st.initial_balance) / st.initial_balance;
   return (!st.halted_permanently
           && profit_pct >= c.profit_target_pct
           && st.trading_days_count >= c.min_trading_days);
  }
//+------------------------------------------------------------------+
