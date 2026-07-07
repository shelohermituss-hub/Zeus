//+------------------------------------------------------------------+
//| ZeusPivot.mqh — port fidèle de zeus/strategy/supply_demand/      |
//| pivot_candle.py (commit de référence : branche trading-bot).     |
//|                                                                  |
//| Toute modification ici doit être répliquée dans le Python et     |
//| re-validée par le harnais de comparaison (voir ZeusP11.mq5,      |
//| input InpSignalLogMode).                                         |
//+------------------------------------------------------------------+
#property strict

// ── Constantes (pivot_candle.py) ─────────────────────────────────────
#define ZP_MIN_RANGE   1e-8
#define ZP_MIN_SCORE   3.0
#define ZP_MIN_WICK_R  0.25
#define ZP_DOMINANCE_R 1.5

// ── Côté du pivot ────────────────────────────────────────────────────
enum ENUM_PIVOT_SIDE
  {
   PIVOT_DEMAND = 0,   // longue mèche basse → acheteurs
   PIVOT_SUPPLY = 1,   // longue mèche haute → vendeurs
   PIVOT_DOJI   = 2    // indécision
  };

// ── Résultat d'analyse d'une bougie ─────────────────────────────────
struct ZeusPivotCandle
  {
   datetime          time;
   double            open, high, low, close;
   ENUM_PIVOT_SIDE   side;
   double            score;        // 0–10
   double            body_high;    // max(open, close)
   double            body_low;     // min(open, close)
   double            wick_high;    // == high
   double            wick_low;     // == low
   double            upper_wick_ratio;
   double            lower_wick_ratio;
   double            body_ratio;
  };

// ── _score_demand (identique Python) ─────────────────────────────────
double ZeusScoreDemand(const double rng, const double body,
                       const double upper, const double lower,
                       const double body_mid)
  {
   if(rng < ZP_MIN_RANGE)
      return 0.0;
   double lower_r = lower / rng;
   double upper_r = upper / rng;
   double body_r  = body  / rng;

   double pts = 0.0;
   pts += MathMin(4.0, lower_r * 8.0);                    // mèche basse
   pts += MathMax(0.0, 2.0 - body_r * (2.0 / 0.30));      // petit corps
   pts += MathMin(2.0, body_mid * 2.0);                   // corps vers le haut
   pts += MathMax(0.0, 2.0 - upper_r * (2.0 / 0.10));     // mèche haute minuscule
   return MathMin(10.0, pts);
  }

// ── _score_supply (identique Python) ─────────────────────────────────
double ZeusScoreSupply(const double rng, const double body,
                       const double upper, const double lower,
                       const double body_mid)
  {
   if(rng < ZP_MIN_RANGE)
      return 0.0;
   double lower_r = lower / rng;
   double upper_r = upper / rng;
   double body_r  = body  / rng;

   double pts = 0.0;
   pts += MathMin(4.0, upper_r * 8.0);                    // mèche haute
   pts += MathMax(0.0, 2.0 - body_r * (2.0 / 0.30));      // petit corps
   pts += MathMin(2.0, (1.0 - body_mid) * 2.0);           // corps vers le bas
   pts += MathMax(0.0, 2.0 - lower_r * (2.0 / 0.10));     // mèche basse minuscule
   return MathMin(10.0, pts);
  }

// ── analyze_candle (identique Python, arrondis inclus) ───────────────
void ZeusAnalyzeCandle(const datetime tm,
                       const double o, const double h,
                       const double l, const double c,
                       ZeusPivotCandle &out)
  {
   double rng    = h - l;
   double body_h = MathMax(o, c);
   double body_l = MathMin(o, c);

   double body = 0.0, upper = 0.0, lower = 0.0, body_mid = 0.5;
   double upper_r = 0.0, lower_r = 0.0, body_r = 0.0;
   double score_d = 0.0, score_s = 0.0;

   if(rng >= ZP_MIN_RANGE)
     {
      body     = body_h - body_l;
      upper    = h - body_h;
      lower    = body_l - l;
      body_mid = ((body_h + body_l) / 2.0 - l) / rng;
      upper_r  = upper / rng;
      lower_r  = lower / rng;
      body_r   = body  / rng;
      score_d  = ZeusScoreDemand(rng, body, upper, lower, body_mid);
      score_s  = ZeusScoreSupply(rng, body, upper, lower, body_mid);
     }

   // _determine_side
   bool lower_dominates = (lower_r >= ZP_MIN_WICK_R
                           && lower_r >= upper_r * ZP_DOMINANCE_R
                           && score_d >= ZP_MIN_SCORE);
   bool upper_dominates = (upper_r >= ZP_MIN_WICK_R
                           && upper_r >= lower_r * ZP_DOMINANCE_R
                           && score_s >= ZP_MIN_SCORE);

   ENUM_PIVOT_SIDE side;
   double score;
   if(lower_dominates && (!upper_dominates || score_d >= score_s))
     { side = PIVOT_DEMAND; score = score_d; }
   else if(upper_dominates)
     { side = PIVOT_SUPPLY; score = score_s; }
   else
     { side = PIVOT_DOJI;   score = MathMax(score_d, score_s); }

   out.time             = tm;
   out.open             = o;
   out.high             = h;
   out.low              = l;
   out.close            = c;
   out.side             = side;
   out.score            = NormalizeDouble(score, 2);      // round(score, 2)
   out.body_high        = body_h;
   out.body_low         = body_l;
   out.wick_high        = h;
   out.wick_low         = l;
   out.upper_wick_ratio = NormalizeDouble(upper_r, 4);
   out.lower_wick_ratio = NormalizeDouble(lower_r, 4);
   out.body_ratio       = NormalizeDouble(body_r, 4);
  }
//+------------------------------------------------------------------+
