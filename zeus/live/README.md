# Bot propfirm P11-V4 — guide d'utilisation

Bot multi-symboles (11 marchés) exécutant la stratégie S&D/Wyckoff validée :
XAUUSD en V4 production (Runner@20R) + 10 paires forex/indices/argent en
WS7.5 (TIERED-5R), sous garde-fou propfirm strict.

## Prérequis

- **Windows** avec un terminal **MetaTrader 5** installé et connecté au broker
- Python 3.11+ et `pip install MetaTrader5 pandas numpy pyyaml`
- Les 11 symboles visibles dans le Market Watch MT5 (noms exacts → voir
  `broker_symbol` dans le YAML, ex. `GER40` pour le DAX)
- Identifiants dans les variables d'environnement (jamais dans un fichier) :
  ```
  set ZEUS_MT5_LOGIN=12345678
  set ZEUS_MT5_PASSWORD=...
  set ZEUS_MT5_SERVER=FTMO-Server
  ```

## Démarrage

```bash
# 1. Paper trading (OBLIGATOIRE avant tout live — règle du projet)
python -m zeus.live.run_p11

# 2. Répétition générale : ordres construits mais jamais envoyés
python -m zeus.live.run_p11 --dry-run

# 3. Live — exige mode: live dans le YAML ET --mode live (double opt-in)
python -m zeus.live.run_p11 --mode live
```

## Personnalisation — config/p11_v4.yaml

Tout se règle dans le YAML, le code ne se touche pas :

| Réglage | Où | Exemple |
|---|---|---|
| Risque par trade | `risk.risk_per_trade_pct` | `0.006` (0.6% validé) |
| Limites propfirm | `risk.max_daily_loss_pct` / `max_total_dd_pct` | `0.03` / `0.06` |
| Activer/couper une paire | `symbols.<PAIRE>.enabled` | `false` |
| Nom broker différent | `symbols.<PAIRE>.broker_symbol` | `GER40`, `XAUUSD.a` |
| RR / échelle de sorties | `strategy_groups.<groupe>.exits` | `runner_rr: 15.0` |
| Seuils de signaux | `strategy_groups.<groupe>.signals` | `min_wyckoff_score_long` |
| Heure de coupure | `risk.flat_hour_utc` | `21` |

Surcharges rapides sans éditer le YAML :

```bash
python -m zeus.live.run_p11 --risk 0.005 --pairs XAUUSD,EURUSD,CADJPY
```

Toute valeur invalide (risque > limite journalière, TP dans le désordre,
paire sans groupe…) fait **refuser le démarrage** avec un message explicite.

## Garde-fous intégrés (non désactivables)

- Aucun ordre sans stop-loss attaché à la requête MT5
- Risque fixe sur le solde initial — pas de compounding
- Arrêt journalier : perte 3% ou 3 pertes dans la journée
- Arrêt définitif : drawdown 6% depuis le solde initial
- Aucune entrée après `flat_hour_utc` ; un seul trade ouvert par symbole
- Caps par symbole/direction : 1 perte/jour, 4 pertes/mois (identiques au backtest)
- Erreur de données ou rejet broker → aucun trade (fail closed)
- `--risk` CLI borné au maximum validé par backtest (0.75%)

## Références de performance (backtest 2024-2025, compte 100k$)

| Risque | PnL/an | DD max | Détail |
|---|---|---|---|
| 0.50% | +38% / +101% | 3.75% | prudent |
| 0.60% | +46% / +121% | ~4.5% | **recommandé** |
| 0.75% | +58% / +151% | 5.63% | marge fine sous la limite 6% |

OOS juin 2026 : −0.2%, DD 1.13%, zéro breach.
Scripts de reproduction : `zeus/backtest/run_multi_symbol_propfirm.py`.
