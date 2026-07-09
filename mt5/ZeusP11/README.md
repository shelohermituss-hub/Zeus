# ZeusP11 — Expert Advisor MQL5 (portefeuille propfirm P11-V4)

Port MQL5 complet de la stratégie Zeus S&D/Wyckoff : XAUUSD en V4 production
(Runner@20R) + 10 paires WS7.5 (TIERED-5R), sous guard propfirm strict.
**Un seul graphique suffit** — l'EA gère les 11 symboles par timer.

## ZeusVision — indicateur visuel (zones + Wyckoff, sans trading)

`ZeusVision.mq5` est un **indicateur** (pas un EA — n'envoie jamais d'ordre)
qui trace sur le graphique exactement ce que la stratégie V4 voit, en
réutilisant sans aucune modification la détection déjà validée
(`ZeusZones.mqh` + `ZeusWyckoff.mqh` via `ZeusSignals.mqh`) :

- **Zones Supply/Demand** (rectangles), calculées sur M15 resamplé depuis M1
  — jamais `CopyRates` M15 direct, parité stricte avec l'EA. Zones actives en
  couleur pleine, zones mitigées en pointillés grisés (configurable).
- **Patterns Wyckoff** (Accumulation → Manipulation → MSS) détectés sur M1 :
  flèche sur la barre de Spring/Upthrust, flèche + étiquette de score sur la
  barre de confirmation MSS, boîte d'accumulation en arrière-plan. Les
  patterns dont le score dépasse le seuil de signal de production
  (`InpXauWSLong/Short`, `InpFxWSLong/Short`) sont marqués `[SIGNAL]`.

**Multi-timeframe** : la détection tourne toujours sur le couple M1/M15 que
le bot trade réellement, mais l'indicateur peut être posé sur n'importe quel
graphique (M1, M5, M15, H1, H4...) — les objets sont positionnés par
prix/temps absolus, donc visibles quelle que soit la résolution d'affichage.

### Installation (indicateur, dossier séparé de l'EA)

1. Copie dans `MQL5\Indicators\ZeusVision\` :
   ```
   MQL5\Indicators\ZeusVision\ZeusVision.mq5
   MQL5\Indicators\ZeusVision\Include\ZeusPivot.mqh
   MQL5\Indicators\ZeusVision\Include\ZeusZones.mqh
   MQL5\Indicators\ZeusVision\Include\ZeusWyckoff.mqh
   MQL5\Indicators\ZeusVision\Include\ZeusSignals.mqh
   ```
2. Ouvre `ZeusVision.mq5` dans MetaEditor → **F7** (compiler)
3. Glisse `ZeusVision` sur n'importe quel graphique du symbole voulu

Aucun paramètre de risque/trading — purement visuel. Les tooltips des objets
(survol souris) détaillent le scoring (BOS/impulsion/temps/fraîcheur/sweep
pour une zone ; les 5 critères Wyckoff pour un pattern).

**Non testé en compilation dans cet environnement** (pas de MetaEditor sur
cette machine) — chaque appel de fonction a été vérifié manuellement contre
les signatures de `ZeusZones.mqh`/`ZeusWyckoff.mqh`/`ZeusSignals.mqh`, mais
une compilation réelle (étape 2 ci-dessus) reste nécessaire avant usage.

## Installation

1. Ouvre MetaEditor (F4 depuis MT5)
2. Copie ce dossier dans `MQL5\Experts\ZeusP11\` :
   ```
   MQL5\Experts\ZeusP11\ZeusP11.mq5
   MQL5\Experts\ZeusP11\ExportBars.mq5
   MQL5\Experts\ZeusP11\SymbolSpecs.mq5
   MQL5\Experts\ZeusP11\Include\ZeusPivot.mqh
   MQL5\Experts\ZeusP11\Include\ZeusZones.mqh
   MQL5\Experts\ZeusP11\Include\ZeusWyckoff.mqh
   MQL5\Experts\ZeusP11\Include\ZeusSignals.mqh
   MQL5\Experts\ZeusP11\Include\ZeusGuard.mqh
   MQL5\Experts\ZeusP11\Include\ZeusExits.mqh
   ```
3. Ouvre `ZeusP11.mq5` → **F7** (compiler) → le `.ex5` est créé
4. Dans MT5 : glisse `ZeusP11` sur un graphique (ex. XAUUSD M1),
   coche « Autoriser le trading algorithmique »
5. Ajuste `InpSymbols` avec les noms EXACTS de ton broker
   (ex. `DAX`, `XAUUSD.a` — le nom exact varie selon le broker)

## ⚠️ Protocole de validation OBLIGATOIRE avant tout ordre réel

Un port de 2 300 lignes de logique critique ne se croit pas sur parole —
il se prouve. Quatre étapes, dans l'ordre :

**0. Spécifications broker (une fois, avant tout backtest)**
   - Glisse `SymbolSpecs.mq5` sur un graphique quelconque
     → journal Experts + `MQL5\Files\SymbolSpecs.csv`
   - Vérifie pour CHAQUE symbole de `InpSymbols` : `contract_size`,
     `tick_size`, `tick_value_loss`, `currency_profit`, `spread_points`,
     `swap_long/short`. Un `tick_value_loss` incohérent avec
     `contract_size × tick_size` a déjà causé un sizing de lot 10x trop
     gros sur XAUUSD/XAGUSD chez un broker — `LotsForRisk()` corrige ce
     cas quand `currency_profit == devise du compte`, mais toute anomalie
     signalée dans le journal (`tick_value_loss incohérent...`) mérite
     vérification manuelle avant de continuer.
   - Compare le spread réel à celui supposé par le backtest Python
     (`zeus/backtest/run_multi_symbol_propfirm.py`, colonne `spread quote`
     de `SYMBOLS`) — un écart important dégradera la performance réelle
     même avec une logique de signal identique.

**1. Équivalence des signaux (harnais)**
   - Glisse `ExportBars.mq5` sur le graphique de chaque symbole
     → `MQL5\Files\ZeusBars_<SYM>.csv`
   - Lance l'EA dans le **testeur de stratégie** avec
     `InpSignalLogMode=true` → `ZeusP11_signals_*.csv` (aucun ordre envoyé)
   - Compare avec la référence Python :
     ```
     python -m zeus.backtest.compare_ea_signals \
         --bars ZeusBars_XAUUSD.csv --signals ZeusP11_signals.csv \
         --symbol XAUUSD
     ```
   - **Exigence : « ÉQUIVALENCE CONFIRMÉE » (100%) pour chaque symbole.**
     La moindre divergence = bug de port à corriger AVANT de continuer.

**2. Démo** — compte démo plusieurs semaines, `InpTradingEnabled=true`,
   comparer les trades avec le moteur Python paper en parallèle.

**3. Réel** — seulement après 1 et 2. Règle du projet : jamais de live
   sans paper validé.

## Paramètres principaux (défauts = valeurs validées par backtest)

| Input | Défaut | Rôle |
|---|---|---|
| `InpRiskPerTradePct` | 0.006 | risque fixe par trade (0.6%) |
| `InpMaxDailyLossPct` / `InpMaxTotalDDPct` | 0.03 / 0.06 | limites internes guard |
| `InpMaxLossesPerDay` | 3 | stop journalier après N pertes |
| `InpFlatHourUTC` | 21 | aucune entrée après cette heure UTC |
| `InpSymbols` | les 11 | liste des symboles (noms broker) |
| `InpXauWSLong/Short` | 5.9 / 8.5 | seuils Wyckoff or |
| `InpFxWSLong/Short` | 7.5 / 6.5 | seuils Wyckoff forex |
| `InpXauRunnerRR` / `InpFxRunnerRR` | 20 / 10 | runner final |
| `InpServerUTCOffsetH` | auto | décalage heure serveur→UTC |
| `InpSignalLogMode` | false | log CSV, aucun ordre |
| `InpTradingEnabled` | true | false = observation |

## Garde-fous intégrés

- Aucun ordre sans SL attaché à la requête
- Risque fixe sur le solde de référence (pas de compounding)
- Guard : arrêt journalier (perte 3% ou 3 pertes), arrêt définitif (DD 6%)
- Caps par symbole/direction : 1 perte/jour, 4 pertes/mois
- Un seul trade ouvert par symbole ; config invalide = refus de démarrer
- Réconciliation : position fermée par le SL broker correctement comptabilisée

## Correspondance avec le code Python de référence

| MQL5 | Python |
|---|---|
| `ZeusPivot.mqh` | `zeus/strategy/supply_demand/pivot_candle.py` |
| `ZeusZones.mqh` | `zeus/strategy/supply_demand/zone_detector.py` |
| `ZeusWyckoff.mqh` | `zeus/strategy/supply_demand/wyckoff.py` |
| `ZeusSignals.mqh` | `zeus/strategy/supply_demand/sd_strategy.py` (config prod) |
| `ZeusGuard.mqh` | `zeus/risk/propfirm.py` |
| `ZeusExits.mqh` | `zeus/live/p11_engine.py` (`_manage_exits`) |

Toute modification d'un côté DOIT être répliquée de l'autre et re-validée
par le harnais (étape 1).

## Différences assumées vs backtest (documentées, non éliminables)

- **Entrée** : marché au tick suivant le signal (backtest : open de la barre
  M1 suivante + spread fixe). Écart ≈ spread réel vs 0.30$ simulé.
- **Détection intra-barre** : le backtest voit TP/SL sur la barre M1 complète ;
  l'EA vérifie à chaque timer (15s) — résolution quasi identique.
- **Données** : le feed de ton broker diffère de HistData — les signaux réels
  peuvent légitimement différer du backtest historique (c'est le rôle de
  l'étape démo). Le harnais, lui, compare à données IDENTIQUES.
