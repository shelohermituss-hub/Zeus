# ZeusOpenMomentum — Expert Advisor MQL5 (continuation de momentum, Open US)

⚠️ **STRATÉGIE NOUVELLE, NON VALIDÉE.** Contrairement à `ZeusP11` (port d'une
stratégie déjà auditée sur backtest Python, avec `zeus/backtest/sd_simulation.py`
comme référence de performance), cette logique **n'a aucune référence de
backtest établie** dans ce dépôt. Elle a été conçue par composition de
briques déjà validées (guard propfirm, échelle de sorties) autour d'une
logique de détection entièrement nouvelle (range d'ouverture US). C'est un
point de départ à valider, pas une stratégie prouvée.

**Ne jamais activer `InpTradingEnabled=true` avant d'avoir fait tourner l'EA
en `InpSignalLogMode=true` (ou en compte démo) pendant plusieurs semaines et
d'avoir inspecté les signaux générés.** Les valeurs par défaut du fichier
(`InpTradingEnabled=false`, `InpSignalLogMode=true`) reflètent ce principe non
négociable du projet — ne pas les changer sans validation préalable.

## Stratégie

**Continuation de momentum sur l'ouverture US (9h30 ET).**

1. **Range d'ouverture** — mesure haut/bas/open sur les `InpORMinutes`
   premières minutes après 9h30 ET (défaut 15 min).
2. **Momentum figé une seule fois**, à la clôture de cette fenêtre : sens du
   close vs l'open de fenêtre. Si le mouvement est inférieur à
   `InpMinMomentumPips`, **aucun trade n'est pris ce jour-là** (pas de
   fallback, pas de retry).
3. **Entrée** dans le sens du momentum déjà établi :
   - `InpRequireBreakout=false` (défaut) : entrée immédiate à la clôture de la
     fenêtre — "continuation", suit le sens déjà confirmé sans attendre.
   - `InpRequireBreakout=true` : attend que le prix casse le bord du range
     dans ce même sens avant d'entrer — confirmation plus stricte, entrée
     plus tardive, moins de faux départs mais aussi moins d'opportunités.
4. **SL** = extrémité opposée du range, borné à `[InpMinSLPips, InpMaxSLPips]`
   — un range trop étroit (bruit) ou trop large (volatilité anormale) fait
   rejeter le signal plutôt que de forcer un trade mal dimensionné.
5. **Sorties** gérées par la même échelle progressive que ZeusP11
   (`ZeusExits.mqh`, réutilisé sans modification) : TP1 → SL à BE, TP2 →
   clôture partielle + verrouillage du gain TP1, Runner final. Comptabilité
   uniquement sur confirmation broker réelle (aucune fermeture supposée).
6. **Un seul trade par jour et par symbole.** Le guard propfirm
   (`ZeusGuard.mqh`, identique à ZeusP11 — même garanties fail-closed : perte
   journalière max, drawdown total max, série de pertes max, heure plate)
   peut bloquer une entrée indépendamment de la logique de range.
7. **Clôture forcée à `InpFlatHourUTC`** même sans SL/TP touché, en plus du
   blocage des nouvelles entrées après cette heure — le trade ne traverse
   jamais la nuit.

### Gestion du DST — point important

L'heure d'ouverture US (9h30 ET) est calculée avec la règle de changement
d'heure **des États-Unis** (2ᵉ dimanche de mars 07:00 UTC → 1ᵉʳ dimanche de
novembre 06:00 UTC, en vigueur depuis 2007), recalculée chaque jour dans
`Include/ZeusOpenRange.mqh` (`ZeusUSOpenTimeUTC`). C'est **distinct** de la
conversion serveur→UTC du broker (`CurrentUTCOffsetSec`/`ToUTC`, dans
`ZeusOpenMomentum.mq5`) qui gère le DST du broker lui-même — les deux ne
suivent pas forcément le même calendrier de changement d'heure, ce qui est
géré correctement en gardant les deux logiques strictement séparées.

### Limite connue : démarrage de l'EA en cours de fenêtre d'ouverture

Si l'EA (re)démarre alors que la fenêtre d'ouverture du jour est déjà
entamée, le range capturé pour **ce jour-là uniquement** part de la première
barre M1 vue après le démarrage, pas du véritable open 9h30 ET — le range
peut donc être artificiellement plus étroit ce jour précis. Les jours
suivants ne sont pas affectés (le range se reconstruit proprement depuis
`or_start`). Si l'EA démarre carrément après la fin de la fenêtre du jour, le
jour est proprement ignoré (`OR_PHASE_DONE`, aucun signal basé sur des
données incomplètes).

## Réutilisation de l'infrastructure existante

Aucune logique de risque/sortie n'a été réécrite : `Include/ZeusGuard.mqh` et
`Include/ZeusExits.mqh` sont des copies **verbatim** de celles de `ZeusP11/`
(même convention de ce dépôt : chaque EA a sa propre copie locale des
fichiers partagés — voir le README de ZeusP11). Le dimensionnement par risque
(`LotsForRisk`) reprend à l'identique le garde-fou anti-`tick_value_loss`
corrompu de ZeusP11 (vérification a priori que la perte au lot minimum ne
dépasse pas ~1.5× le risque visé, avec recalcul direct via `contract_size`
quand la devise de profit du symbole est celle du compte).

## Installation

1. Ouvre MetaEditor (F4 depuis MT5)
2. Copie ce dossier dans `MQL5\Experts\ZeusOpenMomentum\` :
   ```
   MQL5\Experts\ZeusOpenMomentum\ZeusOpenMomentum.mq5
   MQL5\Experts\ZeusOpenMomentum\Include\ZeusOpenRange.mqh
   MQL5\Experts\ZeusOpenMomentum\Include\ZeusGuard.mqh
   MQL5\Experts\ZeusOpenMomentum\Include\ZeusExits.mqh
   ```
3. Ouvre `ZeusOpenMomentum.mq5` dans MetaEditor → **F7** (compiler)
4. Glisse `ZeusOpenMomentum` sur **un seul graphique** (n'importe lequel —
   l'EA gère lui-même la liste `InpSymbols` via son propre timer, comme
   ZeusP11), avec **"Autoriser le trading algorithmique"** coché.
5. Vérifie dans l'onglet **Experts** les lignes de confirmation :
   `InpTradingEnabled=false — mode OBSERVATION` et/ou
   `InpSignalLogMode=true — signaux journalisés en CSV`.

Le CSV de signaux (`ZeusOpenMomentum_signals_<date>.csv`, dans `MQL5\Files\`)
journalise chaque signal qui *aurait* déclenché un trade (heure UTC, symbole,
direction, range, SL en pips, prix) même en mode observation — c'est le
support attendu pour valider la stratégie avant d'envisager
`InpTradingEnabled=true`.

## Paramètres principaux

| Paramètre | Défaut | Rôle |
|---|---|---|
| `InpAccountBalance` | 100000 | Solde de référence pour le risque fixe (non-compounding) |
| `InpRiskPerTradePct` | 0.005 | Risque par trade |
| `InpMaxDailyLossPct` / `InpMaxTotalDDPct` | 0.03 / 0.06 | Arrêt journalier / définitif (guard) |
| `InpFlatHourUTC` | 20 | Aucune nouvelle entrée après, clôture forcée |
| `InpSymbols` | `XAUUSD` | Liste de symboles (noms broker exacts, séparés par virgule) |
| `InpORMinutes` | 15 | Durée de la fenêtre d'ouverture US |
| `InpMinMomentumPips` | 15.0 | Mouvement minimum pour valider un momentum |
| `InpRequireBreakout` | false | true = attend la cassure du range avant d'entrer |
| `InpMinSLPips` / `InpMaxSLPips` | 5.0 / 30.0 | Bornes d'acceptation du SL (range trop plat/trop large = rejet) |
| `InpTP1R` / `InpTP2R` / `InpRunnerRR` | 1.0 / 2.0 / 4.0 | Échelle de sorties (R) |
| `InpSignalLogMode` | **true** | Journalise en CSV, n'envoie aucun ordre |
| `InpTradingEnabled` | **false** | Doit rester `false` tant que la stratégie n'est pas validée en démo |

**Non testé en compilation dans cet environnement** (pas de MetaEditor sur
cette machine) — chaque appel de fonction a été vérifié manuellement contre
les signatures de `ZeusOpenRange.mqh`/`ZeusGuard.mqh`/`ZeusExits.mqh`, mais
une compilation réelle (étape 3 ci-dessus) reste nécessaire avant usage. En
cas d'erreur de compilation, renvoie le message exact du compilateur pour un
correctif ciblé.
