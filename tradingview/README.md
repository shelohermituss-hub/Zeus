# ZeusVision — Pine Script (TradingView)

Portage Pine Script v6 de l'indicateur `mt5/ZeusVision.mq5` : trace les zones
Supply/Demand détectées par la logique déjà validée de la stratégie de
production (`zeus/strategy/supply_demand/zone_detector.py` /
`pivot_candle.py`, portées en MQL5 dans `mt5/ZeusP11/Include/ZeusZones.mqh`
/ `ZeusPivot.mqh`).

## Ce qu'il fait

- Score de bougie pivot (demande/offre) sur 10, calculé sur le timeframe
  choisi (`InpZoneTF`, M15 par défaut — comme en production).
- Recherche d'une cassure de structure (BOS) après chaque pivot valide.
- Score de zone sur 10 = BOS + Impulsion + Temps + Fraîcheur + Sweep.
- Suivi de mitigation : une zone traversée à la clôture perd sa validité et
  disparaît de l'affichage par défaut.
- Affichage simplifié (retour d'expérience sur la version MQL5) : peu de
  zones affichées, seulement les valides et déjà correctes, gros mot
  **"DEMANDE"/"OFFRE"** collé au prix actuel.

**Multi-timeframe** : la détection tourne sur `InpZoneTF` (M15 par défaut)
récupéré via `request.security()` en mode non-repaint, quel que soit le
graphique sur lequel l'indicateur est posé.

## Ce qu'il ne fait PAS (encore)

Le pattern Wyckoff (Accumulation → Manipulation → MSS) n'est pas inclus
dans cette première version — il est désactivé par défaut dans la version
MQL5 également. Peut être ajouté en suivi si besoin.

## Installation

1. Ouvre TradingView → **Pine Editor** (en bas de l'écran).
2. Crée un nouveau script vierge, colle le contenu de `ZeusVision.pine`.
3. **Enregistrer** puis **Ajouter au graphique**.
4. Ajuste les inputs si besoin (groupes "Affichage des zones" et
   "Paramètres de détection").

## ⚠️ Statut de validation

Comme la version MQL5, ce script n'a **pas pu être compilé/testé dans cet
environnement** (pas d'accès à l'éditeur Pine de TradingView ici) — chaque
fonction a été relue manuellement pour la syntaxe Pine v6 (types, boucles,
tuples, `request.security` non-repaint), mais une première compilation
réelle dans TradingView reste nécessaire. **Si TradingView signale une
erreur de compilation, renvoie-la telle quelle pour correction.**

Si le chargement est lent sur un historique très long, réduis
`InpMaxHtfBars` (barres HTF conservées en mémoire, 1500 par défaut) — le
scan de détection re-parcourt tout ce tampon à chaque nouvelle bougie HTF
fermée.

## Différences connues vs MQL5/Python

- Le calcul tourne sur `InpZoneTF` directement (M15 par défaut) via
  `request.security()`, plutôt que sur un resampling M1→M15 manuel comme
  le fait l'EA MQL5 pour la parité stricte avec le moteur Python. Sur la
  plupart des brokers/flux, le M15 natif de TradingView et le M15
  resamplé depuis M1 doivent coïncider, mais une divergence est possible
  si le flux de données diffère (mêmes réserves que documentées dans
  `mt5/ZeusP11/README.md`).
