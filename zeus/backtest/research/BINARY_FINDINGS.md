# Options binaires (Pocket Option, payout 92 %) — conclusions d'étude

Script : `run_binary_study.py` — résultats complets : `data/results/binary_study.json`.

## Protocole
- Seuil de rentabilité : win rate > 1/(1+0,92) = **52,08 %** (égalités remboursées, exclues du calcul).
- Entraînement : M1 2023-2025, 11 jeux (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, USDCHF).
- Hors échantillon : juin 2026 (GBPUSD, NZDUSD, USDCAD), jamais utilisé pour choisir.
- Entrée à l'ouverture de la bougie suivant le signal, slippage adverse de 0,2 pip, 1 trade ouvert à la fois.
- Grille : 12 variantes retour à la moyenne + 12 variantes momentum × expirations 1/3/5 min (72 configurations).
- Critère : borne basse Wilson 95 % > 52,08 % ET win rate > seuil sur ≥ 80 % des jeux.

## Résultats
**Aucune configuration ne passe le critère à 0,2 pip de slippage.**

| Famille | Meilleur résultat (train) | Verdict |
|---|---|---|
| Momentum (séries de bougies) | ≈ 49 % sans slippage, 38-43 % avec | Aucun edge : rejeté. |
| Retour à la moyenne, filtres larges (Bollinger 2σ) | 40-48 % | Rejeté. |
| Retour à la moyenne extrême (Bollinger 3σ, RSI(7) 15/85, toutes heures, expiration 5 min) | 52,99 %, borne basse 51,84 % | Edge réel mais **sous le seuil** une fois les coûts inclus. |

Sensibilité du candidat (Bollinger 3σ, RSI 15/85, 5 min) au slippage d'entrée :

| Slippage | Win rate train (n = 7 308) | Borne basse 95 % | Jeux > seuil | Hors échantillon (n = 156) |
|---|---|---|---|---|
| 0 pip | 56,96 % | 55,81 % | 11/11 | 58,06 % |
| 0,1 pip | 54,69 % | 53,54 % | 11/11 | 57,14 % |
| 0,15 pip | 53,78 % | 52,63 % | 9/11 | 55,77 % |
| 0,2 pip | 52,99 % | 51,84 % | 8/11 | 55,48 % |
| 0,3 pip | 51,20 % | 50,04 % | 2/11 | 53,33 % |

## Interprétation
- À 1 minute, deux stratégies de sens opposés tombent toutes deux à ~39 % : à cet horizon, le
  mouvement moyen est du même ordre que le coût d'entrée. **L'expiration 1 min est inexploitable.**
- Le candidat retour à la moyenne n'est rentable que si l'écart entre le prix de notre flux et
  le prix d'exécution Pocket Option reste **≤ ~0,1 pip**. Cet écart est inconnu : Pocket Option
  utilise son propre flux de prix.
- Les données HistData sont des cotations bid ; une partie des « excès » M1 peut être du bruit
  de microstructure absent du flux Pocket Option. Seul un paper trading sur le flux du broker
  peut trancher.
- 72 configurations testées : le candidat a été choisi a posteriori, le hors échantillon
  (156 trades) est trop petit pour le confirmer seul.

## Décision
Pas de trading réel. Le candidat passe en **paper trading obligatoire** avec mesure, à chaque
signal, de l'écart entre notre prix d'entrée et le prix d'exercice affiché par Pocket Option.
Critère pour aller plus loin : écart médian ≤ 0,1 pip et win rate paper > 52,08 % sur au moins
300 trades.

---

## Recherche élargie (`run_binary_search.py`) — artefact du rollover

Six familles (Bollinger/RSI, pullback RSI(2) avec/sans tendance, épuisement de série,
étirement Keltner avec/sans rejet, Donchian cassure/fade, momentum) × M5/M15/H1 ×
expiration 1-3 bougies = 333 configurations, 42 jeux (18 instruments). Sélection
2023-2024, validation 2025, contrôle 2026. Écart d'exécution 0,2 pip.

### Premier passage : 60 % « validé »… par un artefact
Bollinger 3σ + RSI(7) 15/85, M5, expiration 15 min : 61,4 % (2023-2024), 60,4 % (2025, n = 1 448).
Le détail par heure UTC montre que tout l'avantage vient du **rollover (20h-23h UTC)** :

| Heures UTC | Win rate | Trades |
|---|---|---|
| 20h | 72,6 % | 201 |
| 21h | 84,3 % | 892 |
| 22h | 83,2 % | 564 |
| autres heures | ≈ 50 % | ≈ 4 000 |

Les CSV HistData sont des cotations **bid** : à l'élargissement du spread du rollover, le bid
plonge puis revient, ce qui crée des « excès » qui se résorbent mécaniquement. Les croisées
(spread de rollover le plus large) affichaient les meilleurs scores. Cet effet n'existe pas sur
un flux de prix moyen comme celui de Pocket Option. Résultats bruts conservés dans
`data/results/binary_search_with_rollover.json`.

Le candidat M1 de la première étude était touché aussi : 52,99 % toutes heures, **51,75 %**
hors rollover (borne basse 50,3 %).

### Second passage : rollover exclu (signaux 20h-24h UTC écartés)
**0 configuration sur 333 présélectionnée.** Meilleure borne basse d'entraînement : 52,1 %
(win rate 52,9 %), retombée à 49,3 % sur 2025. Toutes familles et timeframes confondus,
les win rates 2025 sont entre 47 et 52 %.

Filtres complémentaires testés sur le retour à la moyenne M5 (analyse ad hoc) :
- régime de range (efficiency ratio 20 bougies, 5 tranches) : 50-52 % dans chaque tranche ;
- confluence RSI H1 en excès dans le sens du pari : 52,1 % (train), 50,5 % (2025).

### Conclusion
Les indicateurs techniques classiques n'ont **aucun avantage exploitable** en options binaires
forex/or de M1 à H1 une fois l'artefact du rollover retiré. Viser 60-70 % de réussite avec ces
outils n'est pas réaliste : un tel score en backtest doit être traité comme un artefact jusqu'à
preuve du contraire (contrôler d'abord la répartition horaire).

Règle pour toute étude future sur ces données : **exclure 20h-24h UTC** et vérifier le win rate
par heure avant de conclure.
