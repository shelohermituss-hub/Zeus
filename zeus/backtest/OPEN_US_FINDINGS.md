# Stratégie "Open US" — synthèse de l'exploration (2026-08-10)

## Contexte

Demande initiale : une stratégie basée sur l'ouverture actions US (9h30 ET).
Deux approches structurellement différentes ont été implémentées et testées
en profondeur :

- **`open_us_strategy.py`** (Wyckoff sweep + retournement) — réutilise le
  `WyckoffDetector` déjà validé en production (`SDStrategy`), avec un filtre
  horaire limitant les signaux à la fenêtre suivant l'ouverture US.
- **`open_us_orb_strategy.py`** (Opening Range Breakout) — continuation
  directe sur cassure du range construit dans les premières minutes après
  l'ouverture, sans confirmation de retournement.

## Ce qui a été testé

| Étape | Script | Portée |
|---|---|---|
| 1 | `run_open_us_backtest.py` | Config par défaut, XAUUSD 2024/2025/2026H1 |
| 2 | `run_open_us_sweep.py` | 435 configs Wyckoff (fenêtre × variante détecteur × score min × R:R), XAUUSD |
| 3 | `run_open_us_orb_sweep.py` | 432 configs ORB (durée range × fenêtre recherche × range min × R:R), XAUUSD |
| 4 | `run_open_us_regime_filter.py` | Filtre de volatilité (ATR journalier percentile glissant), XAUUSD |
| 5 | `run_open_us_gbpusd_check.py` | 40 configs (Wyckoff + ORB), GBPUSD 2023/2024/2025 |
| 6 | `run_open_us_gbpusd_oos_check.py` | Validation hors-échantillon (GBPUSD juin 2026, jamais vu en 1-5) |

Au total, **~900 configurations distinctes** ont été évaluées à travers deux
instruments (XAUUSD, GBPUSD), deux styles d'entrée (retournement, breakout),
et plusieurs régimes de marché (2023 à mi-2026).

## Résultats

- **XAUUSD** : aucune configuration (Wyckoff ou ORB) n'est profitable sur
  2024 ET 2025 ET 2026H1 simultanément. 2024 est systématiquement le
  maillon faible.
- **Filtre de volatilité** : un filtre "ne trade que si l'ATR journalier
  récent est au-dessus de sa médiane glissante" améliore nettement 2024 et
  2025 pour l'ORB, mais 2024 reste négatif dans tous les cas testés.
  Signal directionnellement cohérent avec l'hypothèse d'un effet de régime,
  mais non concluant avec seulement 3 années indépendantes.
- **GBPUSD** : Wyckoff avec `min_score=0.0, risk_reward=3.0` est la SEULE
  configuration trouvée positive sur les trois années testées
  (2023 : +3 à +5R · 2024 : +8 à +12R · 2025 : +6 à +9R selon la fenêtre
  90/120 min), avec un win rate ≈ 25.6–26.2 % (seuil de rentabilité à
  R:R=3 : 25 %) — une marge très fine mais positive et cohérente.
- **Validation hors-échantillon (GBPUSD juin 2026, jamais utilisé dans la
  recherche de paramètres)** : **l'edge ne se confirme PAS** —
  R=-8, n=20, WR=15 % (vs ≈26 % en échantillon). Sur seulement 20 trades le
  résultat n'est pas définitif à lui seul, mais combiné à la marge déjà très
  fine observée en échantillon, ceci est cohérent avec un résultat trouvé
  par recherche exhaustive sur ~900 configurations (sur-ajustement) plutôt
  qu'un edge réel.

## Conclusion

**Aucune configuration testée n'a d'edge robuste et validé hors-échantillon.**
Conformément à la règle du projet ("ne jamais conclure qu'une stratégie est
bonne sans backtest réaliste"), ni `OpenUSStrategy` ni `OpenUSORBStrategy` ne
doivent être considérées comme prêtes pour le paper trading dans leur forme
actuelle. Le code et les tests restent dans le dépôt (logique correcte,
bien testée, réutilisable) — c'est la recherche de configuration profitable
qui n'a pas abouti, pas un défaut d'implémentation.

## Pistes non explorées (pour une reprise future)

- Autres sessions (ouverture Londres, ouverture Asie) ou autres instruments
  (indices, autres paires forex) — le concept "ouverture de session"
  pourrait fonctionner ailleurs même s'il ne fonctionne pas ici.
- Filtre de biais directionnel HTF (H4/daily) en plus du filtre horaire —
  n'a pas été testé faute de temps, pourrait réduire le nombre de faux
  signaux à contre-tendance.
- Plus d'années de données XAUUSD (seules 2024–2026H1 étaient disponibles
  dans ce dépôt) pour donner plus de puissance statistique au test de
  régime de volatilité.
