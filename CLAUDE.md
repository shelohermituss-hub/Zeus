# Trading Bot Project

## Mission
Construire un bot de trading automatisé robuste, testable, modulaire et sûr en production.
Le code doit être conçu pour la fiabilité avant la performance brute.

## Principes non négociables
- Toujours analyser le dépôt avant de modifier quoi que ce soit.
- Toujours proposer un plan en petites étapes avant d'implémenter.
- Toujours tester la logique critique.
- Toujours séparer stratégie, risk management, exécution, backtesting, paper trading et monitoring.
- Toujours faire fail closed en cas d'ambiguïté ou d'erreur critique.
- Ne jamais envoyer d'ordre sans validation complète du risque.
- Ne jamais hardcoder de secrets.
- Ne jamais déployer en live avant paper trading validé.
- Ne jamais ignorer un test qui échoue.

## Priorités du projet
1. Fiabilité.
2. Contrôle du risque.
3. Lisibilité du code.
4. Testabilité.
5. Observabilité.
6. Performance seulement après stabilité.

## Architecture attendue
Le projet doit être découpé en modules clairs :

- strategy/ : génération des signaux.
- risk/ : règles de taille de position, stop-loss, drawdown, kill switch.
- execution/ : envoi et gestion des ordres.
- backtest/ : simulation historique.
- paper/ : exécution en simulation temps réel.
- monitoring/ : logs, métriques, alertes.
- config/ : paramètres, environnements, validation.
- tests/ : tests unitaires et d'intégration.

## Structure du dépôt
Le code Python vit dans le package `zeus/`, découpé selon l'architecture ci-dessus :

- `zeus/strategy/` : stratégies (une par fichier) et briques de signal (`smc/`, `supply_demand/`).
- `zeus/risk/` : gestionnaire de risque, règles prop firm.
- `zeus/orders/` et `zeus/exchange/` : modèles d'ordre, exécution, connecteurs (paper, live, MT5).
- `zeus/backtest/` : moteurs de simulation réutilisables (engine, data_loader, walk_forward, monte_carlo, report).
- `zeus/paper/` : moteurs de paper trading.
- `zeus/live/` : runner live (P11).
- `zeus/monitoring/`, `zeus/utils/` : alertes, métriques, logs.
- `zeus/config.py` et `config/` : paramètres validés.
- `mt5/` (EA MQL5), `tradingview/` (Pine) : portages hors Python.
- `data/historical/` (données, non versionnées), `data/results/` (résultats de backtest versionnés).
- `tests/unit/`, `tests/integration/` : un fichier de test par module.

Règles de placement :
- Un script d'étude ou de sweep (`run_*.py`) n'est pas un moteur : il ne contient que du paramétrage et des appels au moteur, jamais de logique métier propre.
- Chaque nouvelle famille de stratégie a son sous-dossier dans `strategy/` si elle dépasse un fichier.
- Les options binaires ont des règles de gain/perte propres (payout fixe, expiration) : leur simulation et leur risque doivent être des modules dédiés, jamais des cas particuliers cachés dans les moteurs forex.
- Les conclusions d'étude vont dans un fichier `*_FINDINGS.md` à côté des scripts, pas dans ce fichier.

## Skills Claude Code
Skills installés au niveau du projet dans `.claude/skills/` :

- **graphify** : graphe de connaissances du dépôt. À utiliser pour les étapes 1 à 3 du workflow (comprendre l'existant, résumer l'architecture, repérer les dépendances) avant toute modification non triviale. Si `graphify-out/graph.json` existe, interroger le graphe (`/graphify query`) avant de parcourir les fichiers à la main ; après des changements importants, le mettre à jour (`/graphify . --update`). Nécessite la CLI `graphifyy`. `graphify-out/` n'est pas versionné.
- **ponytail** : solution la plus simple qui résout entièrement la tâche. Actif par défaut pour l'implémentation (niveau `full`). `/ponytail-review` pour relire un diff, `/ponytail-audit` et `/ponytail-debt` pour repérer la complexité inutile.

Ordre de priorité : ce fichier prime sur les skills. Ponytail ne justifie jamais de supprimer un contrôle de risque, un test, une validation d'entrée, une gestion d'erreur ou un log d'audit.

## Règles de développement
- Une responsabilité par module.
- Fonctions petites et explicites.
- Noms clairs et sans ambiguïté.
- Aucune logique métier cachée dans des helpers génériques.
- Pas de duplication inutile.
- Préférer des structures déterministes et vérifiables.
- Tout comportement critique doit être couvert par des tests.
- Toute nouvelle logique doit inclure une stratégie de validation.

## Risk management
Le bot doit toujours appliquer les contrôles suivants avant tout ordre :
- Validation du solde disponible.
- Vérification de la taille de position.
- Vérification du stop-loss.
- Vérification du take-profit si utilisé.
- Vérification du drawdown journalier.
- Vérification du drawdown maximal global.
- Vérification de l'exposition totale.
- Vérification du nombre de positions ouvertes.
- Vérification de l'état de connexion broker/exchange.
- Kill switch disponible et testable.

Règle de base :
- Si une condition est incertaine, l'ordre est refusé.

## Execution rules
- Gérer les timeouts, retries et erreurs d'API.
- Vérifier les confirmations d'ordre.
- Empêcher les envois doubles.
- Logger chaque ordre, rejet, annulation et fill.
- Séparer clairement logique de décision et logique d'exécution.
- En cas d'état incohérent, s'arrêter proprement.

## Backtesting rules
- Inclure frais, slippage et latence.
- Utiliser des données propres et datées.
- Tester plusieurs périodes de marché.
- Évaluer aussi les pertes, drawdown et stabilité.
- Ne jamais conclure qu'une stratégie est bonne sans backtest réaliste.

## Paper trading rules
- Le paper trading est obligatoire avant le live.
- Le moteur de décision doit être identique au live.
- Les différences entre simulation et exécution réelle doivent être tracées.
- Les anomalies doivent être investiguées avant mise en production.

## Logging and observability
- Utiliser des logs structurés.
- Inclure timestamp, symbole, action, taille, prix, raison, résultat.
- Suivre PnL, drawdown, win rate, taux d'erreur API et latence.
- Prévoir des alertes sur anomalies et échecs critiques.
- Conserver un audit trail complet.

## Security rules
- Les clés API doivent venir des variables d'environnement.
- Aucune clé dans le code, les tests ou les exemples.
- Appliquer le principe du moindre privilège.
- Valider toutes les entrées externes.
- Ne jamais exposer les secrets dans les logs.

## Testing rules
- Écrire des tests unitaires pour toute logique critique.
- Ajouter des tests d'intégration pour le flux complet.
- Ajouter des tests de non-régression après correction de bug.
- Mock les dépendances externes quand nécessaire.
- Les tests doivent être reproductibles.

## Workflow demandé à Claude
Quand tu travailles sur ce dépôt :
1. Lis les fichiers de base et comprends l'existant.
2. Résume l'architecture actuelle.
3. Identifie les risques et dettes techniques.
4. Propose un plan en étapes courtes.
5. Implémente une seule étape à la fois.
6. Exécute les tests liés à l'étape.
7. Corrige uniquement ce qui est nécessaire.
8. Attends une validation avant de poursuivre.

## Format de sortie attendu
Quand tu modifies le code, réponds avec :
- Résumé de l'état actuel.
- Plan proposé.
- Fichiers modifiés.
- Changements effectués.
- Tests exécutés.
- Résultat des tests.
- Prochaine étape recommandée.

## Definition of Done
Une tâche est terminée seulement si :
- le code est cohérent,
- les tests passent,
- le risque est contrôlé,
- le comportement est vérifiable,
- la modification est documentée si nécessaire.

## Notes de travail
- Favoriser la simplicité avant l'optimisation.
- Refactoriser quand cela réduit la complexité.
- Garder l'historique des décisions importantes.
- Ne pas transformer ce fichier en journal de session.
- Mettre à jour ce fichier seulement pour les règles stables et importantes.
