# IAnord Data Bridge

Datalake et API de mise à disposition des données pour IANord (éditeur de logiciels de restauration collective — WebGerest et WebResto).

Ce projet consolide les données issues des deux logiciels d'IANord (WebGerest, legacy HFSQL ; WebResto, nouvelle stack clusterisée) dans un datalake souverain (OVH Data Platform, Trino/Iceberg), et les expose via une API REST pour les besoins data/IA (reporting, statistiques, prédiction de fréquentation).

## Architecture

```
WebGerest API ─┐
                ├─► fetch (data_process/fetch) ─► transform/clean (data_process/process,
WebResto API  ──┘                                 job_forepass/*) ─► Trino/Iceberg (OVH)
                                                                          │
                                                                          ▼
                                                          API/app.py (FastAPI) ─► consommateurs
                                                          (dashboards, prediction_passages, ...)
```

- **`data_process/`** : clients de fetch (Webgerest, Webresto), schémas de tables (`schemas_webgerest.py`, `schemas_webresto.py`, `ddl.py`), transformations génériques, client Trino.
- **`job_forepass/`** : jobs de chargement par table, jobs de nettoyage (déduplication), jobs de reporting/statistiques.
- **`API/app.py`** : API FastAPI exposant les routes de synchronisation, de consultation des données et des prédictions.
- **`prediction_passages/`** : modèles de prédiction de fréquentation (ARIMA, Prophet, XGBoost, ensemble).
- **`monitoring/`** : configuration Prometheus/Grafana.
- **`.woodpecker/`** : pipelines CI/CD (build, test, déploiement).

## Prérequis

- Docker et Docker Compose.
- Un accès OVH Data Platform (clés API Trino, lecture et/ou écriture).
- Des identifiants d'accès aux API Webgerest et/ou Webresto pour les établissements à synchroniser.

## Installation

1. Cloner le dépôt.
2. Copier le fichier d'exemple de configuration et renseigner vos propres identifiants :

   ```bash
   cp .env.example .env
   # éditer .env avec vos identifiants — ne jamais commiter ce fichier ni y placer
   # des identifiants de production partagés
   ```

3. Démarrer l'API :

   ```bash
   docker compose up --build
   ```

4. Vérifier que l'API est fonctionnelle :

   ```bash
   curl http://localhost:8000/health
   # {"status": "healthy", ...}
   ```

5. Consulter la documentation interactive de l'API (générée automatiquement par FastAPI) :

   http://localhost:8000/docs

6. (Optionnel, nécessite de vrais identifiants Webgerest/Webresto/OVH) Déclencher une synchronisation de test :

   ```bash
   curl -X POST "http://localhost:8000/sync/{job_name}?environnement_client=..." \
     -H "Content-Type: application/json" \
     -d '{"base_url": "...", "prefix_table": "..."}'
   ```

Une installation réussie signifie que l'API démarre et répond sur `/health` et `/docs` — cela ne nécessite pas de données réelles de production. La synchronisation effective des tables Trino nécessite en revanche de vrais accès Webgerest/Webresto/OVH.

## Variables d'environnement

| Variable | Rôle |
|---|---|
| `OVH_API_KEY` / `OVH_SECRET_KEY` | Accès Trino en écriture (jobs de synchronisation) |
| `OVH_API_KEY_READ_ONLY` / `OVH_SECRET_KEY_READ_ONLY` | Accès Trino en lecture seule (routes de consultation) |
| `CLIENT_KEY_ADMIN` / `CLIENT_SECRET_ADMIN` | Identifiants du rôle `admin` de l'API |
| `CLIENT_KEY_WEBGEREST_RO` / `CLIENT_SECRET_WEBGEREST_RO` | Identifiants du rôle `webgerest_readonly` |
| `CLIENT_KEY_WEBRESTO_RO` / `CLIENT_SECRET_WEBRESTO_RO` | Identifiants du rôle `webresto_readonly` |
| `WEBGEREST_SECRETS` / `WEBRESTO_SECRETS` | Secrets par établissement/serveur source (JSON) |
| `SUPERSET_URLS` / `SUPERSET_USERNAME` / `SUPERSET_PASSWORD` | Intégration Superset (dashboards) |

Voir `.env.example` pour le détail des formats attendus.

## Développement

Dépendances de l'API : `requirements-api.txt`. Dépendances du projet complet (notebooks, prédiction) : `pyproject.toml`.

```bash
uv pip install -r requirements-api.txt
uvicorn API.app:app --reload
```

## Tests

Trois suites, aucune n'a besoin d'un vrai accès OVH/Trino/Webgerest/Webresto (connexions externes mockées) :

- `API/tests/` : routes `/predict/*` (authentification, rôles, paramètres invalides, succès nominal).
- `prediction_passages/tests/` : préparation des données (schéma, valeurs aberrantes, découpage train/test, absence de fuite via les moyennes mobiles décalées) et forecasters individuels (non-régression MAE/MAPE, cas limites — établissement peu de données/valeurs manquantes/établissement inconnu).
- `data_process/tests/` : non-régression sur la gateway WebResto (`test_bankdetail_gateway.py`) — verrouille le contrat GET + paramètres à plat de la route `/findAll/bankDetails` après sa migration côté back (l'ancien contrat POST + corps imbriqué répond désormais 404).

Installation des dépendances de test (groupe `dev` de `pyproject.toml`) :

```bash
uv sync --group dev
# ou : pip install pytest httpx pytest-cov
```

Exécution :

```bash
pytest API/tests/ prediction_passages/tests/ data_process/tests/ -v
```

Avec couverture de code :

```bash
pytest API/tests/ prediction_passages/tests/ data_process/tests/ --cov=API --cov=prediction_passages/src --cov=data_process --cov-report=term-missing
```

## CI/CD

Pipelines Woodpecker (`.woodpecker/`) : build de l'image Docker, déploiement via Ansible sur les événements `tag`/`release` de la branche `main`.
