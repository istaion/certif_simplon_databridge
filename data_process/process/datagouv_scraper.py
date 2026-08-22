"""
Scraping des pages data.gouv.fr pour récupérer les liens de téléchargement
à jour des jeux de données statiques utilisés dans data_process/data/
(fr-en-ips-lycees-ap2022.csv, fr-en-annuaire-education.csv, ...).

Les pages de dataset data.gouv.fr sont rendues côté serveur (SSR) : le HTML
brut renvoyé par une simple requête GET contient déjà l'identifiant et le
titre de chaque ressource, sans avoir besoin d'exécuter de JavaScript
(vérifié : `curl` seul suffit à retrouver ces informations). BeautifulSoup +
requests sont donc largement suffisants ici — Scrapy (ou un navigateur
piloté type Selenium/Playwright) serait disproportionné pour scraper 1 ou 2
pages ponctuelles sans pagination ni navigation.

Chaque ressource a une URL de téléchargement pérenne, indépendante du nom
de fichier physique côté data.gouv.fr :
    https://www.data.gouv.fr/api/1/datasets/r/{resource_id}
Cette URL redirige toujours vers la version la plus récente du fichier.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; IANordDataBridge/1.0)"}
_DOWNLOAD_URL_TEMPLATE = "https://www.data.gouv.fr/api/1/datasets/r/{resource_id}"
_RESOURCE_TITLE_ID_RE = re.compile(r"^resource-([0-9a-f-]{36})-title$")

_DATA_DIR = Path(__file__).parent.parent / "data"

# Jeux de données suivis : clé -> (page dataset, titre exact de la ressource
# à télécharger, nom du fichier de destination dans data_process/data/)
DATASETS: dict[str, dict[str, str]] = {
    "ips_lycees": {
        "dataset_url": "https://www.data.gouv.fr/datasets/indices-de-position-sociale-dans-les-lycees-a-partir-de-2022",
        "resource_title": "fr-en-ips-lycees-ap2022.csv",
        "dest_filename": "fr-en-ips-lycees-ap2022.csv",
    },
    "annuaire_education": {
        "dataset_url": "https://www.data.gouv.fr/datasets/annuaire-de-leducation",
        "resource_title": "fr-en-annuaire-education.csv",
        "dest_filename": "fr-en-annuaire-education.csv",
    },
}


@dataclass
class DatasetResource:
    resource_id: str
    title: str
    download_url: str


def fetch_dataset_resources(dataset_url: str, timeout: int = 30) -> list[DatasetResource]:
    """Scrape la page HTML d'un dataset data.gouv.fr et retourne ses ressources."""
    resp = requests.get(dataset_url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    resources: list[DatasetResource] = []
    for h3 in soup.find_all("h3", id=_RESOURCE_TITLE_ID_RE):
        resource_id = _RESOURCE_TITLE_ID_RE.match(h3["id"]).group(1)
        title_div = h3.find("div", attrs={"max-lines": "1"})
        title = title_div["text"].strip() if title_div and title_div.has_attr("text") else ""
        resources.append(DatasetResource(
            resource_id=resource_id,
            title=title,
            download_url=_DOWNLOAD_URL_TEMPLATE.format(resource_id=resource_id),
        ))

    logger.info(f"[datagouv_scraper] {len(resources)} ressource(s) trouvée(s) sur {dataset_url}")
    if not resources:
        logger.warning(
            "[datagouv_scraper] Aucune ressource trouvée — la structure HTML de "
            "data.gouv.fr a peut-être changé (sélecteur h3#resource-*-title à revérifier)."
        )
    return resources


def find_resource_by_title(resources: list[DatasetResource], title: str) -> DatasetResource:
    """Retourne la ressource dont le titre correspond exactement, lève ValueError sinon."""
    for r in resources:
        if r.title == title:
            return r
    available = [r.title for r in resources]
    raise ValueError(f"Aucune ressource nommée {title!r} trouvée. Disponibles : {available}")


def download_resource(resource: DatasetResource, dest_path: Path, timeout: int = 60) -> Path:
    """
    Télécharge une ressource vers dest_path.

    Écriture atomique via fichier temporaire + rename : en cas d'erreur
    réseau en cours de téléchargement, le fichier existant n'est jamais
    remplacé par un fichier partiel/corrompu.
    """
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")
    with requests.get(resource.download_url, headers=_HEADERS, timeout=timeout, stream=True) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    size = tmp_path.stat().st_size
    tmp_path.replace(dest_path)
    logger.info(f"[datagouv_scraper] {resource.title} téléchargé ({size:,} octets) → {dest_path}")
    return dest_path


def refresh_dataset(key: str, data_dir: Path = _DATA_DIR) -> Path:
    """
    Scrape la page data.gouv.fr du jeu de données `key` (cf. DATASETS) et
    télécharge la ressource correspondante dans data_dir, en écrasant le
    fichier existant.
    """
    if key not in DATASETS:
        raise ValueError(f"Jeu de données inconnu : {key!r}. Valeurs : {list(DATASETS)}")

    config = DATASETS[key]
    resources = fetch_dataset_resources(config["dataset_url"])
    resource = find_resource_by_title(resources, config["resource_title"])
    dest_path = data_dir / config["dest_filename"]
    return download_resource(resource, dest_path)
