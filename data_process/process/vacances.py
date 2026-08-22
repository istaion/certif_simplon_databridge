"""
Parsing des fichiers ICS de vacances scolaires (Zones A, B, C) → DataFrame.

Chaque VEVENT devient un row avec :
  zone         : A | B | C
  school_year  : ex. "2023-2024"  (calculé depuis date_debut)
  type_vacances: Toussaint | Noel | Hiver | Printemps | Ascension | Ete
  date_debut   : premier jour de vacances
  date_fin     : dernier jour de vacances (DTEND iCal est exclusif → DTEND - 1 jour)

Événements exclus :
  - "prérentrée Enseignants" (chevauchement été, spécifique enseignants)
  - "Début des Vacances" (marqueur ponctuel sans durée réelle)
"""

import logging
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent.parent / "data"

ICS_FILES: list[tuple[Path, str]] = [
    (_DATA_DIR / "Zone-A.ics", "A"),
    (_DATA_DIR / "Zone-B.ics", "B"),
    (_DATA_DIR / "Zone-C.ics", "C"),
]

# Mots-clés dans SUMMARY → type normalisé (ordre important : plus spécifique en premier)
_SUMMARY_MAP: list[tuple[str, str]] = [
    ("Toussaint",  "Toussaint"),
    ("Noël",       "Noel"),
    ("Hiver",      "Hiver"),
    ("Printemps",  "Printemps"),
    ("Ascension",  "Ascension"),
    ("Été",        "Ete"),
]

# Fragments de SUMMARY à ignorer complètement
_EXCLUDE = ("prérentrée", "Début des")


def _school_year(d: date) -> str:
    """Retourne l'année scolaire (ex: '2023-2024') pour une date donnée."""
    if d.month >= 9:
        return f"{d.year}-{d.year + 1}"
    return f"{d.year - 1}-{d.year}"


def _parse_ics(path: Path, zone: str) -> pd.DataFrame:
    """Parse un fichier ICS et retourne un DataFrame de périodes de vacances."""
    text = path.read_text(encoding="utf-8")
    events = re.findall(r"BEGIN:VEVENT(.*?)END:VEVENT", text, re.DOTALL)

    records = []
    for ev in events:
        m_summary = re.search(r"SUMMARY:(.+)", ev)
        m_start   = re.search(r"DTSTART;VALUE=DATE:(\d{8})", ev)
        m_end     = re.search(r"DTEND;VALUE=DATE:(\d{8})", ev)

        if not (m_summary and m_start and m_end):
            continue

        summary = m_summary.group(1).strip()

        if any(kw in summary for kw in _EXCLUDE):
            continue

        type_vacances = next(
            (t for kw, t in _SUMMARY_MAP if kw in summary), None
        )
        if type_vacances is None:
            logger.warning(f"Zone {zone} — SUMMARY non reconnu, ignoré : {summary!r}")
            continue

        date_debut = date(
            int(m_start.group(1)[:4]),
            int(m_start.group(1)[4:6]),
            int(m_start.group(1)[6:]),
        )
        # DTEND est exclusif en iCal ; on soustrait 1 jour pour avoir le dernier jour réel.
        # Si DTSTART == DTEND (donnée incomplète), on garde date_debut comme date_fin.
        dtend_raw = date(
            int(m_end.group(1)[:4]),
            int(m_end.group(1)[4:6]),
            int(m_end.group(1)[6:]),
        )
        date_fin = max(date_debut, dtend_raw - timedelta(days=1))

        records.append({
            "zone":          zone,
            "school_year":   _school_year(date_debut),
            "type_vacances": type_vacances,
            "date_debut":    date_debut,
            "date_fin":      date_fin,
        })

    logger.info(f"Zone {zone} ({path.name}) : {len(records)} périodes parsées")
    return pd.DataFrame(records)


def load_vacances() -> pd.DataFrame:
    """
    Charge et concatène les périodes de vacances des trois zones (A, B, C).
    Retourne un DataFrame avec colonnes :
      zone, school_year, type_vacances, date_debut, date_fin
    """
    frames = []
    for path, zone in ICS_FILES:
        if not path.exists():
            logger.warning(f"Fichier ICS introuvable, ignoré : {path.name}")
            continue
        frames.append(_parse_ics(path, zone))

    if not frames:
        return pd.DataFrame(columns=["zone", "school_year", "type_vacances", "date_debut", "date_fin"])

    df = pd.concat(frames, ignore_index=True)
    logger.info(
        f"Vacances total : {len(df)} périodes, "
        f"{df['school_year'].nunique()} années scolaires, "
        f"zones {sorted(df['zone'].unique().tolist())}"
    )
    return df
