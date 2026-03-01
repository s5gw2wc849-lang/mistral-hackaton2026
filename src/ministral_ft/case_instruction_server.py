from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import threading
import tempfile
import unicodedata
from dataclasses import dataclass
from datetime import UTC, date, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    from faker import Faker
except Exception:  # pragma: no cover - optional dependency during bootstrap
    Faker = None  # type: ignore[assignment]

DEFAULT_TARGET_TOTAL_CASES = 5000
DEFAULT_SEED = 42
DEFAULT_CORPUS_FILE = Path("data/succession_e2e/e2e_cases.jsonl")
DEFAULT_MASTER_SCHEMA_FILE = Path("../w5/glinerExtract/schema/schema.full.json")
CONFIG_FILENAME = "config.json"
ISSUED_FILENAME = "issued_instructions.jsonl"
SUBMITTED_FILENAME = "generated_cases.jsonl"
SUMMARY_JSON_FILENAME = "summary.json"
SUMMARY_MD_FILENAME = "summary.md"
GENERATED_TRAIN_FILENAME = "generated_cases_train_mistral.jsonl"
FULL_TRAIN_FILENAME = "full_training_cases_mistral.jsonl"
FORBIDDEN_CAPS_UNDERSCORE_RE = re.compile(r"\b[A-Z]{2,}(?:_[A-Z0-9]{2,})+\b")
FORBIDDEN_PYTHON_BOOL_RE = re.compile(r"\b(?:True|False)\b")
FORBIDDEN_PATH_DUMP_RE = re.compile(r"\s>\s")
FORBIDDEN_ENUM_BASIC_RE = re.compile(
    r"\b(?:CELIBATAIRE|MARIE|PACSE|DIVORCE|VEUF|JOURS|MOIS|ANNEES)\b"
)
FORBIDDEN_SCHEMAISH_PHRASES_RE = re.compile(
    r"\b(?:famille\s+defunt|contexte\s+procedure|patrimoine\s+actifs?|liberalites?\s+donations?)\b",
    re.IGNORECASE,
)
FORBIDDEN_SCHEMAISH_DEFUNT_FIELDS_RE = re.compile(
    r"\bdefunt\s+(?:date\s+deces|date\s+naissance|age\s+au\s+deces)\b",
    re.IGNORECASE,
)
MAX_SEMICOLONS_IN_CASE_TEXT = 10
MAX_COLONS_IN_CASE_TEXT = 10
PAIR_TRAINING_SYSTEM_PROMPT = (
    "Tu extrais les informations d'un énoncé de succession en français. "
    "Tu réponds uniquement par du TOON valide conforme au schéma cible attendu."
)

PERSONA_TARGETS = {
    "enfant": 0.18,
    "conjoint": 0.12,
    "beau_enfant": 0.09,
    "fratrie": 0.08,
    "notaire": 0.08,
    "avocat": 0.07,
    "partenaire_pacs": 0.07,
    "concubin": 0.06,
    "associe": 0.07,
    "petit_enfant": 0.05,
    "tiers": 0.05,
    "narrateur_neutre": 0.08,
}
VOICE_TARGETS = {
    "premiere_personne": 0.45,
    "troisieme_personne": 0.35,
    "note_dossier": 0.10,
    "parole_rapportee": 0.10,
}
FORMAT_TARGETS = {
    "question_directe": 0.22,
    "mail_brouillon": 0.18,
    "recit_libre": 0.22,
    "note_professionnelle": 0.14,
    "oral_retranscrit": 0.14,
    "message_conflictuel": 0.10,
}
LENGTH_TARGETS = {
    "court": 0.18,
    "moyen": 0.42,
    "long": 0.32,
    "tres_long": 0.08,
}
NOISE_TARGETS = {
    "propre": 0.42,
    "legeres_fautes": 0.22,
    "fautes_et_abreviations": 0.17,
    "ambigu": 0.16,
    "tres_brouillon": 0.03,
}
NUMERIC_TARGETS = {
    "sans_montant": 0.06,
    "un_montant": 0.26,
    "plusieurs_montants": 0.38,
    "montants_et_dates": 0.30,
}
DATE_PRECISION_TARGETS = {
    "aucune": 0.15,
    "approx": 0.20,
    "exacte": 0.65,
}
COMPLEXITY_TARGETS = {
    "simple": 0.20,
    "intermediaire": 0.40,
    "complexe": 0.24,
    "hard_negative": 0.16,
}
TOPIC_TARGETS = {
    "ordre_heritiers": 0.08,
    "famille_recomposee": 0.12,
    "regimes_matrimoniaux": 0.08,
    "donations_reduction": 0.10,
    "assurance_vie": 0.10,
    "indivision_partage": 0.09,
    "entreprise_dutreil": 0.08,
    "demembrement_usufruit": 0.06,
    "testament_legs": 0.08,
    "dettes_passif": 0.06,
    "pacs_concubinage": 0.07,
    "international_procedure": 0.08,
}
HARD_NEGATIVE_TARGETS = {
    "pas_de_deces_clair": 0.30,
    "infos_incompletes": 0.30,
    "faits_contradictoires": 0.25,
    "hors_perimetre_mal_qualifie": 0.15,
}
HARD_NEGATIVE_INTENSITY_TARGETS = {
    "soft": 0.80,
    "hard": 0.20,
}

PERSONA_LABELS = {
    "enfant": "un enfant du défunt",
    "conjoint": "le conjoint survivant",
    "beau_enfant": "un beau-fils ou une belle-fille",
    "fratrie": "un frère ou une sœur",
    "notaire": "un notaire ou un clerc",
    "avocat": "un avocat en contentieux",
    "partenaire_pacs": "le partenaire de PACS",
    "concubin": "le concubin ou la concubine",
    "associe": "un associé ou coindivisaire",
    "petit_enfant": "un petit-enfant",
    "tiers": "un voisin, aidant ou proche extérieur",
    "narrateur_neutre": "un narrateur externe neutre",
}
VOICE_LABELS = {
    "premiere_personne": "à la première personne",
    "troisieme_personne": "à la troisième personne",
    "note_dossier": "en note de dossier",
    "parole_rapportee": "en parole rapportée",
}
FORMAT_LABELS = {
    "question_directe": "question directe courte",
    "mail_brouillon": "mail brouillon ou message client",
    "recit_libre": "récit libre",
    "note_professionnelle": "synthèse professionnelle",
    "oral_retranscrit": "oral retranscrit avec ponctuation irrégulière",
    "message_conflictuel": "message conflictuel ou familial tendu",
}
LENGTH_LABELS = {
    "court": "court (1 à 3 phrases)",
    "moyen": "moyen (un paragraphe net)",
    "long": "long (paragraphe dense ou deux blocs)",
    "tres_long": "très long (cas détaillé quasi dossier)",
}
NOISE_LABELS = {
    "propre": "français propre, quasiment sans bruit",
    "legeres_fautes": "1 ou 2 fautes crédibles",
    "fautes_et_abreviations": "fautes légères + abréviations réalistes",
    "ambigu": "formulation floue avec zones d'ombre",
    "tres_brouillon": "message très brouillon mais compréhensible",
}
NUMERIC_LABELS = {
    "sans_montant": "aucun montant obligatoire",
    "un_montant": "au moins un montant ou une valeur approximative",
    "plusieurs_montants": "plusieurs montants ou valorisations",
    "montants_et_dates": "montants + au moins une date utile",
}
DATE_PRECISION_LABELS = {
    "aucune": "aucune date imposée",
    "approx": "repères temporels approximatifs",
    "exacte": "au moins une date exacte",
}
COMPLEXITY_LABELS = {
    "simple": "cas simple",
    "intermediaire": "cas intermédiaire",
    "complexe": "cas complexe",
    "hard_negative": "hard negative volontaire",
}
HARD_NEGATIVE_LABELS = {
    "pas_de_deces_clair": "faux ami sans décès clairement exploitable",
    "infos_incompletes": "dossier incomplet avec infos majeures manquantes",
    "faits_contradictoires": "faits contradictoires ou incohérents",
    "hors_perimetre_mal_qualifie": "hors périmètre ou mal qualifié mais proche de la succession",
}
HARD_NEGATIVE_INTENSITY_LABELS = {
    "soft": "hard negative léger, très proche d'un vrai cas",
    "hard": "hard negative dur, plus piégeux et plus bruité",
}
DIMENSION_PURPOSES = {
    "persona": (
        "Définit qui parle ou depuis quel point de vue social / familial / professionnel "
        "le cas est raconté. Cela change le biais du narrateur, son niveau d'information "
        "et le vocabulaire attendu."
    ),
    "voice": (
        "Définit la posture narrative et la grammaire du récit. Cela change la distance "
        "émotionnelle, la clarté et la manière d'exposer les faits."
    ),
    "format": (
        "Définit la forme matérielle du texte. Cela évite que tous les cas ressemblent "
        "à des énoncés scolaires homogènes."
    ),
    "length_band": (
        "Définit la profondeur factuelle attendue. Cela contrôle la quantité de détails "
        "et la densité d'information à inclure."
    ),
    "noise": (
        "Définit le niveau de bruit linguistique. Cela simule des entrées plus ou moins "
        "propres, plus ou moins réalistes côté utilisateur."
    ),
    "numeric_density": (
        "Définit la quantité de chiffres, montants, proportions ou valorisations à faire "
        "apparaître dans le cas."
    ),
    "date_precision": (
        "Définit le niveau de précision temporelle attendu, afin de varier entre absence "
        "de date, repères flous et dates réellement exploitables."
    ),
    "complexity": (
        "Définit la difficulté globale du dossier. Cela contrôle le nombre de couches "
        "juridiques, de tensions factuelles et la part de cas piégeux."
    ),
    "primary_topic": (
        "Définit le coeur juridique du cas. C'est la matière principale qui doit structurer "
        "l'énoncé."
    ),
    "secondary_topic": (
        "Ajoute une seconde couche facultative au dossier pour éviter les cas trop plats. "
        "Le sujet secondaire complique ou enrichit le sujet principal."
    ),
    "hard_negative_mode": (
        "Définit la nature du piège lorsque le cas est volontairement un hard negative. "
        "Ce champ reste inactif si la complexité n'est pas hard negative."
    ),
    "hard_negative_intensity": (
        "Dose la violence du piège sur les hard negatives. Ce champ reste inactif si la "
        "complexité n'est pas hard negative."
    ),
}
PERSONA_DETAILS = {
    "enfant": "Le narrateur connaît souvent bien les faits, mais il peut être émotionnel ou partiel.",
    "conjoint": "Le narrateur met souvent en avant sa protection, ses droits et le patrimoine de couple.",
    "beau_enfant": "Le narrateur est souvent dans un angle conflictuel ou comparatif avec les autres branches.",
    "fratrie": "Le narrateur parle souvent de collatéraux, de tensions familiales et d'égalité entre proches.",
    "notaire": "Le ton attendu est plus sec, structuré et factuel, avec un prisme de dossier.",
    "avocat": "Le ton attendu met davantage l'accent sur le litige, la contestation et les points sensibles.",
    "partenaire_pacs": "Le narrateur met souvent l'accent sur la protection insuffisante ou incertaine du survivant.",
    "concubin": "Le narrateur est souvent dans une situation fragile, mal protégée ou mal comprise.",
    "associe": "Le narrateur met souvent en avant la copropriété, la gestion ou la valeur d'un actif partagé.",
    "petit_enfant": "Le narrateur fait souvent apparaître la représentation, une branche familiale ou un décalage générationnel.",
    "tiers": "Le narrateur est utile pour introduire de l'imprécision ou une compréhension partielle des faits.",
    "narrateur_neutre": "Le narrateur expose les faits sans implication personnelle directe, de façon plus neutre.",
}
VOICE_DETAILS = {
    "premiere_personne": "Le texte doit ressembler à une personne qui expose sa propre situation.",
    "troisieme_personne": "Le texte doit ressembler à une présentation extérieure d'un dossier ou d'un cas d'espèce.",
    "note_dossier": "Le texte doit ressembler à une note interne ou une fiche de dossier.",
    "parole_rapportee": "Le texte doit donner l'impression que les faits sont rapportés, transmis ou reformulés.",
}
FORMAT_DETAILS = {
    "question_directe": "Le cas doit se terminer comme une vraie demande adressée à un professionnel.",
    "mail_brouillon": "Le cas doit ressembler à un message envoyé vite, imparfait mais exploitable.",
    "recit_libre": "Le cas doit dérouler les faits sans plan apparent ni structure scolaire.",
    "note_professionnelle": "Le cas doit avoir une forme sèche, quasi cabinet, quasi-notaire.",
    "oral_retranscrit": "Le cas doit garder une cadence parlée, avec une ponctuation un peu irrégulière.",
    "message_conflictuel": "Le cas doit laisser sentir une tension explicite ou un désaccord familial.",
}
LENGTH_DETAILS = {
    "court": "Le cas doit rester bref mais contenir l'essentiel sans tomber dans le télégraphique.",
    "moyen": "Le cas doit tenir dans un bloc lisible avec un bon niveau de matière.",
    "long": "Le cas doit être nettement développé avec plusieurs informations utiles.",
    "tres_long": "Le cas doit ressembler à un mini-dossier riche, sans basculer dans l'analyse.",
}
NOISE_DETAILS = {
    "propre": "Le texte peut être très propre, avec peu ou pas de défaut volontaire.",
    "legeres_fautes": "Le texte peut contenir 1 ou 2 fautes crédibles, pas davantage.",
    "fautes_et_abreviations": "Le texte doit garder une bonne lisibilité tout en injectant des abréviations réalistes.",
    "ambigu": "Le texte doit comporter au moins une zone d'ombre, un point mal posé ou discutable.",
    "tres_brouillon": "Le texte peut être haché, hésitant ou mal ponctué, mais il doit rester compréhensible.",
}
NUMERIC_DETAILS = {
    "sans_montant": "Les chiffres ne sont pas obligatoires si le cas reste crédible sans eux.",
    "un_montant": "Il faut au moins une valeur ou un ordre de grandeur exploitable.",
    "plusieurs_montants": "Il faut plusieurs chiffres utiles pour enrichir le dossier.",
    "montants_et_dates": "Il faut au moins un montant et une date utile, de préférence bien exploitable.",
}
DATE_PRECISION_DETAILS = {
    "aucune": "Aucune date n'est imposée si cela ne sert pas le cas.",
    "approx": "Les repères temporels peuvent rester flous, relatifs ou approximatifs.",
    "exacte": "Au moins une date doit être réellement exploitable (jour/mois/année ou ISO).",
}
COMPLEXITY_DETAILS = {
    "simple": "Le cas doit rester lisible, direct et peu imbriqué.",
    "intermediaire": "Le cas doit comporter quelques couches factuelles mais rester assez standard.",
    "complexe": "Le cas doit cumuler plusieurs éléments ou tensions sans devenir confus.",
    "hard_negative": "Le cas doit volontairement piéger un extracteur ou un lecteur trop confiant.",
}
HARD_NEGATIVE_MODE_DETAILS = {
    "pas_de_deces_clair": "Le texte doit ressembler à une succession sans poser clairement un décès exploitable.",
    "infos_incompletes": "Une donnée pivot doit manquer, empêchant une lecture trop simple du cas.",
    "faits_contradictoires": "Une contradiction réaliste doit être présente sans être explicitement résolue.",
    "hors_perimetre_mal_qualifie": "Le texte doit sembler successoral alors qu'une partie du problème relève d'autre chose.",
}
HARD_NEGATIVE_INTENSITY_DETAILS = {
    "soft": "Un seul défaut majeur suffit, le cas doit rester très crédible au premier regard.",
    "hard": "Le cas doit cumuler plusieurs confusions tout en restant plausible.",
}
COMMON_MUST_AVOID = [
    "Ne pas donner la solution ni conclure sur les droits exacts.",
    "Ne pas fournir d'analyse juridique, de calcul ou de raisonnement explicatif.",
    "Ne pas répondre en liste de points juridiques ou en checklist.",
    "Ne pas recopier mot pour mot les exemples de référence.",
    "Ne pas remplacer la paire demandée par un texte libre, une checklist ou un pseudo-format.",
]
TOPIC_TEMPLATES: dict[str, dict[str, Any]] = {
    "ordre_heritiers": {
        "label": "ordre des héritiers / dévolution",
        "keywords": ["enfant", "célibataire", "frère", "marié", "représentation"],
        "elements": [
            "préciser les liens de parenté utiles",
            "indiquer s'il existe ou non un testament",
        ],
    },
    "famille_recomposee": {
        "label": "famille recomposée / enfants non communs",
        "keywords": ["recompos", "premier lit", "enfant non commun", "beau", "adoption simple"],
        "elements": [
            "inclure au moins un enfant d'une autre union",
            "laisser un point de friction entre branches familiales",
        ],
    },
    "regimes_matrimoniaux": {
        "label": "régime matrimonial / liquidation préalable",
        "keywords": ["communauté", "séparation de biens", "participation", "récompense"],
        "elements": [
            "mentionner le régime matrimonial ou son absence de contrat",
            "faire apparaître un enjeu de propriété entre époux",
        ],
    },
    "donations_reduction": {
        "label": "donation / rapport / réduction",
        "keywords": ["donation", "hors part", "réduction", "rapport", "donation-partage"],
        "elements": [
            "inclure une libéralité antérieure",
            "laisser planer un doute sur son traitement civil",
        ],
    },
    "assurance_vie": {
        "label": "assurance-vie / bénéficiaires / primes",
        "keywords": ["assurance vie", "AV", "bénéficiaire", "primes exag"],
        "elements": [
            "mentionner un contrat d'assurance-vie ou un bénéficiaire",
            "glisser un doute sur la place du contrat dans le calcul global",
        ],
    },
    "indivision_partage": {
        "label": "indivision / partage bloqué / licitation",
        "keywords": ["indivision", "vendre", "licitation", "occupation"],
        "elements": [
            "faire apparaître au moins deux héritiers en désaccord",
            "inclure un bien difficile à partager",
        ],
    },
    "entreprise_dutreil": {
        "label": "entreprise / titres / Dutreil",
        "keywords": ["société", "parts", "Dutreil", "SARL", "SCI", "fonds"],
        "elements": [
            "inclure des titres, une société ou un outil professionnel",
            "laisser un enjeu de valorisation ou de reprise",
        ],
    },
    "demembrement_usufruit": {
        "label": "démembrement / usufruit / nue-propriété",
        "keywords": ["usufruit", "nue-propriété", "quasi-usufruit", "démembrement"],
        "elements": [
            "inclure un usufruit existant ou à choisir",
            "faire apparaître un effet différé ou une créance future",
        ],
    },
    "testament_legs": {
        "label": "testament / legs / clause contestée",
        "keywords": ["testament", "legs", "olographe", "légataire"],
        "elements": [
            "inclure une disposition testamentaire ou un legs",
            "laisser un doute sur la portée ou la validité de la clause",
        ],
    },
    "dettes_passif": {
        "label": "dettes / passif / déficit",
        "keywords": ["dette", "impôts", "URSSAF", "passif", "déficit"],
        "elements": [
            "inclure un passif significatif",
            "faire sentir une tension sur le règlement des dettes",
        ],
    },
    "pacs_concubinage": {
        "label": "PACS / concubinage",
        "keywords": ["PACS", "concubin", "union libre", "partenaire"],
        "elements": [
            "inclure une relation non matrimoniale",
            "faire apparaître un doute sur la protection du survivant",
        ],
    },
    "international_procedure": {
        "label": "international / procédure / blocage",
        "keywords": ["étranger", "Belgique", "Espagne", "procédure", "mandat", "juge"],
        "elements": [
            "inclure un élément procédural ou international",
            "laisser au moins un point de compétence ou de formalité flou",
        ],
    },
}
TOPIC_SCHEMA_PREFIXES: dict[str, list[tuple[str, ...]]] = {
    "ordre_heritiers": [
        ("famille", "descendants"),
        ("famille", "ascendants"),
        ("famille", "collateraux"),
    ],
    "famille_recomposee": [
        ("famille", "descendants"),
        ("famille", "partenaire"),
        ("famille", "collateraux"),
    ],
    "regimes_matrimoniaux": [
        ("famille", "defunt", "regime_matrimonial"),
        ("famille", "partenaire"),
        ("patrimoine", "actifs"),
        ("patrimoine", "recompenses"),
    ],
    "donations_reduction": [
        ("liberalites", "donations"),
        ("liberalites", "testament"),
        ("liberalites", "legs"),
        ("liberalites", "renonciations_action_reduction"),
        ("liberalites", "raar"),
    ],
    "assurance_vie": [
        ("assurance_vie", "contrats"),
        ("contexte", "procedure", "contestation_clause_beneficiaire_assurance_vie"),
    ],
    "indivision_partage": [
        ("indivision", "gestion"),
        ("indivision", "comptes"),
        ("indivision", "creances"),
        ("operations_de_partage", "licitation"),
        ("operations_de_partage", "attributions_preferentielles"),
        ("operations_de_partage", "soultes_mentionnees"),
    ],
    "entreprise_dutreil": [
        ("patrimoine", "actifs"),
        ("liberalites", "donations"),
        ("operations_de_partage", "attributions_preferentielles"),
    ],
    "demembrement_usufruit": [
        ("patrimoine", "actifs"),
        ("operations_de_partage", "conversion_usufruit"),
    ],
    "testament_legs": [
        ("liberalites", "testament"),
        ("liberalites", "legs"),
        ("contexte", "procedure", "contestation_testament"),
    ],
    "dettes_passif": [
        ("patrimoine", "passifs"),
        ("operations_de_partage", "creances_entre_copartageants"),
    ],
    "pacs_concubinage": [
        ("famille", "partenaire"),
        ("famille", "droits_du_partenaire"),
    ],
    "international_procedure": [
        ("contexte", "international"),
        ("contexte", "procedure"),
        ("famille", "defunt"),
        ("famille", "partenaire"),
    ],
}
TOPIC_REQUIRED_LEAF_PATHS: dict[str, list[tuple[str, ...]]] = {
    "ordre_heritiers": [
        ("famille", "descendants", "enfants", "*", "nom"),
    ],
    "famille_recomposee": [
        ("famille", "descendants", "enfants", "*", "nom"),
        ("famille", "descendants", "enfants", "*", "est_d_une_precedente_union"),
    ],
    "regimes_matrimoniaux": [
        ("famille", "defunt", "regime_matrimonial", "type"),
        ("patrimoine", "actifs", "*", "type"),
        ("patrimoine", "actifs", "*", "propriete", "nature"),
    ],
    "donations_reduction": [
        ("liberalites", "donations", "*", "donateur_nom"),
        ("liberalites", "donations", "*", "beneficiaire_nom"),
        ("liberalites", "donations", "*", "type"),
    ],
    "assurance_vie": [
        ("assurance_vie", "contrats", "*", "libelle"),
        ("assurance_vie", "contrats", "*", "assure_nom"),
    ],
    "indivision_partage": [
        ("contexte", "procedure", "refus_de_vendre_ou_de_partager", "existe"),
        ("operations_de_partage", "licitation", "est_prevue"),
    ],
    "entreprise_dutreil": [
        ("patrimoine", "actifs", "*", "type"),
        ("patrimoine", "actifs", "*", "entreprise", "type"),
        ("patrimoine", "actifs", "*", "entreprise", "est_presente_comme_eligible_dutreil"),
    ],
    "demembrement_usufruit": [
        ("patrimoine", "actifs", "*", "demembrement", "droits_du_defunt"),
    ],
    "testament_legs": [
        ("liberalites", "testament", "existe"),
        ("liberalites", "legs", "*", "beneficiaire_nom"),
        ("liberalites", "legs", "*", "type"),
    ],
    "dettes_passif": [
        ("patrimoine", "passifs", "*", "type"),
        ("patrimoine", "passifs", "*", "valeur"),
    ],
    "pacs_concubinage": [
        ("famille", "partenaire", "nom"),
        ("famille", "partenaire", "lien", "type"),
    ],
    "international_procedure": [
        ("contexte", "international", "professio_juris", "existe"),
        ("contexte", "procedure", "divorce_ou_separation_en_cours", "existe"),
    ],
}
SPARSE_COVERAGE_PREFIXES: list[tuple[str, ...]] = [
    ("famille", "adoption_simple_du_defunt"),
    ("liberalites", "donation_entre_epoux"),
    ("patrimoine", "ameliorations_bien_propre"),
]
SYNTH_FIRST_NAMES = [
    "Jean",
    "Marie",
    "Claire",
    "Thomas",
    "Camille",
    "Hugo",
    "Lucie",
    "Nicolas",
    "Sophie",
    "Julien",
    "Emma",
    "Paul",
    "Lea",
    "Antoine",
]
SYNTH_LAST_NAMES = [
    "Durand",
    "Morel",
    "Lefevre",
    "Martin",
    "Roux",
    "Bernard",
    "Petit",
    "Garcia",
    "Thomas",
    "Robert",
    "Leroy",
    "Girard",
]
SYNTH_CITIES = [
    "Paris",
    "Lyon",
    "Marseille",
    "Nantes",
    "Bordeaux",
    "Lille",
    "Toulouse",
    "Montpellier",
    "Grenoble",
]
SYNTH_COMPANIES = [
    "SARL Atelier Delta",
    "SAS Nova Conseil",
    "SCI Les Tilleuls",
    "SARL Horizon Bois",
    "SAS Aquila Services",
]
SYNTH_INSURERS = [
    "Generali",
    "AXA",
    "MAIF",
    "Credit Agricole Predica",
    "CNP Assurances",
]
FORMAT_REQUIREMENTS = {
    "question_directe": ["terminer comme une vraie question ou une demande de conseil"],
    "mail_brouillon": ["faire sentir un message envoyé vite, sans mise en forme parfaite"],
    "recit_libre": ["laisser le narrateur dérouler les faits sans structure trop scolaire"],
    "note_professionnelle": ["style sec, quasi-notarial ou cabinet"],
    "oral_retranscrit": ["ponctuation un peu irrégulière, rythme oral"],
    "message_conflictuel": ["faire sentir un conflit ou une tension explicite"],
}
LENGTH_REQUIREMENTS = {
    "court": ["viser un cas bref et dense, sans devenir télégraphique"],
    "moyen": ["viser un niveau de détail intermédiaire, lisible d'un seul bloc"],
    "long": ["ajouter assez de matière factuelle pour un cas nettement développé"],
    "tres_long": ["viser un cas riche, détaillé et multi-couches, sans donner la solution"],
}
NOISE_REQUIREMENTS = {
    "propre": ["pas d'erreur volontaire obligatoire"],
    "legeres_fautes": ["ajouter 1 ou 2 fautes réalistes maximum"],
    "fautes_et_abreviations": ["ajouter quelques abréviations réalistes (AV, RP, M., Mme, etc.)"],
    "ambigu": ["laisser au moins un détail flou, approximatif ou contesté"],
    "tres_brouillon": ["laisser des morceaux incomplets, hésitants ou mal ponctués"],
}
NUMERIC_REQUIREMENTS = {
    "sans_montant": ["aucun chiffre n'est obligatoire"],
    "un_montant": ["inclure au moins un montant ou une valeur"],
    "plusieurs_montants": ["inclure plusieurs montants, valeurs ou proportions"],
    "montants_et_dates": ["inclure au moins un montant et une date utile, de préférence exacte"],
}
DATE_PRECISION_REQUIREMENTS = {
    "aucune": ["aucune date n'est obligatoire si elle n'apporte rien"],
    "approx": ["utiliser un repère temporel flou ou approximatif si une date apparaît"],
    "exacte": ["inclure au moins une date exacte (jour/mois/année ou format ISO)"],
}
HARD_NEGATIVE_REQUIREMENTS = {
    "pas_de_deces_clair": [
        "le texte doit ressembler à une succession mais sans décès exploitable clairement posé"
    ],
    "infos_incompletes": [
        "laisser manquer une donnée-clé (date, lien, testament, régime, composition des héritiers)"
    ],
    "faits_contradictoires": [
        "introduire une contradiction factuelle réaliste sans la résoudre"
    ],
    "hors_perimetre_mal_qualifie": [
        "faire croire à une succession alors qu'une partie du problème relève d'autre chose"
    ],
}
HARD_NEGATIVE_INTENSITY_REQUIREMENTS = {
    "soft": ["ne mettre qu'un défaut principal, le cas doit rester très crédible au premier regard"],
    "hard": ["cumuler au moins deux sources de confusion sans rendre le texte absurde"],
}


@dataclass(slots=True)
class CorpusSeed:
    case_id: str
    source_type: str
    source_name: str
    text: str


@dataclass(slots=True)
class MasterSchemaIndex:
    allowed_nodes: set[tuple[str, ...]]
    leaf_specs: dict[tuple[str, ...], dict[str, Any]]


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _normalize_text(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_key(value: str) -> str:
    normalized = unicodedata.normalize("NFD", _normalize_text(value).lower())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")


def _decode_toon_with_cli(toon_text: str) -> Any:
    # Validate TOON syntax using the official CLI in decode mode
    # and return decoded JSON payload.
    with tempfile.NamedTemporaryFile("w", suffix=".toon", encoding="utf-8", delete=False) as handle:
        handle.write(toon_text)
        temp_path = handle.name
    try:
        result = subprocess.run(
            ["npx", "-y", "@toon-format/cli", temp_path],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("validation TOON expirée") from exc
    finally:
        Path(temp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        error = (result.stderr or result.stdout or "").strip()
        first_line = error.splitlines()[0] if error else "TOON invalide"
        raise ValueError(f"target_toon invalide: {first_line}")

    decoded_raw = (result.stdout or "").strip()
    if not decoded_raw:
        raise ValueError("target_toon invalide: sortie de décodage vide")
    try:
        return json.loads(decoded_raw)
    except json.JSONDecodeError as exc:
        raise ValueError("target_toon invalide: sortie de décodage illisible") from exc


def _encode_json_to_toon(payload: dict[str, Any]) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False)
        temp_path = handle.name
    try:
        result = subprocess.run(
            ["npx", "-y", "@toon-format/cli", "--encode", temp_path],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError("encodage TOON expiré") from exc
    finally:
        Path(temp_path).unlink(missing_ok=True)

    if result.returncode != 0:
        error = (result.stderr or result.stdout or "").strip()
        first_line = error.splitlines()[0] if error else "échec encodage TOON"
        raise ValueError(f"encodage TOON invalide: {first_line}")

    toon_text = (result.stdout or "").strip()
    if not toon_text:
        raise ValueError("encodage TOON invalide: sortie vide")

    normalized, _ = _normalize_target_toon(toon_text)
    return normalized

def _clean_name(value: str) -> str:
    normalized = _normalize_key(value)
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _collect_named_values(payload: Any) -> list[str]:
    names: list[str] = []

    def visit(node: Any, parent_key: str | None = None) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                key_norm = key.lower()
                if isinstance(value, str):
                    if key_norm == "nom" or key_norm.endswith("_nom"):
                        cleaned = value.strip()
                        if cleaned:
                            names.append(cleaned)
                    elif key_norm.endswith("_noms"):
                        # Should usually be a list, but keep defensive handling.
                        cleaned = value.strip()
                        if cleaned:
                            names.append(cleaned)
                visit(value, key_norm)
            return
        if isinstance(node, list):
            if parent_key and parent_key.endswith("_noms"):
                for item in node:
                    if isinstance(item, str):
                        cleaned = item.strip()
                        if cleaned:
                            names.append(cleaned)
            for item in node:
                visit(item, parent_key)

    visit(payload, None)

    # Stable dedup while preserving insertion order.
    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = _clean_name(name)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(name)
    return deduped


def _name_appears_in_case_text(name: str, normalized_case_text: str) -> bool:
    cleaned = _clean_name(name)
    if not cleaned:
        return True
    if cleaned in normalized_case_text:
        return True
    tokens = [token for token in cleaned.split() if len(token) >= 2]
    if not tokens:
        return True
    last_token = tokens[-1]
    if len(last_token) >= 4 and last_token in normalized_case_text:
        return True
    if last_token in normalized_case_text and any(
        token in normalized_case_text for token in tokens[:-1]
    ):
        return True
    return False


def _missing_names_from_case_text(case_text: str, decoded_target: Any) -> list[str]:
    if not isinstance(decoded_target, dict):
        return []
    normalized_case_text = _normalize_key(case_text)
    names = _collect_named_values(decoded_target)
    missing: list[str] = []
    for name in names:
        if not _name_appears_in_case_text(name, normalized_case_text):
            missing.append(name)
    return missing


def _normalize_target_toon(value: Any) -> tuple[str, Any]:
    if not isinstance(value, str):
        raise ValueError("target_toon doit être une chaîne TOON")
    raw_text = value.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    toon_text = "\n".join(line.rstrip() for line in raw_text.splitlines())
    if not toon_text:
        raise ValueError("target_toon vide")

    # Explicit guard: reject JSON-looking payloads.
    raw = toon_text.strip()
    if raw.startswith("{") or raw.startswith("["):
        raise ValueError("target_toon semble être du JSON, TOON attendu")

    decoded = _decode_toon_with_cli(toon_text)
    if not isinstance(decoded, dict):
        raise ValueError("target_toon invalide: la racine doit être un objet")
    return toon_text, decoded


def _pair_training_record(case_text: str, target_toon: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": PAIR_TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": _normalize_text(case_text)},
            {
                "role": "assistant",
                "content": target_toon,
            },
        ]
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_schema_leaf(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    meta_keys = {"description", "type", "valeurs_possibles", "pickOne"}
    if not any(key in node for key in meta_keys):
        return False
    # Leaf specs in this schema only contain metadata keys.
    if any(key not in meta_keys for key in node):
        return False
    # Guard against structural nodes that use "type" as a child object key.
    if isinstance(node.get("type"), dict):
        return False
    return True


def _enum_values_from_schema_leaf(node: dict[str, Any]) -> list[str]:
    raw = node.get("valeurs_possibles")
    if not isinstance(raw, list):
        raw = node.get("pickOne")
    if not isinstance(raw, list):
        return []
    values: list[str] = []
    for item in raw:
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                values.append(stripped)
    return values


def _leaf_expected_type(node: dict[str, Any]) -> str:
    schema_type = node.get("type")
    if isinstance(schema_type, str) and schema_type in {"string", "number", "boolean"}:
        return schema_type
    if _enum_values_from_schema_leaf(node):
        return "string"
    # Default for descriptive leaves is string (names, libellés, lois, etc.).
    return "string"


def _build_master_schema_index(schema: dict[str, Any]) -> MasterSchemaIndex:
    allowed_nodes: set[tuple[str, ...]] = {()}
    leaf_specs: dict[tuple[str, ...], dict[str, Any]] = {}

    def walk(node: Any, path: tuple[str, ...]) -> None:
        allowed_nodes.add(path)
        if _is_schema_leaf(node):
            leaf_specs[path] = node if isinstance(node, dict) else {}
            return
        if isinstance(node, dict):
            for key, child in node.items():
                if isinstance(key, str):
                    walk(child, path + (key,))
            return
        if isinstance(node, list):
            allowed_nodes.add(path + ("*",))
            if node:
                walk(node[0], path + ("*",))

    walk(schema, ())
    return MasterSchemaIndex(allowed_nodes=allowed_nodes, leaf_specs=leaf_specs)


def _validate_target_payload_against_schema(
    payload: dict[str, Any],
    schema_index: MasterSchemaIndex,
) -> None:
    errors: list[str] = []

    def path_str(path: tuple[str, ...]) -> str:
        return ".".join(path) if path else "<root>"

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if path not in schema_index.allowed_nodes:
            errors.append(f"chemin non autorisé: {path_str(path)}")
            return

        if isinstance(node, dict):
            for key, child in node.items():
                if not isinstance(key, str):
                    errors.append(f"clé non texte à {path_str(path)}")
                    continue
                child_path = path + (key,)
                if child_path not in schema_index.allowed_nodes:
                    errors.append(f"clé inconnue: {path_str(child_path)}")
                    continue
                walk(child, child_path)
            return

        if isinstance(node, list):
            list_path = path + ("*",)
            if list_path not in schema_index.allowed_nodes:
                errors.append(f"liste non autorisée: {path_str(path)}")
                return
            for item in node:
                walk(item, list_path)
            return

        if path not in schema_index.leaf_specs:
            errors.append(f"valeur scalaire à un chemin non-feuille: {path_str(path)}")
            return

        spec = schema_index.leaf_specs[path]
        expected_type = _leaf_expected_type(spec)
        if expected_type == "string":
            if not isinstance(node, str):
                errors.append(f"type attendu string à {path_str(path)}")
        elif expected_type == "number":
            if not isinstance(node, (int, float)) or isinstance(node, bool):
                errors.append(f"type attendu number à {path_str(path)}")
        elif expected_type == "boolean":
            if not isinstance(node, bool):
                errors.append(f"type attendu boolean à {path_str(path)}")

        enum_values = _enum_values_from_schema_leaf(spec)
        if enum_values:
            if not isinstance(node, str) or node not in enum_values:
                errors.append(
                    f"valeur hors enum à {path_str(path)} (reçu={node!r}, attendu={enum_values})"
                )

    walk(payload, ())

    if errors:
        preview = "; ".join(errors[:3])
        if len(errors) > 3:
            preview += "; ..."
        raise ValueError(f"target généré non conforme au schema.full: {preview}")


def _validate_sparse_payload(payload: dict[str, Any]) -> None:
    errors: list[str] = []

    def path_str(path: tuple[str, ...]) -> str:
        return ".".join(path) if path else "<root>"

    def walk(node: Any, path: tuple[str, ...]) -> None:
        if node is None:
            errors.append(f"null interdit à {path_str(path)}")
            return
        if isinstance(node, dict):
            if not node:
                errors.append(f"objet vide interdit à {path_str(path)}")
                return
            for key, value in node.items():
                if not isinstance(key, str) or not key:
                    errors.append(f"clé invalide à {path_str(path)}")
                    continue
                walk(value, path + (key,))
            return
        if isinstance(node, list):
            if not node:
                errors.append(f"liste vide interdite à {path_str(path)}")
                return
            for idx, value in enumerate(node):
                walk(value, path + (f"[{idx}]",))
            return
        if isinstance(node, str):
            if not node.strip():
                errors.append(f"string vide interdite à {path_str(path)}")
            return
        if isinstance(node, (int, float, bool)):
            return
        errors.append(f"type non supporté à {path_str(path)}: {type(node).__name__}")

    walk(payload, ())
    if errors:
        preview = "; ".join(errors[:3])
        if len(errors) > 3:
            preview += "; ..."
        raise ValueError(f"target généré non sparse: {preview}")


def _parse_iso_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _years_between_dates(start: date, end: date) -> int:
    years = end.year - start.year
    if (end.month, end.day) < (start.month, start.day):
        years -= 1
    return years


def _collect_person_records(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any], str | None]]:
    records: list[tuple[str, dict[str, Any], str | None]] = []
    famille = payload.get("famille")
    if not isinstance(famille, dict):
        return records

    defunt = famille.get("defunt")
    defunt_deces = defunt.get("date_deces") if isinstance(defunt, dict) else None
    if isinstance(defunt, dict):
        records.append(("famille.defunt", defunt, defunt_deces if isinstance(defunt_deces, str) else None))

    partenaire = famille.get("partenaire")
    if isinstance(partenaire, dict):
        records.append(
            (
                "famille.partenaire",
                partenaire,
                defunt_deces if isinstance(defunt_deces, str) else None,
            )
        )

    for bloc in ("descendants", "ascendants", "collateraux"):
        root = famille.get(bloc)
        if not isinstance(root, dict):
            continue
        for group_name, values in root.items():
            if not isinstance(values, list):
                continue
            for idx, person in enumerate(values):
                if isinstance(person, dict):
                    records.append(
                        (
                            f"famille.{bloc}.{group_name}[{idx}]",
                            person,
                            defunt_deces if isinstance(defunt_deces, str) else None,
                        )
                    )

    return records


def _validate_business_coherence(
    payload: dict[str, Any],
    *,
    dimensions: dict[str, Any],
) -> None:
    errors: list[str] = []

    famille = payload.get("famille")
    defunt = famille.get("defunt") if isinstance(famille, dict) else None
    partenaire = famille.get("partenaire") if isinstance(famille, dict) else None

    defunt_name = defunt.get("nom") if isinstance(defunt, dict) else None
    statut = defunt.get("statut_matrimonial") if isinstance(defunt, dict) else None
    partner_link = None
    if isinstance(partenaire, dict):
        link = partenaire.get("lien")
        if isinstance(link, dict):
            partner_link = link.get("type")

    if not isinstance(defunt_name, str) or not defunt_name.strip():
        errors.append("défunt.nom manquant")

    if statut == "MARIE":
        if not isinstance(partenaire, dict):
            errors.append("statut MARIE sans partenaire")
        if partner_link != "CONJOINT":
            errors.append("statut MARIE incohérent avec partenaire.lien.type")
    if statut == "PACSE":
        if not isinstance(partenaire, dict):
            errors.append("statut PACSE sans partenaire")
        if partner_link != "PARTENAIRE_PACS":
            errors.append("statut PACSE incohérent avec partenaire.lien.type")
    if statut in {"CELIBATAIRE", "DIVORCE", "VEUF"} and partner_link == "CONJOINT":
        errors.append("statut sans conjoint incohérent avec partenaire CONJOINT")

    for label, person, ref_death in _collect_person_records(payload):
        birth_raw = person.get("date_naissance")
        age_raw = person.get("age_au_deces")
        minor_raw = person.get("est_mineur")
        birth = _parse_iso_date(birth_raw)
        ref = _parse_iso_date(ref_death)
        if isinstance(age_raw, (int, float)) and not isinstance(age_raw, bool):
            if age_raw < 0 or age_raw > 125:
                errors.append(f"age hors plage à {label}")
            if isinstance(minor_raw, bool):
                if minor_raw and age_raw >= 18:
                    errors.append(f"est_mineur incohérent avec âge à {label}")
                if not minor_raw and age_raw < 18:
                    errors.append(f"est_mineur incohérent avec âge à {label}")
        if birth is not None and ref is not None:
            if birth > ref:
                errors.append(f"date_naissance postérieure au décès à {label}")
            elif isinstance(age_raw, (int, float)) and not isinstance(age_raw, bool):
                computed_age = _years_between_dates(birth, ref)
                if abs(int(round(age_raw)) - computed_age) > 1:
                    errors.append(f"age/date incohérent à {label}")

    assurance_vie = payload.get("assurance_vie")
    contracts = assurance_vie.get("contrats") if isinstance(assurance_vie, dict) else None
    if isinstance(contracts, list):
        for idx, contract in enumerate(contracts):
            if not isinstance(contract, dict):
                continue
            assure_nom = contract.get("assure_nom")
            if isinstance(defunt_name, str) and isinstance(assure_nom, str) and assure_nom != defunt_name:
                errors.append(f"assurance_vie.contrats[{idx}].assure_nom != défunt.nom")
            versements = contract.get("versements")
            if isinstance(versements, list):
                for vidx, versement in enumerate(versements):
                    if not isinstance(versement, dict):
                        continue
                    age = versement.get("age_assure_au_versement")
                    apres = versement.get("apres_70_ans")
                    if isinstance(age, (int, float)) and not isinstance(age, bool) and isinstance(apres, bool):
                        if age >= 70 and not apres:
                            errors.append(
                                f"versement[{vidx}] incohérent: age >= 70 mais apres_70_ans=false"
                            )
                        if age < 70 and apres:
                            errors.append(
                                f"versement[{vidx}] incohérent: age < 70 mais apres_70_ans=true"
                            )

    liberalites = payload.get("liberalites")
    donations = liberalites.get("donations") if isinstance(liberalites, dict) else None
    if isinstance(donations, list):
        for idx, donation in enumerate(donations):
            if not isinstance(donation, dict):
                continue
            donateur = donation.get("donateur_nom")
            beneficiaire = donation.get("beneficiaire_nom")
            if isinstance(donateur, str) and isinstance(beneficiaire, str) and donateur == beneficiaire:
                errors.append(f"donation[{idx}] donateur == beneficiaire")

    patrimoine = payload.get("patrimoine")
    actifs = patrimoine.get("actifs") if isinstance(patrimoine, dict) else None
    if isinstance(actifs, list):
        for idx, actif in enumerate(actifs):
            if not isinstance(actif, dict):
                continue
            valeur = actif.get("valeur")
            if isinstance(valeur, (int, float)) and not isinstance(valeur, bool) and valeur <= 0:
                errors.append(f"actif[{idx}] valeur <= 0")
    passifs = patrimoine.get("passifs") if isinstance(patrimoine, dict) else None
    if isinstance(passifs, list):
        for idx, passif in enumerate(passifs):
            if not isinstance(passif, dict):
                continue
            valeur = passif.get("valeur")
            if isinstance(valeur, (int, float)) and not isinstance(valeur, bool) and valeur <= 0:
                errors.append(f"passif[{idx}] valeur <= 0")

    primary_topic = str(dimensions.get("primary_topic") or "")
    if primary_topic == "assurance_vie":
        if not isinstance(contracts, list) or not contracts:
            errors.append("topic assurance_vie sans contrat")
    if primary_topic == "donations_reduction":
        if not isinstance(donations, list) or not donations:
            errors.append("topic donations_reduction sans donation")
    if primary_topic == "entreprise_dutreil":
        if not isinstance(actifs, list) or not actifs:
            errors.append("topic entreprise_dutreil sans actif")
        else:
            has_company = any(
                isinstance(a, dict) and isinstance(a.get("entreprise"), dict)
                for a in actifs
            )
            if not has_company:
                errors.append("topic entreprise_dutreil sans bloc entreprise")

    if errors:
        preview = "; ".join(errors[:3])
        if len(errors) > 3:
            preview += "; ..."
        raise ValueError(f"target généré incohérent métier: {preview}")


def _path_exists_in_payload(node: Any, path: tuple[str, ...], idx: int = 0) -> bool:
    if idx >= len(path):
        return True
    token = path[idx]
    if token == "*":
        if not isinstance(node, list) or not node:
            return False
        return any(_path_exists_in_payload(item, path, idx + 1) for item in node)
    if not isinstance(node, dict):
        return False
    if token not in node:
        return False
    return _path_exists_in_payload(node[token], path, idx + 1)


def _topic_present_in_payload(payload: dict[str, Any], topic: str) -> bool:
    required = TOPIC_REQUIRED_LEAF_PATHS.get(topic, [])
    if required:
        return all(_path_exists_in_payload(payload, path) for path in required)
    prefixes = TOPIC_SCHEMA_PREFIXES.get(topic, [])
    for prefix in prefixes:
        if _path_exists_in_payload(payload, prefix):
            return True
    return False


def _validate_topic_alignment(
    payload: dict[str, Any],
    *,
    primary_topic: str,
    secondary_topic: str | None,
) -> None:
    errors: list[str] = []
    if not _topic_present_in_payload(payload, primary_topic):
        errors.append(f"primary_topic={primary_topic} absent du TOON")
    if isinstance(secondary_topic, str) and secondary_topic:
        if not _topic_present_in_payload(payload, secondary_topic):
            errors.append(f"secondary_topic={secondary_topic} absent du TOON")
    if errors:
        raise ValueError("alignment topic/TOON invalide: " + "; ".join(errors))


def _load_master_schema(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"master schema introuvable: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("master schema invalide: racine non objet")
    return payload


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9àâçéèêëîïôûùüÿñæœ]+", _normalize_key(text))
        if len(token) > 1
    }


def _jaccard_similarity(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    if union == 0:
        return 0.0
    return intersection / union


def _load_seed_cases(path: Path) -> list[CorpusSeed]:
    seeds: list[CorpusSeed] = []
    for row in _load_jsonl(path):
        text = row.get("text")
        if not isinstance(text, str):
            continue
        seeds.append(
            CorpusSeed(
                case_id=str(row.get("case_id") or f"seed_{len(seeds) + 1:04d}"),
                source_type=str(row.get("source_type") or "unknown"),
                source_name=str(row.get("source_name") or ""),
                text=_normalize_text(text),
            )
        )
    return seeds


def _pick_underrepresented(
    targets: dict[str, float],
    counts: dict[str, int],
    rng: random.Random,
    exclude: set[str] | None = None,
) -> str:
    blocked = exclude or set()
    best_key = ""
    best_score: tuple[float, int, float] | None = None
    for key, share in targets.items():
        if key in blocked:
            continue
        current = counts.get(key, 0)
        ratio = current / share
        score = (ratio, current, rng.random())
        if best_score is None or score < best_score:
            best_key = key
            best_score = score
    if not best_key:
        raise RuntimeError("No available option for target selection.")
    return best_key


class InstructionServerApp:
    def __init__(
        self,
        *,
        state_dir: Path,
        corpus_file: Path,
        master_schema_file: Path,
        target_total_cases: int,
        generation_target: int | None,
        seed: int,
    ) -> None:
        self.lock = threading.Lock()
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.instructions_dir = self.state_dir / "instructions"
        self.submissions_dir = self.state_dir / "submissions"
        self.instructions_dir.mkdir(parents=True, exist_ok=True)
        self.submissions_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = self.state_dir / CONFIG_FILENAME
        self.issued_path = self.state_dir / ISSUED_FILENAME
        self.submitted_path = self.state_dir / SUBMITTED_FILENAME
        self.summary_json_path = self.state_dir / SUMMARY_JSON_FILENAME
        self.summary_md_path = self.state_dir / SUMMARY_MD_FILENAME
        self.generated_train_path = self.state_dir / GENERATED_TRAIN_FILENAME
        self.full_train_path = self.state_dir / FULL_TRAIN_FILENAME

        self.master_schema_file = master_schema_file
        self.master_schema = _load_master_schema(master_schema_file)
        self.master_schema_index = _build_master_schema_index(self.master_schema)
        self.faker = Faker("fr_FR") if Faker is not None else None

        self.seed_cases = _load_seed_cases(corpus_file)
        self.config = self._load_or_create_config(
            target_total_cases=target_total_cases,
            generation_target=generation_target,
            seed=seed,
            corpus_file=corpus_file,
        )
        if str(corpus_file) != str(self.config["corpus_file"]):
            self.seed_cases = _load_seed_cases(Path(self.config["corpus_file"]))
        self.issued = _load_jsonl(self.issued_path)
        self.submitted = _load_jsonl(self.submitted_path)
        self._sanitize_legacy_state()
        self._refresh_training_exports()
        self._refresh_summary()

    def _sanitize_legacy_state(self) -> None:
        issued_changed = False
        sanitized_issued: list[dict[str, Any]] = []
        for row in self.issued:
            if not isinstance(row, dict):
                issued_changed = True
                continue
            updated = dict(row)

            response_format = updated.get("response_format")
            if isinstance(response_format, dict):
                rf = dict(response_format)
                required_keys = rf.get("required_keys")
                if isinstance(required_keys, list) and "target_json" in required_keys:
                    rf["required_keys"] = [
                        "target_toon" if key == "target_json" else key
                        for key in required_keys
                    ]
                    issued_changed = True
                if "target_json_rule" in rf:
                    toon_rule = rf.pop("target_json_rule")
                    if "target_toon_rule" not in rf and isinstance(toon_rule, str):
                        rf["target_toon_rule"] = toon_rule.replace("JSON", "TOON")
                    issued_changed = True
                updated["response_format"] = rf

            submission_contract = updated.get("submission_contract")
            if isinstance(submission_contract, dict):
                sc = dict(submission_contract)
                required_fields = sc.get("required_fields")
                if isinstance(required_fields, list) and "target_json" in required_fields:
                    sc["required_fields"] = [
                        "target_toon" if key == "target_json" else key
                        for key in required_fields
                    ]
                    issued_changed = True
                if "target_json_rule" in sc:
                    toon_rule = sc.pop("target_json_rule")
                    if "target_toon_rule" not in sc and isinstance(toon_rule, str):
                        sc["target_toon_rule"] = toon_rule.replace("JSON", "TOON")
                    issued_changed = True
                updated["submission_contract"] = sc

            prompt = updated.get("prompt")
            if isinstance(prompt, str) and "target_json" in prompt:
                updated["prompt"] = prompt.replace("target_json", "target_toon").replace(
                    "JSON cible rempli",
                    "TOON cible valide",
                )
                issued_changed = True

            sanitized_issued.append(updated)
        self.issued = sanitized_issued
        if issued_changed:
            _rewrite_jsonl(self.issued_path, self.issued)

        submitted_changed = False
        sanitized_submitted: list[dict[str, Any]] = []
        for row in self.submitted:
            if not isinstance(row, dict):
                submitted_changed = True
                continue
            updated = dict(row)
            if "target_json" in updated:
                updated.pop("target_json", None)
                submitted_changed = True
            target_toon = updated.get("target_toon")
            if not isinstance(target_toon, str):
                submitted_changed = True
                continue
            cleaned = "\n".join(
                line.rstrip()
                for line in target_toon.replace("\r\n", "\n").replace("\r", "\n").strip("\n").splitlines()
            )
            if not cleaned:
                submitted_changed = True
                continue
            if cleaned != target_toon:
                updated["target_toon"] = cleaned
                submitted_changed = True
            sanitized_submitted.append(updated)
        self.submitted = sanitized_submitted
        if submitted_changed:
            _rewrite_jsonl(self.submitted_path, self.submitted)

        legacy_instruction_file = self.state_dir / "_last_instruction.json"
        if legacy_instruction_file.exists():
            try:
                payload = json.loads(legacy_instruction_file.read_text(encoding="utf-8"))
                if isinstance(payload, dict) and "target_json" in json.dumps(payload, ensure_ascii=False):
                    legacy_instruction_file.unlink(missing_ok=True)
            except Exception:
                # Keep startup robust even if this legacy file is malformed.
                legacy_instruction_file.unlink(missing_ok=True)

    def _collect_mandatory_elements(self, dimensions: dict[str, str | None]) -> list[str]:
        primary_topic = str(dimensions["primary_topic"])
        secondary_topic = dimensions.get("secondary_topic")
        hard_negative_mode = dimensions.get("hard_negative_mode")
        hard_negative_intensity = dimensions.get("hard_negative_intensity")

        mandatory_elements: list[str] = []
        mandatory_elements.extend(TOPIC_TEMPLATES[primary_topic]["elements"])
        if isinstance(secondary_topic, str) and secondary_topic:
            mandatory_elements.extend(TOPIC_TEMPLATES[secondary_topic]["elements"])
        mandatory_elements.extend(FORMAT_REQUIREMENTS[str(dimensions["format"])])
        mandatory_elements.extend(LENGTH_REQUIREMENTS[str(dimensions["length_band"])])
        mandatory_elements.extend(NOISE_REQUIREMENTS[str(dimensions["noise"])])
        mandatory_elements.extend(NUMERIC_REQUIREMENTS[str(dimensions["numeric_density"])])
        mandatory_elements.extend(DATE_PRECISION_REQUIREMENTS[str(dimensions["date_precision"])])
        if isinstance(hard_negative_mode, str) and hard_negative_mode:
            mandatory_elements.extend(HARD_NEGATIVE_REQUIREMENTS[hard_negative_mode])
        if isinstance(hard_negative_intensity, str) and hard_negative_intensity:
            mandatory_elements.extend(HARD_NEGATIVE_INTENSITY_REQUIREMENTS[hard_negative_intensity])

        deduped_elements: list[str] = []
        seen = set()
        for item in mandatory_elements:
            if item in seen:
                continue
            seen.add(item)
            deduped_elements.append(item)
        return deduped_elements

    def _collect_must_avoid(self, dimensions: dict[str, str | None]) -> list[str]:
        items = list(COMMON_MUST_AVOID)
        if str(dimensions["complexity"]) == "hard_negative":
            items.append("Ne pas signaler explicitement qu'il s'agit d'un hard negative ou d'un piège.")
        return items

    def _build_style_brief(self, dimensions: dict[str, str | None]) -> str:
        persona = str(dimensions["persona"])
        voice = str(dimensions["voice"])
        format_name = str(dimensions["format"])
        primary_topic = str(dimensions["primary_topic"])
        secondary_topic = dimensions.get("secondary_topic")
        secondary_label = (
            TOPIC_TEMPLATES[secondary_topic]["label"]
            if isinstance(secondary_topic, str) and secondary_topic
            else None
        )

        parts = [
            f"Le cas doit être raconté comme si {PERSONA_LABELS[persona]} s'exprimait.",
            f"La tournure attendue est {VOICE_LABELS[voice]}.",
            f"La forme doit ressembler à {FORMAT_LABELS[format_name]}.",
            f"Le coeur juridique doit tourner autour de {TOPIC_TEMPLATES[primary_topic]['label']}.",
        ]
        if secondary_label:
            parts.append(f"Une seconde couche doit faire intervenir {secondary_label}.")
        return " ".join(parts)

    def _build_dimension_guide(self, dimensions: dict[str, str | None]) -> dict[str, dict[str, Any]]:
        topic_allowed = {
            key: value["label"]
            for key, value in TOPIC_TEMPLATES.items()
        }

        def _axis_payload(
            *,
            key: str,
            selected: str | None,
            selected_label: str | None,
            selected_effect: str,
            allowed_values: dict[str, str],
            only_when: str | None = None,
        ) -> dict[str, Any]:
            return {
                "selected_value": selected,
                "selected_label": selected_label,
                "purpose": DIMENSION_PURPOSES[key],
                "selected_effect": selected_effect,
                "allowed_values": allowed_values,
                "only_active_when": only_when,
            }

        primary_topic = str(dimensions["primary_topic"])
        secondary_topic = dimensions.get("secondary_topic")
        hard_negative_mode = dimensions.get("hard_negative_mode")
        hard_negative_intensity = dimensions.get("hard_negative_intensity")

        secondary_selected_effect = (
            "Aucune couche secondaire n'est imposée sur cette consigne."
            if not isinstance(secondary_topic, str) or not secondary_topic
            else (
                "Cette couche ajoute une contrainte supplémentaire : "
                + " ; ".join(TOPIC_TEMPLATES[secondary_topic]["elements"])
            )
        )
        hard_negative_mode_effect = (
            "Inactif ici, car la complexité tirée n'est pas un hard negative."
            if not isinstance(hard_negative_mode, str) or not hard_negative_mode
            else HARD_NEGATIVE_MODE_DETAILS[hard_negative_mode]
        )
        hard_negative_intensity_effect = (
            "Inactif ici, car la complexité tirée n'est pas un hard negative."
            if not isinstance(hard_negative_intensity, str) or not hard_negative_intensity
            else HARD_NEGATIVE_INTENSITY_DETAILS[hard_negative_intensity]
        )

        return {
            "persona": _axis_payload(
                key="persona",
                selected=str(dimensions["persona"]),
                selected_label=PERSONA_LABELS[str(dimensions["persona"])],
                selected_effect=PERSONA_DETAILS[str(dimensions["persona"])],
                allowed_values=PERSONA_LABELS,
            ),
            "voice": _axis_payload(
                key="voice",
                selected=str(dimensions["voice"]),
                selected_label=VOICE_LABELS[str(dimensions["voice"])],
                selected_effect=VOICE_DETAILS[str(dimensions["voice"])],
                allowed_values=VOICE_LABELS,
            ),
            "format": _axis_payload(
                key="format",
                selected=str(dimensions["format"]),
                selected_label=FORMAT_LABELS[str(dimensions["format"])],
                selected_effect=FORMAT_DETAILS[str(dimensions["format"])],
                allowed_values=FORMAT_LABELS,
            ),
            "length_band": _axis_payload(
                key="length_band",
                selected=str(dimensions["length_band"]),
                selected_label=LENGTH_LABELS[str(dimensions["length_band"])],
                selected_effect=LENGTH_DETAILS[str(dimensions["length_band"])],
                allowed_values=LENGTH_LABELS,
            ),
            "noise": _axis_payload(
                key="noise",
                selected=str(dimensions["noise"]),
                selected_label=NOISE_LABELS[str(dimensions["noise"])],
                selected_effect=NOISE_DETAILS[str(dimensions["noise"])],
                allowed_values=NOISE_LABELS,
            ),
            "numeric_density": _axis_payload(
                key="numeric_density",
                selected=str(dimensions["numeric_density"]),
                selected_label=NUMERIC_LABELS[str(dimensions["numeric_density"])],
                selected_effect=NUMERIC_DETAILS[str(dimensions["numeric_density"])],
                allowed_values=NUMERIC_LABELS,
            ),
            "date_precision": _axis_payload(
                key="date_precision",
                selected=str(dimensions["date_precision"]),
                selected_label=DATE_PRECISION_LABELS[str(dimensions["date_precision"])],
                selected_effect=DATE_PRECISION_DETAILS[str(dimensions["date_precision"])],
                allowed_values=DATE_PRECISION_LABELS,
            ),
            "complexity": _axis_payload(
                key="complexity",
                selected=str(dimensions["complexity"]),
                selected_label=COMPLEXITY_LABELS[str(dimensions["complexity"])],
                selected_effect=COMPLEXITY_DETAILS[str(dimensions["complexity"])],
                allowed_values=COMPLEXITY_LABELS,
            ),
            "primary_topic": _axis_payload(
                key="primary_topic",
                selected=primary_topic,
                selected_label=TOPIC_TEMPLATES[primary_topic]["label"],
                selected_effect="Cette matière doit structurer le cas. Exigences métier : "
                + " ; ".join(TOPIC_TEMPLATES[primary_topic]["elements"]),
                allowed_values=topic_allowed,
            ),
            "secondary_topic": _axis_payload(
                key="secondary_topic",
                selected=secondary_topic if isinstance(secondary_topic, str) and secondary_topic else None,
                selected_label=(
                    TOPIC_TEMPLATES[secondary_topic]["label"]
                    if isinstance(secondary_topic, str) and secondary_topic
                    else None
                ),
                selected_effect=secondary_selected_effect,
                allowed_values=topic_allowed,
            ),
            "hard_negative_mode": _axis_payload(
                key="hard_negative_mode",
                selected=hard_negative_mode if isinstance(hard_negative_mode, str) and hard_negative_mode else None,
                selected_label=(
                    HARD_NEGATIVE_LABELS[hard_negative_mode]
                    if isinstance(hard_negative_mode, str) and hard_negative_mode
                    else None
                ),
                selected_effect=hard_negative_mode_effect,
                allowed_values=HARD_NEGATIVE_LABELS,
                only_when="complexity == hard_negative",
            ),
            "hard_negative_intensity": _axis_payload(
                key="hard_negative_intensity",
                selected=(
                    hard_negative_intensity
                    if isinstance(hard_negative_intensity, str) and hard_negative_intensity
                    else None
                ),
                selected_label=(
                    HARD_NEGATIVE_INTENSITY_LABELS[hard_negative_intensity]
                    if isinstance(hard_negative_intensity, str) and hard_negative_intensity
                    else None
                ),
                selected_effect=hard_negative_intensity_effect,
                allowed_values=HARD_NEGATIVE_INTENSITY_LABELS,
                only_when="complexity == hard_negative",
            ),
        }

    def _load_or_create_config(
        self,
        *,
        target_total_cases: int,
        generation_target: int | None,
        seed: int,
        corpus_file: Path,
    ) -> dict[str, Any]:
        resolved_generation_target = (
            int(generation_target)
            if generation_target is not None
            else max(int(target_total_cases) - len(self.seed_cases), 0)
        )

        if self.config_path.exists():
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload.pop("campaign_size", None)
                payload["target_total_cases"] = int(target_total_cases)
                payload["generation_target"] = resolved_generation_target
                payload["seed"] = int(seed)
                payload["corpus_file"] = str(corpus_file)
                if "created_at" not in payload:
                    payload["created_at"] = _utc_now()
                self.config_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return payload

        payload = {
            "target_total_cases": int(target_total_cases),
            "generation_target": resolved_generation_target,
            "seed": int(seed),
            "corpus_file": str(corpus_file),
            "created_at": _utc_now(),
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "target_total_cases": self.config["target_total_cases"],
            "generation_target": self.config["generation_target"],
            "seed_cases": len(self.seed_cases),
            "issued": len(self.issued),
            "submitted": len(self.submitted),
            "training_cases_current": len(self.submitted),
        }

    def dashboard(self) -> dict[str, Any]:
        if self.summary_json_path.exists():
            return json.loads(self.summary_json_path.read_text(encoding="utf-8"))
        return self._coverage_snapshot()

    def next_instruction(self, payload: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(payload.get("agent_id") or "").strip() or None
        force_topic = str(payload.get("topic") or "").strip() or None

        with self.lock:
            generation_target = int(self.config.get("generation_target") or 0)
            if generation_target and len(self.submitted) >= generation_target:
                return {
                    "done": True,
                    "message": "generation_target reached",
                    "coverage": self._coverage_snapshot(),
                }

            instruction = self._build_instruction(agent_id=agent_id, force_topic=force_topic)
            sequence = len(self.issued) + 1
            target_payload: dict[str, Any] | None = None
            last_error: Exception | None = None
            for attempt in range(1, 51):
                rng = random.Random(int(self.config["seed"]) * 1000 + sequence * 100 + attempt)
                try:
                    candidate = self._build_target_payload_for_instruction(instruction, rng)
                    _validate_sparse_payload(candidate)
                    _validate_business_coherence(candidate, dimensions=instruction.get("dimensions", {}))
                    _validate_target_payload_against_schema(candidate, self.master_schema_index)
                    dims = instruction.get("dimensions", {})
                    if isinstance(dims, dict):
                        _validate_topic_alignment(
                            candidate,
                            primary_topic=str(dims.get("primary_topic") or "ordre_heritiers"),
                            secondary_topic=(
                                str(dims.get("secondary_topic"))
                                if isinstance(dims.get("secondary_topic"), str) and dims.get("secondary_topic")
                                else None
                            ),
                        )
                    target_payload = candidate
                    break
                except Exception as exc:
                    last_error = exc
            if target_payload is None:
                message = str(last_error) if last_error else "unknown generation error"
                raise ValueError(f"échec génération target schema-driven: {message}")
            instruction["server_target_toon"] = _encode_json_to_toon(target_payload)
            self.issued.append(instruction)
            _append_jsonl(self.issued_path, instruction)
            self._write_instruction_file(instruction)
            self._refresh_summary()
            server_target_toon = str(instruction.get("server_target_toon") or "").strip()
            public_instruction = {
                "instruction_id": instruction.get("instruction_id"),
                "target_toon": server_target_toon,
                "prompt": self._augment_prompt_with_target_toon(
                    str(instruction.get("prompt") or ""),
                    server_target_toon,
                ),
            }
            return {
                "instruction": public_instruction,
                "coverage": self._coverage_snapshot(),
            }

    def _synth_name(self, rng: random.Random, used: set[str]) -> str:
        if self.faker is not None:
            for _ in range(50):
                candidate = str(self.faker.name()).strip()
                if candidate and candidate not in used:
                    used.add(candidate)
                    return candidate
        for _ in range(200):
            candidate = f"{rng.choice(SYNTH_FIRST_NAMES)} {rng.choice(SYNTH_LAST_NAMES)}"
            if candidate not in used:
                used.add(candidate)
                return candidate
        fallback = f"Personne {len(used) + 1}"
        used.add(fallback)
        return fallback

    def _random_iso_date(self, rng: random.Random, year_min: int = 2019, year_max: int = 2026) -> str:
        year = rng.randint(year_min, year_max)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        return f"{year:04d}-{month:02d}-{day:02d}"

    def _topic_prefixes_for_dimensions(self, dimensions: dict[str, Any]) -> list[tuple[str, ...]]:
        prefixes: list[tuple[str, ...]] = [("famille", "defunt")]
        primary_topic = str(dimensions.get("primary_topic") or "ordre_heritiers")
        prefixes.extend(TOPIC_SCHEMA_PREFIXES.get(primary_topic, []))
        secondary_topic = dimensions.get("secondary_topic")
        if isinstance(secondary_topic, str) and secondary_topic:
            prefixes.extend(TOPIC_SCHEMA_PREFIXES.get(secondary_topic, []))
        # Keep optional cross-topic context for diversity.
        if str(dimensions.get("complexity") or "") in {"complexe", "hard_negative"}:
            prefixes.extend([("contexte", "procedure"), ("operations_de_partage",)])
        deduped: list[tuple[str, ...]] = []
        seen: set[tuple[str, ...]] = set()
        for prefix in prefixes:
            if prefix in seen:
                continue
            seen.add(prefix)
            deduped.append(prefix)
        return deduped

    def _required_leaf_paths_for_dimensions(self, dimensions: dict[str, Any]) -> set[tuple[str, ...]]:
        paths: set[tuple[str, ...]] = set()
        primary_topic = str(dimensions.get("primary_topic") or "ordre_heritiers")
        for path in TOPIC_REQUIRED_LEAF_PATHS.get(primary_topic, []):
            paths.add(path)
        secondary_topic = dimensions.get("secondary_topic")
        if isinstance(secondary_topic, str) and secondary_topic:
            for path in TOPIC_REQUIRED_LEAF_PATHS.get(secondary_topic, []):
                paths.add(path)
        return paths

    def _path_matches_prefix(self, path: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
        if len(path) < len(prefix):
            return False
        return path[: len(prefix)] == prefix

    def _set_path_value(self, payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
        node: Any = payload
        for idx, token in enumerate(path):
            last = idx == len(path) - 1
            next_token = path[idx + 1] if not last else None

            if token == "*":
                if not isinstance(node, list):
                    raise ValueError("assignation invalide sur chemin liste")
                if not node:
                    node.append(value if last else ([] if next_token == "*" else {}))
                if last:
                    node[0] = value
                    return
                child = node[0]
                if next_token == "*" and not isinstance(child, list):
                    node[0] = []
                elif next_token != "*" and not isinstance(child, dict):
                    node[0] = {}
                node = node[0]
                continue

            if not isinstance(node, dict):
                raise ValueError("assignation invalide sur chemin objet")
            if last:
                node[token] = value
                return
            if token not in node or node[token] is None:
                node[token] = [] if next_token == "*" else {}
            elif next_token == "*" and not isinstance(node[token], list):
                node[token] = []
            elif next_token != "*" and not isinstance(node[token], dict):
                node[token] = {}
            node = node[token]

    def _leaf_key(self, path: tuple[str, ...]) -> str:
        if path and path[-1] == "*" and len(path) >= 2:
            return path[-2]
        return path[-1] if path else ""

    def _path_contains(self, path: tuple[str, ...], token: str) -> bool:
        return token in path

    def _known_name_for_path(
        self,
        path: tuple[str, ...],
        *,
        rng: random.Random,
        context: dict[str, Any],
    ) -> str:
        defunt_name = str(context["defunt_name"])
        partner_name = str(context["partner_name"])
        child_names = list(context["child_names"])
        used_names = context["used_names"]
        if self._path_contains(path, "defunt"):
            return defunt_name
        if self._path_contains(path, "partenaire"):
            return partner_name
        if self._path_contains(path, "enfants"):
            return child_names[0]
        if self._path_contains(path, "petits_enfants"):
            return child_names[1]
        if self._path_contains(path, "beneficiaires") or self._path_contains(path, "beneficiaire"):
            pool = [partner_name, child_names[0], child_names[1], defunt_name]
            return rng.choice(pool)
        return self._synth_name(rng, used_names)

    def _generate_leaf_value(
        self,
        path: tuple[str, ...],
        spec: dict[str, Any],
        *,
        rng: random.Random,
        context: dict[str, Any],
    ) -> Any:
        key = self._leaf_key(path)
        enum_values = _enum_values_from_schema_leaf(spec)
        if enum_values:
            if key == "statut_matrimonial":
                statut = str(context["statut_matrimonial"])
                if statut in enum_values:
                    return statut
            if key == "type" and self._path_contains(path, "lien"):
                statut = str(context["statut_matrimonial"])
                if statut == "MARIE" and "CONJOINT" in enum_values:
                    return "CONJOINT"
                if statut == "PACSE" and "PARTENAIRE_PACS" in enum_values:
                    return "PARTENAIRE_PACS"
                if "CONCUBIN" in enum_values:
                    return "CONCUBIN"
            return rng.choice(enum_values)

        expected_type = _leaf_expected_type(spec)
        if expected_type == "boolean":
            if key == "existe":
                return True if rng.random() < 0.78 else False
            return bool(rng.random() < 0.55)

        if expected_type == "number":
            key_norm = key.lower()
            path_norm = "/".join(path).lower()
            if "age" in key_norm:
                if self._path_contains(path, "defunt"):
                    return rng.randint(55, 94)
                return rng.randint(18, 92)
            if "esperance_de_vie" in key_norm:
                return rng.randint(5, 40)
            if "quote" in key_norm or "quotite" in key_norm or "part" in key_norm:
                return round(rng.uniform(0.1, 1.0), 2)
            if "taux" in key_norm or "decote" in key_norm:
                return round(rng.uniform(0.01, 0.15), 2)
            if "duree" in key_norm or "anciennete" in key_norm:
                return rng.randint(1, 25)
            # Many duration blocks are `{ valeur, unite }` where the leaf key is just `valeur`.
            if key_norm == "valeur" and ("duree" in path_norm or "anciennete" in path_norm or "soins" in path_norm):
                return rng.randint(1, 36)
            if "mois" in key_norm:
                return rng.randint(1, 48)
            if "patrimoine_" in key_norm or "patrimoine" in key_norm:
                return int(rng.randint(50_000, 5_000_000))
            if "montant_mensuel" in key_norm and "indemnite_occupation" in path_norm:
                return int(rng.randint(200, 5_000))
            if "revenus_mensuels" in key_norm or "charges_mensuelles" in key_norm:
                return int(rng.randint(500, 15_000))
            if "loyers_encaisses" in key_norm or "charges_reglees" in key_norm:
                return int(rng.randint(0, 250_000))
            if "valeurs" in path_norm:
                return int(rng.randint(1_000, 900_000))
            if (
                "valeur" in key_norm
                or "montant" in key_norm
                or "capital" in key_norm
                or "prix" in key_norm
                or "cout" in key_norm
                or "revenus" in key_norm
                or "charges" in key_norm
            ):
                return int(rng.randint(1_000, 900_000))
            return int(rng.randint(1, 1000))

        # string
        key_norm = key.lower()
        if key_norm == "nom" or key_norm.endswith("_nom") or key_norm.endswith("_noms"):
            return self._known_name_for_path(path, rng=rng, context=context)
        if "date" in key_norm:
            return self._random_iso_date(rng, 2005, 2026)
        if key_norm in {"debut", "fin"} and self._path_contains(path, "periode"):
            return self._random_iso_date(rng, 2005, 2026)
        if "residence_fiscale" in key_norm:
            return "France"
        if "residence_habituelle" in key_norm:
            return rng.choice(["France", "Belgique", "Espagne", "Suisse"])
        if "nationalite" in key_norm:
            return rng.choice(["Française", "Belge", "Espagnole", "Suisse"])
        if "loi_designee" in key_norm or "loi_applicable" in key_norm:
            return "Loi française"
        if "libelle" in key_norm or "description" in key_norm:
            if self._path_contains(path, "actifs"):
                return rng.choice(
                    [
                        f"Maison à {rng.choice(SYNTH_CITIES)}",
                        f"Appartement à {rng.choice(SYNTH_CITIES)}",
                        f"Terrain à {rng.choice(SYNTH_CITIES)}",
                        f"Résidence secondaire à {rng.choice(SYNTH_CITIES)}",
                        f"Compte bancaire (banque {rng.choice(['BNP', 'SG', 'CA', 'BP'])})",
                        f"Parts {rng.choice(SYNTH_COMPANIES)}",
                    ]
                )
            if self._path_contains(path, "passifs"):
                return rng.choice(["Emprunt bancaire", "Impôt", "Facture prestataire"])
            if self._path_contains(path, "contrats"):
                return f"Contrat {rng.choice(SYNTH_INSURERS)}"
            if "contrat_libelle" in key_norm:
                return f"Contrat {rng.choice(SYNTH_INSURERS)}"
            # Generic fallback for any other `libelle`/`description` leaf.
            return rng.choice(
                [
                    f"Maison à {rng.choice(SYNTH_CITIES)}",
                    f"Appartement à {rng.choice(SYNTH_CITIES)}",
                    f"Bien à {rng.choice(SYNTH_CITIES)}",
                    f"Parts {rng.choice(SYNTH_COMPANIES)}",
                ]
            )
        if "localisation" in key_norm:
            return rng.choice(SYNTH_CITIES)
        if "creancier_nom" in key_norm:
            return rng.choice(["Trésor Public", "Banque Populaire", "URSSAF", "EDF"])
        # Last resort: produce a concrete (but not too specific) string rather than a placeholder.
        return rng.choice(SYNTH_CITIES)

    def _repair_business_integrity(
        self,
        payload: dict[str, Any],
        *,
        dimensions: dict[str, Any],
        context: dict[str, Any],
        rng: random.Random,
    ) -> None:
        def _int_between(value: Any, *, default: int, min_value: int, max_value: int) -> int:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                coerced = int(round(value))
            else:
                coerced = default
            return max(min_value, min(max_value, coerced))

        def _birth_from_age(ref_date: date, age: int) -> date:
            year = max(1900, ref_date.year - age)
            month = ref_date.month
            day = min(ref_date.day, 28)
            return date(year, month, day)

        def _harmonize_person(
            person: dict[str, Any],
            *,
            ref_date: date,
            default_age: int,
            min_age: int,
            max_age: int,
            can_be_minor: bool = False,
        ) -> None:
            age = _int_between(person.get("age_au_deces"), default=default_age, min_value=min_age, max_value=max_age)
            birth = _birth_from_age(ref_date, age)
            person["age_au_deces"] = age
            person["date_naissance"] = birth.isoformat()
            if "est_mineur" in person:
                person["est_mineur"] = bool(can_be_minor and age < 18)

            option = person.get("option_successorale")
            est_decede = person.get("est_decede")
            if isinstance(est_decede, bool):
                if est_decede and option in {None, "ACCEPTE", "RENONCE", "CANTONNE"}:
                    person["option_successorale"] = "PREDECEDE"
                if not est_decede and option == "PREDECEDE":
                    person["option_successorale"] = "ACCEPTE"

        famille = payload.setdefault("famille", {})
        if not isinstance(famille, dict):
            return
        defunt = famille.setdefault("defunt", {})
        if not isinstance(defunt, dict):
            return

        statut = str(defunt.get("statut_matrimonial") or context.get("statut_matrimonial") or "CELIBATAIRE")
        defunt["nom"] = context["defunt_name"]
        defunt["statut_matrimonial"] = statut
        raw_death = _parse_iso_date(defunt.get("date_deces"))
        if raw_death is None:
            raw_death = _parse_iso_date(self._random_iso_date(rng, 2023, 2026))
        assert raw_death is not None
        defunt["date_deces"] = raw_death.isoformat()
        _harmonize_person(defunt, ref_date=raw_death, default_age=rng.randint(62, 90), min_age=35, max_age=105)
        defunt["est_handicape"] = bool(defunt.get("est_handicape", False))

        # Regime matrimonial coherence: only keep it when the case actually presents a marriage context.
        regime = defunt.get("regime_matrimonial")
        if isinstance(regime, dict):
            if statut in {"CELIBATAIRE", "PACSE", "DIVORCE"}:
                defunt.pop("regime_matrimonial", None)
                regime = None
            else:
                # If participation subobject is present, force the regime type.
                if "participation" in regime:
                    regime["type"] = "PARTICIPATION_AUX_ACQUETS"
                # If clauses imply a specific regime, make it explicit.
                regime_type = regime.get("type")
                if not isinstance(regime_type, str) or not regime_type:
                    if bool(regime.get("clause_attribution_integrale")):
                        regime["type"] = "COMMUNAUTE_UNIVERSELLE"
                    else:
                        regime["type"] = rng.choice(
                            [
                                "COMMUNAUTE_REDUITE_AUX_ACQUETS",
                                "SEPARATION_DE_BIENS",
                                "COMMUNAUTE_UNIVERSELLE",
                                "PARTICIPATION_AUX_ACQUETS",
                            ]
                        )

        partenaire = famille.get("partenaire")
        if statut in {"MARIE", "PACSE"}:
            if not isinstance(partenaire, dict):
                partenaire = {}
                famille["partenaire"] = partenaire
            partenaire["nom"] = context["partner_name"]
            lien = partenaire.get("lien")
            if not isinstance(lien, dict):
                lien = {}
                partenaire["lien"] = lien
            lien["type"] = "CONJOINT" if statut == "MARIE" else "PARTENAIRE_PACS"
            _harmonize_person(
                partenaire,
                ref_date=raw_death,
                default_age=_int_between(defunt.get("age_au_deces"), default=75, min_value=40, max_value=105) - 4,
                min_age=18,
                max_age=105,
            )
        else:
            if isinstance(partenaire, dict):
                partenaire["nom"] = context["partner_name"]
                lien = partenaire.get("lien")
                if isinstance(lien, dict):
                    if lien.get("type") in {"CONJOINT", "PARTENAIRE_PACS"}:
                        lien["type"] = "CONCUBIN"
                _harmonize_person(
                    partenaire,
                    ref_date=raw_death,
                    default_age=60,
                    min_age=18,
                    max_age=100,
                )

        primary_topic = str(dimensions.get("primary_topic") or "")
        descendants = famille.get("descendants")
        if not isinstance(descendants, dict):
            descendants = None
        needs_children = primary_topic in {
            "ordre_heritiers",
            "famille_recomposee",
            "donations_reduction",
            "testament_legs",
        }
        if needs_children and descendants is None:
            descendants = {}
            famille["descendants"] = descendants
        if isinstance(descendants, dict):
            children = descendants.get("enfants")
            if needs_children and (not isinstance(children, list) or not children):
                children = [{"nom": context["child_names"][0]}]
                descendants["enfants"] = children
            if isinstance(children, list):
                max_child_age = max(
                    1,
                    min(75, _int_between(defunt.get("age_au_deces"), default=75, min_value=35, max_value=105) - 14),
                )
                for idx, child in enumerate(children):
                    if not isinstance(child, dict):
                        child = {}
                        children[idx] = child
                    child["nom"] = context["child_names"][idx % len(context["child_names"])]
                    default_age = rng.randint(2, max(3, max_child_age))
                    _harmonize_person(
                        child,
                        ref_date=raw_death,
                        default_age=default_age,
                        min_age=0,
                        max_age=max_child_age,
                        can_be_minor=True,
                    )
                    child.setdefault("est_decede", False)
                    if primary_topic == "famille_recomposee":
                        child["est_d_une_precedente_union"] = idx == 0

            petits_enfants = descendants.get("petits_enfants")
            if isinstance(petits_enfants, list):
                for idx, grand_child in enumerate(petits_enfants):
                    if not isinstance(grand_child, dict):
                        grand_child = {}
                        petits_enfants[idx] = grand_child
                    default_age = rng.randint(0, 35)
                    _harmonize_person(
                        grand_child,
                        ref_date=raw_death,
                        default_age=default_age,
                        min_age=0,
                        max_age=55,
                        can_be_minor=True,
                    )
                    grand_child.setdefault("nom", self._synth_name(rng, context["used_names"]))
                    parent_name = context["child_names"][0]
                    grand_child.setdefault("parent_nom", parent_name)
            if not descendants:
                famille.pop("descendants", None)

        # Safety pass: ensure every person block stays date/age coherent.
        for label, person, _ in _collect_person_records(payload):
            if label == "famille.defunt":
                continue
            if label.startswith("famille.descendants"):
                _harmonize_person(
                    person,
                    ref_date=raw_death,
                    default_age=_int_between(person.get("age_au_deces"), default=25, min_value=0, max_value=75),
                    min_age=0,
                    max_age=75,
                    can_be_minor=True,
                )
                continue
            if label.startswith("famille.ascendants"):
                _harmonize_person(
                    person,
                    ref_date=raw_death,
                    default_age=_int_between(person.get("age_au_deces"), default=82, min_value=40, max_value=110),
                    min_age=40,
                    max_age=110,
                )
                continue
            if label.startswith("famille.collateraux"):
                _harmonize_person(
                    person,
                    ref_date=raw_death,
                    default_age=_int_between(person.get("age_au_deces"), default=48, min_value=0, max_value=100),
                    min_age=0,
                    max_age=100,
                    can_be_minor=("neveux_nieces" in label),
                )
                continue
            if label == "famille.partenaire":
                _harmonize_person(
                    person,
                    ref_date=raw_death,
                    default_age=_int_between(person.get("age_au_deces"), default=66, min_value=18, max_value=105),
                    min_age=18,
                    max_age=105,
                )

        av = payload.get("assurance_vie")
        if primary_topic == "assurance_vie" and not isinstance(av, dict):
            av = {}
            payload["assurance_vie"] = av
        if isinstance(av, dict):
            contracts = av.get("contrats")
            if primary_topic == "assurance_vie" and (not isinstance(contracts, list) or not contracts):
                av["contrats"] = [{
                    "libelle": f"Contrat {rng.choice(SYNTH_INSURERS)}",
                    "assure_nom": context["defunt_name"],
                }]
                contracts = av["contrats"]
            if isinstance(contracts, list):
                for idx, contract in enumerate(contracts):
                    if not isinstance(contract, dict):
                        contract = {}
                        contracts[idx] = contract
                    contract.setdefault("libelle", f"Contrat {rng.choice(SYNTH_INSURERS)}")
                    contract["assure_nom"] = context["defunt_name"]
                    date_souscription = _parse_iso_date(contract.get("date_souscription"))
                    if date_souscription is None or date_souscription >= raw_death:
                        year = rng.randint(max(1970, raw_death.year - 25), raw_death.year - 1)
                        date_souscription = date(year, rng.randint(1, 12), rng.randint(1, 28))
                    contract["date_souscription"] = date_souscription.isoformat()

                    versements = contract.get("versements")
                    if isinstance(versements, list):
                        for vidx, versement in enumerate(versements):
                            if not isinstance(versement, dict):
                                versement = {}
                                versements[vidx] = versement
                            age_v = _int_between(
                                versement.get("age_assure_au_versement"),
                                default=rng.randint(35, 85),
                                min_value=18,
                                max_value=100,
                            )
                            versement["age_assure_au_versement"] = age_v
                            versement["apres_70_ans"] = age_v >= 70

        if primary_topic == "entreprise_dutreil":
            patrimoine = payload.setdefault("patrimoine", {})
            if isinstance(patrimoine, dict):
                actifs = patrimoine.get("actifs")
                if not isinstance(actifs, list) or not actifs:
                    actifs = [{}]
                    patrimoine["actifs"] = actifs
                first = actifs[0] if isinstance(actifs[0], dict) else {}
                actifs[0] = first
                first.setdefault("type", "ENTREPRISE")
                entreprise = first.get("entreprise")
                if not isinstance(entreprise, dict):
                    entreprise = {}
                    first["entreprise"] = entreprise
                entreprise.setdefault("type", "PME")
                entreprise["est_presente_comme_eligible_dutreil"] = True

        if primary_topic == "donations_reduction":
            liberalites = payload.setdefault("liberalites", {})
            if isinstance(liberalites, dict):
                donations = liberalites.get("donations")
                if not isinstance(donations, list) or not donations:
                    donations = [{}]
                    liberalites["donations"] = donations
                first = donations[0] if isinstance(donations[0], dict) else {}
                donations[0] = first
                first.setdefault("donateur_nom", context["defunt_name"])
                first.setdefault("beneficiaire_nom", context["child_names"][0])
                first.setdefault("type", "DONATION_SIMPLE")
                if first.get("beneficiaire_nom") == first.get("donateur_nom"):
                    first["beneficiaire_nom"] = context["child_names"][0]

        patrimoine = payload.get("patrimoine")
        if isinstance(patrimoine, dict):
            actifs = patrimoine.get("actifs")
            if isinstance(actifs, list):
                for idx, actif in enumerate(actifs):
                    if not isinstance(actif, dict):
                        continue
                    valeur = actif.get("valeur")
                    if isinstance(valeur, (int, float)) and not isinstance(valeur, bool):
                        if valeur <= 0:
                            actif["valeur"] = abs(valeur) + 1
            passifs = patrimoine.get("passifs")
            if isinstance(passifs, list):
                for idx, passif in enumerate(passifs):
                    if not isinstance(passif, dict):
                        continue
                    valeur = passif.get("valeur")
                    if isinstance(valeur, (int, float)) and not isinstance(valeur, bool):
                        if valeur <= 0:
                            passif["valeur"] = abs(valeur) + 1

    def _build_target_payload_for_instruction(
        self,
        instruction: dict[str, Any],
        rng: random.Random,
    ) -> dict[str, Any]:
        dimensions = instruction.get("dimensions", {})
        if not isinstance(dimensions, dict):
            raise ValueError("dimensions manquantes pour construire la cible TOON")
        used_names: set[str] = set()
        defunt_name = self._synth_name(rng, used_names)
        partner_name = self._synth_name(rng, used_names)
        child_names = [self._synth_name(rng, used_names), self._synth_name(rng, used_names)]

        persona = str(dimensions.get("persona") or "")
        primary_topic = str(dimensions.get("primary_topic") or "ordre_heritiers")
        secondary_topic = (
            str(dimensions.get("secondary_topic") or "").strip()
            if isinstance(dimensions.get("secondary_topic"), str)
            else ""
        )
        if primary_topic == "regimes_matrimoniaux" or secondary_topic == "regimes_matrimoniaux":
            topic_statut = "MARIE"
        elif primary_topic == "pacs_concubinage" or secondary_topic == "pacs_concubinage":
            topic_statut = "PACSE" if rng.random() < 0.7 else "CELIBATAIRE"
        elif primary_topic == "famille_recomposee":
            topic_statut = "MARIE"
        else:
            topic_statut = rng.choice(["MARIE", "PACSE", "CELIBATAIRE", "DIVORCE", "VEUF"])

        persona_statut: str | None = None
        if persona == "conjoint":
            persona_statut = "MARIE"
        elif persona == "partenaire_pacs":
            persona_statut = "PACSE"
        elif persona == "concubin":
            persona_statut = "CELIBATAIRE"
        elif persona == "beau_enfant":
            persona_statut = "MARIE"

        statut = persona_statut or topic_statut

        complexity = str(dimensions.get("complexity") or "intermediaire")
        include_proba = {
            "simple": 0.18,
            "intermediaire": 0.28,
            "complexe": 0.4,
            "hard_negative": 0.34,
        }.get(complexity, 0.28)

        context: dict[str, Any] = {
            "defunt_name": defunt_name,
            "partner_name": partner_name,
            "child_names": child_names,
            "used_names": used_names,
            "statut_matrimonial": statut,
        }

        prefixes = self._topic_prefixes_for_dimensions(dimensions)
        leaf_specs = self.master_schema_index.leaf_specs

        mandatory_paths: set[tuple[str, ...]] = {
            ("famille", "defunt", "nom"),
            ("famille", "defunt", "statut_matrimonial"),
            ("famille", "defunt", "date_deces"),
        }
        if statut in {"MARIE", "PACSE"}:
            mandatory_paths.update(
                {
                    ("famille", "partenaire", "nom"),
                    ("famille", "partenaire", "lien", "type"),
                }
            )
        if persona == "concubin":
            mandatory_paths.update(
                {
                    ("famille", "partenaire", "nom"),
                    ("famille", "partenaire", "lien", "type"),
                }
            )
        if persona in {"enfant", "beau_enfant"}:
            mandatory_paths.add(("famille", "descendants", "enfants", "*", "nom"))
        if persona == "beau_enfant":
            mandatory_paths.add(
                ("famille", "descendants", "enfants", "*", "est_d_une_precedente_union")
            )
        if persona == "petit_enfant":
            mandatory_paths.update(
                {
                    ("famille", "descendants", "enfants", "*", "nom"),
                    ("famille", "descendants", "petits_enfants", "*", "nom"),
                    ("famille", "descendants", "petits_enfants", "*", "parent_nom"),
                }
            )
        if persona == "fratrie":
            mandatory_paths.add(("famille", "collateraux", "freres_soeurs", "*", "nom"))
        if persona == "associe":
            mandatory_paths.update(
                {
                    ("patrimoine", "actifs", "*", "type"),
                    ("patrimoine", "actifs", "*", "entreprise", "type"),
                }
            )

        selected_paths: set[tuple[str, ...]] = set(mandatory_paths)
        selected_paths.update(self._required_leaf_paths_for_dimensions(dimensions))
        for path in leaf_specs:
            if any(self._path_matches_prefix(path, prefix) for prefix in prefixes):
                if rng.random() <= include_proba:
                    selected_paths.add(path)
        for extra_prefix in SPARSE_COVERAGE_PREFIXES:
            if rng.random() <= 0.16:
                for path in leaf_specs:
                    if self._path_matches_prefix(path, extra_prefix) and rng.random() <= 0.45:
                        selected_paths.add(path)

        payload: dict[str, Any] = {}

        # Step 1: identities and core legal context
        stage1_paths = [p for p in selected_paths if p[:2] in {("famille", "defunt"), ("famille", "partenaire")}]
        for path in sorted(stage1_paths):
            spec = leaf_specs.get(path)
            if not isinstance(spec, dict):
                continue
            value = self._generate_leaf_value(path, spec, rng=rng, context=context)
            self._set_path_value(payload, path, value)

        # Step 2: thematic payload blocks
        stage2_paths = [p for p in selected_paths if p not in stage1_paths]
        for root in [
            "contexte",
            "famille",
            "liberalites",
            "assurance_vie",
            "patrimoine",
            "indivision",
            "operations_de_partage",
        ]:
            root_paths = [p for p in stage2_paths if p and p[0] == root]
            for path in sorted(root_paths):
                spec = leaf_specs.get(path)
                if not isinstance(spec, dict):
                    continue
                value = self._generate_leaf_value(path, spec, rng=rng, context=context)
                self._set_path_value(payload, path, value)

        # Step 3: repair and harmonize business invariants
        self._repair_business_integrity(
            payload,
            dimensions=dimensions,
            context=context,
            rng=rng,
        )

        return payload

    def submit_case(self, payload: dict[str, Any]) -> dict[str, Any]:
        instruction_id = str(payload.get("instruction_id") or "").strip()
        if not instruction_id:
            raise ValueError("instruction_id manquant")

        case_text = _normalize_text(str(payload.get("case_text") or ""))
        if not case_text:
            raise ValueError("case_text vide")
        if "target_toon" in payload:
            raise ValueError("target_toon non attendu: soumettre uniquement instruction_id + case_text")
        agent_id = str(payload.get("agent_id") or "").strip() or None

        with self.lock:
            instruction = self._find_instruction(instruction_id)
            if instruction is None:
                raise ValueError(f"instruction inconnue: {instruction_id}")
            if any(row.get("instruction_id") == instruction_id for row in self.submitted):
                raise ValueError(f"instruction déjà soumise: {instruction_id}")

            instruction_target_toon = instruction.get("server_target_toon")
            if not isinstance(instruction_target_toon, str) or not instruction_target_toon.strip():
                raise ValueError("cible TOON serveur introuvable pour cette instruction")
            target_toon, decoded_target = _normalize_target_toon(instruction_target_toon)

            missing_names = _missing_names_from_case_text(case_text, decoded_target)
            if missing_names:
                preview = ", ".join(missing_names[:3])
                if len(missing_names) > 3:
                    preview += ", …"
                raise ValueError(
                    "incohérence texte/target_toon: noms absents de l'énoncé "
                    f"({preview})"
                )

            validation = self._validate_submission(case_text)
            if re.search(r"\b[a-z]+_[a-z_]+\b", case_text):
                raise ValueError(
                    "format invalide: ne pas inclure de clés internes en snake_case dans l'énoncé "
                    "(ex: statut_matrimonial, option_successorale)"
                )
            caps_match = FORBIDDEN_CAPS_UNDERSCORE_RE.search(case_text)
            if caps_match:
                token = caps_match.group(0)
                raise ValueError(
                    "format invalide: ne pas inclure de codes en MAJUSCULES_AVEC_UNDERSCORE dans l'énoncé "
                    f"(ex: PARTENAIRE_PACS, NEVEU_NIECE). Reçu: {token!r}. "
                    "Traduire en français naturel (sans underscores)."
                )
            if FORBIDDEN_PYTHON_BOOL_RE.search(case_text):
                raise ValueError(
                    "format invalide: ne pas inclure de booléens Python ('True'/'False') dans l'énoncé. "
                    "Utiliser une formulation française (oui/non)."
                )
            if FORBIDDEN_PATH_DUMP_RE.search(case_text):
                raise ValueError(
                    "format invalide: ne pas inclure de chemins type 'famille > defunt > ...' dans l'énoncé. "
                    "Reformuler en phrases françaises."
                )
            if FORBIDDEN_ENUM_BASIC_RE.search(case_text):
                raise ValueError(
                    "format invalide: ne pas inclure de tokens d'énumération en majuscules (ex: CELIBATAIRE, "
                    "JOURS, MOIS). Traduire en français naturel."
                )
            if FORBIDDEN_SCHEMAISH_PHRASES_RE.search(case_text) or FORBIDDEN_SCHEMAISH_DEFUNT_FIELDS_RE.search(
                case_text
            ):
                raise ValueError(
                    "format invalide: l'énoncé ressemble à un dump de champs (ex: 'famille defunt ...', "
                    "'defunt date deces ...'). Reformuler en français naturel."
                )
            if case_text.count(";") > MAX_SEMICOLONS_IN_CASE_TEXT:
                raise ValueError(
                    "format invalide: trop de séparateurs ';' (probable dump de champs). "
                    f"Limite: {MAX_SEMICOLONS_IN_CASE_TEXT}."
                )
            if case_text.count(":") > MAX_COLONS_IN_CASE_TEXT:
                raise ValueError(
                    "format invalide: trop de séparateurs ':' (probable dump de champs). "
                    f"Limite: {MAX_COLONS_IN_CASE_TEXT}."
                )
            record = {
                "instruction_id": instruction_id,
                "agent_id": agent_id or instruction.get("agent_id"),
                "submitted_at": _utc_now(),
                "case_text": case_text,
                "target_toon": target_toon,
                "target_source": "server_instruction",
                "validation": validation,
                "dimensions": instruction.get("dimensions", {}),
            }
            self.submitted.append(record)
            _append_jsonl(self.submitted_path, record)
            self._write_submission_file(record)
            self._write_instruction_file(instruction, submission=record)
            self._refresh_training_exports()
            self._refresh_summary()
            return {
                "stored": True,
                "validation": validation,
                "target_toon_lines": len(target_toon.splitlines()),
                "coverage": self._coverage_snapshot(),
            }

    def _find_instruction(self, instruction_id: str) -> dict[str, Any] | None:
        for row in self.issued:
            if row.get("instruction_id") == instruction_id:
                return row
        return None

    def _dimension_counts(self) -> dict[str, dict[str, int]]:
        counters = {
            "persona": {},
            "voice": {},
            "format": {},
            "length_band": {},
            "noise": {},
            "numeric_density": {},
            "date_precision": {},
            "complexity": {},
            "primary_topic": {},
            "hard_negative_mode": {},
            "hard_negative_intensity": {},
        }
        for row in self.issued:
            dimensions = row.get("dimensions", {})
            if not isinstance(dimensions, dict):
                continue
            for key in list(counters):
                value = dimensions.get(key)
                if isinstance(value, str) and value:
                    bucket = counters[key]
                    bucket[value] = bucket.get(value, 0) + 1
        return counters

    def _recent_signatures(self, limit: int = 12) -> set[str]:
        signatures: set[str] = set()
        for row in self.issued[-limit:]:
            signature = row.get("signature")
            if isinstance(signature, str):
                signatures.add(signature)
        return signatures

    def _build_instruction(self, *, agent_id: str | None, force_topic: str | None) -> dict[str, Any]:
        sequence = len(self.issued) + 1
        rng = random.Random(int(self.config["seed"]) + sequence)
        counts = self._dimension_counts()

        persona = _pick_underrepresented(PERSONA_TARGETS, counts["persona"], rng)
        voice = _pick_underrepresented(VOICE_TARGETS, counts["voice"], rng)
        format_name = _pick_underrepresented(FORMAT_TARGETS, counts["format"], rng)
        length_band = _pick_underrepresented(LENGTH_TARGETS, counts["length_band"], rng)
        noise = _pick_underrepresented(NOISE_TARGETS, counts["noise"], rng)
        numeric_density = _pick_underrepresented(NUMERIC_TARGETS, counts["numeric_density"], rng)
        if numeric_density == "montants_et_dates":
            date_precision = _pick_underrepresented(
                DATE_PRECISION_TARGETS,
                counts["date_precision"],
                rng,
                exclude={"aucune"},
            )
        else:
            date_precision = _pick_underrepresented(DATE_PRECISION_TARGETS, counts["date_precision"], rng)
        complexity = _pick_underrepresented(COMPLEXITY_TARGETS, counts["complexity"], rng)

        blocked_topics: set[str] = set()
        if persona in {"partenaire_pacs", "concubin"}:
            blocked_topics.add("regimes_matrimoniaux")

        if force_topic and force_topic in TOPIC_TARGETS:
            primary_topic = force_topic
        else:
            primary_topic = _pick_underrepresented(
                TOPIC_TARGETS,
                counts["primary_topic"],
                rng,
                exclude=blocked_topics,
            )

        secondary_topic: str | None = None
        if complexity in {"complexe", "hard_negative"} or rng.random() < 0.55:
            secondary_topic = _pick_underrepresented(
                TOPIC_TARGETS,
                counts["primary_topic"],
                rng,
                exclude={primary_topic} | blocked_topics,
            )

        hard_negative_mode: str | None = None
        hard_negative_intensity: str | None = None
        if complexity == "hard_negative":
            hard_negative_intensity = _pick_underrepresented(
                HARD_NEGATIVE_INTENSITY_TARGETS,
                counts["hard_negative_intensity"],
                rng,
            )
            hard_negative_mode = _pick_underrepresented(
                HARD_NEGATIVE_TARGETS,
                counts["hard_negative_mode"],
                rng,
            )

        signature = "|".join(
            filter(
                None,
                (
                    persona,
                    voice,
                    format_name,
                    length_band,
                    noise,
                    numeric_density,
                    date_precision,
                    complexity,
                    hard_negative_intensity,
                    primary_topic,
                    secondary_topic,
                ),
            )
        )
        if signature in self._recent_signatures():
            format_name = _pick_underrepresented(
                FORMAT_TARGETS,
                counts["format"],
                rng,
                exclude={format_name},
            )
            signature = "|".join(
                filter(
                    None,
                    (
                        persona,
                        voice,
                        format_name,
                        length_band,
                        noise,
                        numeric_density,
                        date_precision,
                        complexity,
                        hard_negative_intensity,
                        primary_topic,
                        secondary_topic,
                    ),
                )
            )

        instruction_id = f"INS-{sequence:04d}"
        dimensions = {
            "persona": persona,
            "voice": voice,
            "format": format_name,
            "length_band": length_band,
            "noise": noise,
            "numeric_density": numeric_density,
            "date_precision": date_precision,
            "complexity": complexity,
            "primary_topic": primary_topic,
            "secondary_topic": secondary_topic,
            "hard_negative_mode": hard_negative_mode,
            "hard_negative_intensity": hard_negative_intensity,
        }
        examples = self._pick_reference_examples(primary_topic, secondary_topic, rng)
        must_include = self._collect_mandatory_elements(dimensions)
        must_avoid = self._collect_must_avoid(dimensions)
        style_brief = self._build_style_brief(dimensions)
        dimension_guide = self._build_dimension_guide(dimensions)
        prompt = self._render_instruction_prompt(
            dimensions,
            examples,
            must_include=must_include,
            must_avoid=must_avoid,
        )
        return {
            "instruction_id": instruction_id,
            "agent_id": agent_id,
            "issued_at": _utc_now(),
            "signature": signature,
            "dimensions": dimensions,
            "dimension_guide": dimension_guide,
            "style_brief": style_brief,
            "must_include": must_include,
            "must_avoid": must_avoid,
            "response_format": {
                "root_type": "object",
                "required_keys": ["case_text"],
                "case_text_rule": "Chaîne libre en français contenant l'énoncé complet.",
                "additional_root_keys_allowed": False,
            },
            "submission_contract": {
                "required_fields": ["instruction_id", "case_text"],
                "note": (
                    "Le serveur fournit target_toon dans l'instruction. "
                    "Soumettre uniquement l'énoncé (case_text)."
                ),
            },
            "reference_examples": examples,
            "prompt": prompt,
        }

    def _pick_reference_examples(
        self,
        primary_topic: str,
        secondary_topic: str | None,
        rng: random.Random,
    ) -> list[dict[str, str]]:
        if not self.seed_cases:
            return []

        topics = [primary_topic]
        if secondary_topic:
            topics.append(secondary_topic)

        keywords: list[str] = []
        for topic in topics:
            keywords.extend(TOPIC_TEMPLATES[topic]["keywords"])

        candidates: list[CorpusSeed] = []
        lowered_keywords = [_normalize_key(word) for word in keywords]
        for seed in self.seed_cases:
            seed_key = _normalize_key(seed.text)
            if any(keyword in seed_key for keyword in lowered_keywords):
                candidates.append(seed)

        if len(candidates) < 2:
            candidates = list(self.seed_cases)

        rng.shuffle(candidates)
        selected = candidates[:2]
        return [
            {
                "case_id": item.case_id,
                "source_type": item.source_type,
                "source_name": item.source_name,
                "excerpt": (item.text[:220] + "…") if len(item.text) > 220 else item.text,
            }
            for item in selected
        ]

    def _render_instruction_prompt(
        self,
        dimensions: dict[str, str | None],
        examples: list[dict[str, str]],
        *,
        must_include: list[str] | None = None,
        must_avoid: list[str] | None = None,
    ) -> str:
        primary_topic = str(dimensions["primary_topic"])
        secondary_topic = dimensions.get("secondary_topic")
        hard_negative_mode = dimensions.get("hard_negative_mode")

        topic_labels = [TOPIC_TEMPLATES[primary_topic]["label"]]
        if isinstance(secondary_topic, str) and secondary_topic:
            topic_labels.append(TOPIC_TEMPLATES[secondary_topic]["label"])

        deduped_elements = must_include if must_include is not None else self._collect_mandatory_elements(dimensions)
        forbidden_elements = must_avoid if must_avoid is not None else self._collect_must_avoid(dimensions)
        hard_negative_intensity = dimensions.get("hard_negative_intensity")

        lines = [
            "Génère uniquement un énoncé (case_text) pour un cas de succession en français.",
            f"Persona : {PERSONA_LABELS[str(dimensions['persona'])]}.",
            f"Tournure : {VOICE_LABELS[str(dimensions['voice'])]}.",
            f"Format : {FORMAT_LABELS[str(dimensions['format'])]}.",
            f"Longueur visée : {LENGTH_LABELS[str(dimensions['length_band'])]}.",
            f"Niveau de bruit : {NOISE_LABELS[str(dimensions['noise'])]}.",
            f"Densité chiffrée : {NUMERIC_LABELS[str(dimensions['numeric_density'])]}.",
            f"Précision temporelle : {DATE_PRECISION_LABELS[str(dimensions['date_precision'])]}.",
            f"Niveau : {COMPLEXITY_LABELS[str(dimensions['complexity'])]}.",
            f"Sujet principal : {topic_labels[0]}.",
        ]
        if len(topic_labels) > 1:
            lines.append(f"Sujet secondaire : {topic_labels[1]}.")
        if isinstance(hard_negative_mode, str) and hard_negative_mode:
            lines.append(f"Mode hard negative : {HARD_NEGATIVE_LABELS[hard_negative_mode]}.")
        if isinstance(hard_negative_intensity, str) and hard_negative_intensity:
            lines.append(
                f"Intensité hard negative : {HARD_NEGATIVE_INTENSITY_LABELS[hard_negative_intensity]}."
            )
        lines.append("Contraintes :")
        for item in deduped_elements:
            lines.append(f"- {item}")
        lines.append("À éviter :")
        for item in forbidden_elements:
            lines.append(f"- {item}")
        lines.append("Sortie attendue : texte brut uniquement (l'énoncé), sans JSON, sans TOON, sans analyse.")
        if examples:
            lines.append("Repères de style (à ne pas recopier mot pour mot) :")
            for example in examples:
                lines.append(
                    f"- [{example['case_id']}] {example['excerpt']}"
                )
        return "\n".join(lines)

    def _augment_prompt_with_target_toon(self, base_prompt: str, target_toon: str) -> str:
        base = base_prompt.strip()
        lines: list[str] = []
        if base:
            lines.append(base)
            lines.append("")
        lines.extend(
            [
                "Source de vérité des faits: le TOON ci-dessous.",
                "Règle A: chaque information présente dans le TOON doit apparaître dans l'énoncé, mais reformulée en français naturel.",
                "  - Ne jamais recopier des codes d'énumération du TOON (ex: PARTENAIRE_PACS, NEVEU_NIECE, PROPRE_DEFUNT, IMPOT_SUCCESSION).",
                "  - Si une valeur ressemble à `MAJUSCULES_AVEC_UNDERSCORE`, tu dois la traduire en mots (sans underscores).",
                "  - Exemples: PARTENAIRE_PACS -> partenaire de PACS ; NEVEU_NIECE -> neveu / nièce ;",
                "    COMMUNAUTE_REDUITE_AUX_ACQUETS -> communauté réduite aux acquêts ; A_TITRE_UNIVERSEL -> à titre universel.",
                "Règle B: ne pas ajouter de nouvelles informations structurées (noms, dates, montants, liens, biens) absentes du TOON.",
                "Règle C: ne pas donner la solution juridique, seulement les faits.",
                "Règle D: ne pas recopier la structure ou les clés du TOON (pas de `snake_case`, pas de `champ: valeur`, pas de JSON/TOON dans la réponse).",
                "Règle E: tu peux utiliser des sigles usuels (PACS, SCI, SARL, AV), mais pas des tokens en MAJUSCULES_AVEC_UNDERSCORE.",
                "Sortie attendue: texte brut uniquement (l'énoncé), sans JSON.",
                "",
                "TOON:",
                target_toon.strip(),
            ]
        )
        return "\n".join(lines).strip()

    def _validate_submission(self, case_text: str) -> dict[str, Any]:
        normalized = _normalize_key(case_text)
        warnings: list[str] = []

        exact_duplicate = False
        max_similarity = 0.0
        closest_case_id: str | None = None

        for seed in self.seed_cases:
            seed_key = _normalize_key(seed.text)
            if normalized == seed_key:
                exact_duplicate = True
                closest_case_id = seed.case_id
                max_similarity = 1.0
                break
            score = _jaccard_similarity(case_text, seed.text)
            if score > max_similarity:
                max_similarity = score
                closest_case_id = seed.case_id

        if not exact_duplicate:
            for row in self.submitted:
                existing = row.get("case_text")
                if not isinstance(existing, str):
                    continue
                existing_key = _normalize_key(existing)
                if normalized == existing_key:
                    exact_duplicate = True
                    closest_case_id = str(row.get("instruction_id") or "")
                    max_similarity = 1.0
                    break
                score = _jaccard_similarity(case_text, existing)
                if score > max_similarity:
                    max_similarity = score
                    closest_case_id = str(row.get("instruction_id") or "")

        if exact_duplicate:
            warnings.append("doublon exact détecté")
        elif max_similarity >= 0.72:
            warnings.append("cas très proche d'un cas existant")

        if len(case_text) < 60:
            warnings.append("énoncé très court")

        if re.search(r"\b[a-z]+_[a-z_]+\b", case_text):
            warnings.append("le texte contient du 'snake_case' (probable recrachage de schéma)")
        if FORBIDDEN_CAPS_UNDERSCORE_RE.search(case_text):
            warnings.append(
                "le texte contient un token en MAJUSCULES_AVEC_UNDERSCORE (probable recrachage d'énumération)"
            )

        return {
            "word_count": len(case_text.split()),
            "char_count": len(case_text),
            "contains_digits": bool(re.search(r"\d", case_text)),
            "exact_duplicate": exact_duplicate,
            "max_similarity": round(max_similarity, 4),
            "closest_reference": closest_case_id,
            "warnings": warnings,
        }

    def _coverage_snapshot(self) -> dict[str, Any]:
        counts = self._dimension_counts()
        generation_target = int(self.config["generation_target"])
        target_total_cases = int(self.config["target_total_cases"])
        hard_negative_base = generation_target * COMPLEXITY_TARGETS["hard_negative"]
        summary = {
            "target_total_cases": target_total_cases,
            "generation_target": generation_target,
            "seed_cases": len(self.seed_cases),
            "issued": len(self.issued),
            "submitted": len(self.submitted),
            "training_cases_current": len(self.submitted),
            "remaining": max(generation_target - len(self.submitted), 0),
            "dimensions": {
                "persona": self._dimension_progress(PERSONA_TARGETS, counts["persona"]),
                "voice": self._dimension_progress(VOICE_TARGETS, counts["voice"]),
                "format": self._dimension_progress(FORMAT_TARGETS, counts["format"]),
                "length_band": self._dimension_progress(LENGTH_TARGETS, counts["length_band"]),
                "noise": self._dimension_progress(NOISE_TARGETS, counts["noise"]),
                "numeric_density": self._dimension_progress(NUMERIC_TARGETS, counts["numeric_density"]),
                "date_precision": self._dimension_progress(DATE_PRECISION_TARGETS, counts["date_precision"]),
                "complexity": self._dimension_progress(COMPLEXITY_TARGETS, counts["complexity"]),
                "primary_topic": self._dimension_progress(TOPIC_TARGETS, counts["primary_topic"]),
                "hard_negative_mode": self._dimension_progress(
                    HARD_NEGATIVE_TARGETS,
                    counts["hard_negative_mode"],
                    base_total=hard_negative_base,
                ),
                "hard_negative_intensity": self._dimension_progress(
                    HARD_NEGATIVE_INTENSITY_TARGETS,
                    counts["hard_negative_intensity"],
                    base_total=hard_negative_base,
                ),
            },
        }
        return summary

    def _refresh_training_exports(self) -> None:
        generated_rows: list[str] = []
        for row in self.submitted:
            case_text = row.get("case_text")
            target_toon = row.get("target_toon")
            if (
                isinstance(case_text, str)
                and case_text.strip()
                and isinstance(target_toon, str)
                and target_toon.strip()
            ):
                generated_rows.append(
                    json.dumps(_pair_training_record(case_text, target_toon.strip()), ensure_ascii=False)
                )

        self.generated_train_path.write_text(
            ("\n".join(generated_rows) + ("\n" if generated_rows else "")),
            encoding="utf-8",
        )

        with self.full_train_path.open("w", encoding="utf-8") as handle:
            for row_json in generated_rows:
                handle.write(row_json + "\n")

    def _dimension_progress(
        self,
        targets: dict[str, float],
        counts: dict[str, int],
        *,
        base_total: float | None = None,
    ) -> dict[str, dict[str, float | int]]:
        total = float(base_total if base_total is not None else int(self.config["generation_target"]))
        progress: dict[str, dict[str, float | int]] = {}
        for key, share in targets.items():
            target_count = round(total * share, 1)
            current = counts.get(key, 0)
            progress[key] = {
                "target_share": share,
                "target_count": target_count,
                "current": current,
                "gap": round(target_count - current, 1),
            }
        return progress

    def _refresh_summary(self) -> None:
        snapshot = self._coverage_snapshot()
        self.summary_json_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        lines = [
            "# Case Instruction Server",
            "",
            f"- target_total_cases: {snapshot['target_total_cases']}",
            f"- generation_target: {snapshot['generation_target']}",
            f"- seed_cases: {snapshot['seed_cases']}",
            f"- issued: {snapshot['issued']}",
            f"- submitted: {snapshot['submitted']}",
            f"- training_cases_current: {snapshot['training_cases_current']}",
            f"- remaining: {snapshot['remaining']}",
            "",
            "## Coverage",
        ]
        for dimension, values in snapshot["dimensions"].items():
            lines.append("")
            lines.append(f"### {dimension}")
            for key, row in values.items():
                lines.append(
                    f"- {key}: current={row['current']} target={row['target_count']} gap={row['gap']}"
                )
        self.summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_instruction_file(
        self,
        instruction: dict[str, Any],
        submission: dict[str, Any] | None = None,
    ) -> None:
        payload = dict(instruction)
        payload["status"] = "submitted" if submission is not None else "issued"
        if submission is not None:
            payload["submission"] = submission
        target = self.instructions_dir / f"{instruction['instruction_id']}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_submission_file(self, submission: dict[str, Any]) -> None:
        target = self.submissions_dir / f"{submission['instruction_id']}.json"
        target.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")


class InstructionRequestHandler(BaseHTTPRequestHandler):
    server: "InstructionHTTPServer"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(HTTPStatus.OK, self.server.app.health())
            return
        if parsed.path == "/dashboard":
            self._send_json(HTTPStatus.OK, self.server.app.dashboard())
            return
        if parsed.path == "/next-instruction":
            params = parse_qs(parsed.query)
            payload = {
                "agent_id": params.get("agent_id", [None])[0],
                "topic": params.get("topic", [None])[0],
            }
            self._handle_json_call(self.server.app.next_instruction, payload)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        body = self._read_json_body()
        if parsed.path == "/next-instruction":
            self._handle_json_call(self.server.app.next_instruction, body)
            return
        if parsed.path == "/submit-case":
            self._handle_json_call(self.server.app.submit_case, body)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _handle_json_call(self, handler: Any, payload: dict[str, Any]) -> None:
        try:
            response = handler(payload)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, response)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class InstructionHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], app: InstructionServerApp) -> None:
        super().__init__(server_address, InstructionRequestHandler)
        self.app = app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serveur de consignes pour génération manuelle de cas de succession."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--state-dir", default="data/case_instruction_server")
    parser.add_argument("--corpus-file", default=str(DEFAULT_CORPUS_FILE))
    parser.add_argument("--master-schema-file", default=str(DEFAULT_MASTER_SCHEMA_FILE))
    parser.add_argument("--target-total-cases", type=int, default=DEFAULT_TARGET_TOTAL_CASES)
    parser.add_argument("--generation-target", type=int, default=None)
    parser.add_argument("--campaign-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_target = args.generation_target
    if generation_target is None:
        generation_target = args.campaign_size

    app = InstructionServerApp(
        state_dir=Path(args.state_dir),
        corpus_file=Path(args.corpus_file),
        master_schema_file=Path(args.master_schema_file),
        target_total_cases=args.target_total_cases,
        generation_target=generation_target,
        seed=args.seed,
    )
    server = InstructionHTTPServer((args.host, args.port), app)
    print(
        json.dumps(
            {
                "host": args.host,
                "port": args.port,
                "state_dir": str(Path(args.state_dir)),
                "seed_cases": len(app.seed_cases),
                "master_schema_file": str(Path(args.master_schema_file)),
                "target_total_cases": app.config["target_total_cases"],
                "generation_target": app.config["generation_target"],
            },
            ensure_ascii=False,
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
