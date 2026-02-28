from __future__ import annotations

import argparse
import json
import random
import re
import threading
import unicodedata
from dataclasses import dataclass
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ministral_ft.e2e_case_corpus import mistral_training_record_from_text


DEFAULT_TARGET_TOTAL_CASES = 5000
DEFAULT_SEED = 42
DEFAULT_CORPUS_FILE = Path("data/succession_e2e/e2e_cases.jsonl")
CONFIG_FILENAME = "config.json"
ISSUED_FILENAME = "issued_instructions.jsonl"
SUBMITTED_FILENAME = "generated_cases.jsonl"
SUMMARY_JSON_FILENAME = "summary.json"
SUMMARY_MD_FILENAME = "summary.md"
GENERATED_TRAIN_FILENAME = "generated_cases_train_mistral.jsonl"
FULL_TRAIN_FILENAME = "full_training_cases_mistral.jsonl"

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
        self._refresh_training_exports()
        self._refresh_summary()

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
            "training_cases_current": len(self.seed_cases) + len(self.submitted),
        }

    def dashboard(self) -> dict[str, Any]:
        if self.summary_json_path.exists():
            return json.loads(self.summary_json_path.read_text(encoding="utf-8"))
        return self._coverage_snapshot()

    def next_instruction(self, payload: dict[str, Any]) -> dict[str, Any]:
        agent_id = str(payload.get("agent_id") or "").strip() or None
        force_topic = str(payload.get("topic") or "").strip() or None

        with self.lock:
            instruction = self._build_instruction(agent_id=agent_id, force_topic=force_topic)
            self.issued.append(instruction)
            _append_jsonl(self.issued_path, instruction)
            self._write_instruction_file(instruction)
            self._refresh_summary()
            return {
                "instruction": instruction,
                "coverage": self._coverage_snapshot(),
            }

    def submit_case(self, payload: dict[str, Any]) -> dict[str, Any]:
        instruction_id = str(payload.get("instruction_id") or "").strip()
        if not instruction_id:
            raise ValueError("instruction_id manquant")

        case_text = _normalize_text(str(payload.get("case_text") or ""))
        if not case_text:
            raise ValueError("case_text vide")

        agent_id = str(payload.get("agent_id") or "").strip() or None

        with self.lock:
            instruction = self._find_instruction(instruction_id)
            if instruction is None:
                raise ValueError(f"instruction inconnue: {instruction_id}")
            if any(row.get("instruction_id") == instruction_id for row in self.submitted):
                raise ValueError(f"instruction déjà soumise: {instruction_id}")

            validation = self._validate_submission(case_text)
            record = {
                "instruction_id": instruction_id,
                "agent_id": agent_id or instruction.get("agent_id"),
                "submitted_at": _utc_now(),
                "case_text": case_text,
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

        if force_topic and force_topic in TOPIC_TARGETS:
            primary_topic = force_topic
        else:
            primary_topic = _pick_underrepresented(TOPIC_TARGETS, counts["primary_topic"], rng)

        secondary_topic: str | None = None
        if complexity in {"complexe", "hard_negative"} or rng.random() < 0.55:
            secondary_topic = _pick_underrepresented(
                TOPIC_TARGETS,
                counts["primary_topic"],
                rng,
                exclude={primary_topic},
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
        prompt = self._render_instruction_prompt(dimensions, examples)
        return {
            "instruction_id": instruction_id,
            "agent_id": agent_id,
            "issued_at": _utc_now(),
            "signature": signature,
            "dimensions": dimensions,
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
    ) -> str:
        primary_topic = str(dimensions["primary_topic"])
        secondary_topic = dimensions.get("secondary_topic")
        hard_negative_mode = dimensions.get("hard_negative_mode")

        topic_labels = [TOPIC_TEMPLATES[primary_topic]["label"]]
        if isinstance(secondary_topic, str) and secondary_topic:
            topic_labels.append(TOPIC_TEMPLATES[secondary_topic]["label"])

        mandatory_elements = []
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
        hard_negative_intensity = dimensions.get("hard_negative_intensity")
        if isinstance(hard_negative_intensity, str) and hard_negative_intensity:
            mandatory_elements.extend(HARD_NEGATIVE_INTENSITY_REQUIREMENTS[hard_negative_intensity])

        deduped_elements: list[str] = []
        seen = set()
        for item in mandatory_elements:
            if item in seen:
                continue
            seen.add(item)
            deduped_elements.append(item)

        lines = [
            "Génère un seul énoncé de succession en français.",
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
        lines.extend(
            [
                "- Ne donne ni solution, ni analyse, ni liste de points juridiques.",
                "- Réponds uniquement par l'énoncé final.",
            ]
        )
        if examples:
            lines.append("Repères de style (à ne pas recopier mot pour mot) :")
            for example in examples:
                lines.append(
                    f"- [{example['case_id']}] {example['excerpt']}"
                )
        return "\n".join(lines)

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
            "training_cases_current": len(self.seed_cases) + len(self.submitted),
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
            if isinstance(case_text, str) and case_text.strip():
                generated_rows.append(
                    json.dumps(mistral_training_record_from_text(case_text), ensure_ascii=False)
                )

        self.generated_train_path.write_text(
            ("\n".join(generated_rows) + ("\n" if generated_rows else "")),
            encoding="utf-8",
        )

        with self.full_train_path.open("w", encoding="utf-8") as handle:
            for seed in self.seed_cases:
                handle.write(
                    json.dumps(mistral_training_record_from_text(seed.text), ensure_ascii=False) + "\n"
                )
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
