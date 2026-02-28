from __future__ import annotations

import ast
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


E2E_GLOBS = (
    "src/legal-tests/**/*E2E*.spec.ts",
    "src/legal-tests/**/*_E2E_*.spec.ts",
)
EXPLICIT_E2E_FILES = (
    "src/legal-tests/CodeCivil_Art767_PensionAlimentaire_Extraction.spec.ts",
    "src/legal-tests/Systeme_Orchestration.spec.ts",
)
SEARCH_BAR_SCENARIOS_FILE = Path("src/components/succession-search-bar.scenarios.ts")
SEARCH_BAR_TESTS_GLOB = "src/legal-tests/succession-search-bar/*E2E.spec.ts"
TRAINING_SYSTEM_PROMPT = (
    "Tu es un générateur d'énoncés de succession en français. "
    "Tu écris des cas réalistes, variés, plausibles et exploitables pour des tests juridiques."
)
TRAINING_USER_PROMPT = (
    "Rédige un énoncé de succession crédible en français. "
    "Réponds uniquement par l'énoncé."
)
PROMPT_VARIABLE_RE = re.compile(
    r"const\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<literal>`[\s\S]*?`|'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")\s*;",
    re.DOTALL,
)
PROMPT_ARRAY_RE = re.compile(
    r"const\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*\[(?P<body>[\s\S]*?)\]\.join\((?P<join>[\s\S]*?)\)\s*;",
    re.DOTALL,
)
SCENARIO_FILE_RE = re.compile(
    r"\{\s*label:\s*(?P<label>'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")\s*,\s*prompt:\s*"
    r"(?P<prompt>`[\s\S]*?`|'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")\s*\}",
    re.DOTALL,
)
SEARCH_BAR_LABEL_RE = re.compile(
    r"runSuccessionSearchBarScenarioE2E\((?P<label>'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")\)"
)
CASES_ROOT_RE = re.compile(
    r"casesRoot\s*=\s*path\.join\(root,\s*['\"]cases['\"],\s*['\"](?P<folder>[^'\"]+)['\"]\)"
)
SCENARIO_FILTER_RE = re.compile(
    r"scenarioDirs\s*=\s*listScenarioDirs\(casesRoot\)\.filter\(\(d\)\s*=>\s*(?P<regex>/.*?/)\.test\(d\)\)",
    re.DOTALL,
)
STRING_CHUNK_RE = re.compile(r"('(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")", re.DOTALL)
CASE_HINT_RE = re.compile(
    r"(d[ée]c[èe]s|d[ée]funt|d[ée]c[ée]d|succession|h[ée]rit|patrimoine|usufruit|testament|"
    r"donation|assurance\s*-?\s*vie|mari[ée]|pacs|concubin|indivision|dette|enfant)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class CaseRecord:
    case_id: str
    source_type: str
    origin: str
    source_path: str
    source_name: str
    text: str


def _normalize_whitespace(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalize_for_key(value: str) -> str:
    text = _normalize_whitespace(value).lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return text


def _short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _looks_like_prompt_variable(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("text", "scenario", "prompt", "enonce"))


def _looks_like_case_text(value: str) -> bool:
    normalized = _normalize_whitespace(value)
    if len(normalized) < 24:
        return False
    return bool(CASE_HINT_RE.search(normalized))


def _parse_string_literal(token: str) -> str:
    literal = token.strip()
    if not literal:
        return ""
    if literal.startswith("`") and literal.endswith("`"):
        return literal[1:-1]
    try:
        parsed = ast.literal_eval(literal)
    except (SyntaxError, ValueError):
        quote = literal[0]
        if quote in {"'", '"'} and literal.endswith(quote):
            inner = literal[1:-1]
            return bytes(inner, "utf-8").decode("unicode_escape")
        return literal
    return parsed if isinstance(parsed, str) else str(parsed)


def discover_e2e_specs(w5_root: str | Path) -> list[Path]:
    root = Path(w5_root).resolve()
    files: set[Path] = set()
    for pattern in E2E_GLOBS:
        files.update(root.glob(pattern))
    for relative in EXPLICIT_E2E_FILES:
        candidate = root / relative
        if candidate.exists():
            files.add(candidate)
    return sorted(path for path in files if path.is_file())


def _build_record(
    *,
    source_type: str,
    origin: str,
    source_path: str,
    source_name: str,
    text: str,
) -> CaseRecord:
    normalized = _normalize_whitespace(text)
    record_key = "|".join((source_type, source_path, normalized))
    return CaseRecord(
        case_id=f"{source_type}_{_short_hash(record_key)}",
        source_type=source_type,
        origin=origin,
        source_path=source_path,
        source_name=source_name,
        text=normalized,
    )


def _extract_case_directory_records(spec_path: Path, w5_root: Path) -> list[CaseRecord]:
    text = spec_path.read_text(encoding="utf-8")
    root_match = CASES_ROOT_RE.search(text)
    if root_match is None:
        return []
    folder = root_match.group("folder")
    cases_root = w5_root / "cases" / folder
    if not cases_root.exists():
        return []

    patterns = [match.group("regex")[1:-1] for match in SCENARIO_FILTER_RE.finditer(text)]
    if not patterns:
        return []

    compiled = [re.compile(pattern) for pattern in patterns]
    records: list[CaseRecord] = []
    for child in sorted(cases_root.iterdir()):
        if not child.is_dir():
            continue
        if not any(regex.search(child.name) for regex in compiled):
            continue
        enonce_path = child / "ENONCE.md"
        if not enonce_path.exists():
            continue
        enonce_text = _normalize_whitespace(enonce_path.read_text(encoding="utf-8"))
        if not enonce_text:
            continue
        records.append(
            _build_record(
                source_type="terrain_case",
                origin=str(spec_path.relative_to(w5_root)),
                source_path=str(enonce_path.relative_to(w5_root)),
                source_name=child.name,
                text=enonce_text,
            )
        )
    return records


def _extract_inline_literal_records(spec_path: Path, w5_root: Path) -> list[CaseRecord]:
    text = spec_path.read_text(encoding="utf-8")
    records: list[CaseRecord] = []

    for match in PROMPT_VARIABLE_RE.finditer(text):
        name = match.group("name")
        if not _looks_like_prompt_variable(name):
            continue
        value = _parse_string_literal(match.group("literal"))
        if not _looks_like_case_text(value):
            continue
        records.append(
            _build_record(
                source_type="inline_spec",
                origin=str(spec_path.relative_to(w5_root)),
                source_path=str(spec_path.relative_to(w5_root)),
                source_name=name,
                text=value,
            )
        )

    for match in PROMPT_ARRAY_RE.finditer(text):
        name = match.group("name")
        if not _looks_like_prompt_variable(name):
            continue
        chunks = [_parse_string_literal(token) for token in STRING_CHUNK_RE.findall(match.group("body"))]
        if not chunks:
            continue
        joiner = _parse_string_literal(match.group("join").strip()) if match.group("join").strip() else ""
        value = joiner.join(chunks)
        if not _looks_like_case_text(value):
            continue
        records.append(
            _build_record(
                source_type="inline_spec",
                origin=str(spec_path.relative_to(w5_root)),
                source_path=str(spec_path.relative_to(w5_root)),
                source_name=name,
                text=value,
            )
        )

    return records


def _parse_search_bar_scenarios(scenarios_path: Path) -> dict[str, str]:
    content = scenarios_path.read_text(encoding="utf-8")
    parsed: dict[str, str] = {}
    for match in SCENARIO_FILE_RE.finditer(content):
        label = _parse_string_literal(match.group("label"))
        prompt = _parse_string_literal(match.group("prompt"))
        if label and prompt:
            parsed[label] = _normalize_whitespace(prompt)
    return parsed


def _discover_search_bar_labels(w5_root: Path) -> set[str]:
    labels: set[str] = set()
    for spec_path in sorted(w5_root.glob(SEARCH_BAR_TESTS_GLOB)):
        content = spec_path.read_text(encoding="utf-8")
        for match in SEARCH_BAR_LABEL_RE.finditer(content):
            labels.add(_parse_string_literal(match.group("label")))
    return labels


def _extract_search_bar_records(w5_root: Path) -> list[CaseRecord]:
    scenarios_path = w5_root / SEARCH_BAR_SCENARIOS_FILE
    if not scenarios_path.exists():
        return []

    labels = _discover_search_bar_labels(w5_root)
    if not labels:
        return []

    scenarios = _parse_search_bar_scenarios(scenarios_path)
    records: list[CaseRecord] = []
    for label in sorted(labels):
        prompt = scenarios.get(label)
        if not prompt:
            continue
        records.append(
            _build_record(
                source_type="search_bar",
                origin=str(scenarios_path.relative_to(w5_root)),
                source_path=str(scenarios_path.relative_to(w5_root)),
                source_name=label,
                text=prompt,
            )
        )
    return records


def collect_e2e_case_records(w5_root: str | Path) -> list[CaseRecord]:
    root = Path(w5_root).resolve()
    specs = discover_e2e_specs(root)
    records: list[CaseRecord] = []

    for spec_path in specs:
        records.extend(_extract_case_directory_records(spec_path, root))
        records.extend(_extract_inline_literal_records(spec_path, root))

    records.extend(_extract_search_bar_records(root))
    return dedupe_case_records(records)


def dedupe_case_records(records: list[CaseRecord]) -> list[CaseRecord]:
    deduped: list[CaseRecord] = []
    seen_keys: dict[str, CaseRecord] = {}
    for record in records:
        key = _normalize_for_key(record.text)
        existing = seen_keys.get(key)
        if existing is not None:
            if record.source_type == "terrain_case" and existing.source_type != "terrain_case":
                seen_keys[key] = record
            continue
        seen_keys[key] = record

    for record in sorted(
        seen_keys.values(),
        key=lambda item: (item.source_type, item.source_path, item.source_name, item.case_id),
    ):
        deduped.append(record)
    return deduped


def build_manifest(records: list[CaseRecord]) -> dict[str, Any]:
    by_source_type: dict[str, int] = {}
    by_origin: dict[str, int] = {}
    for record in records:
        by_source_type[record.source_type] = by_source_type.get(record.source_type, 0) + 1
        by_origin[record.origin] = by_origin.get(record.origin, 0) + 1

    return {
        "total_cases": len(records),
        "by_source_type": dict(sorted(by_source_type.items())),
        "by_origin": dict(sorted(by_origin.items())),
    }


def _training_record(record: CaseRecord) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": TRAINING_USER_PROMPT},
            {"role": "assistant", "content": record.text},
        ],
        "metadata": {
            "case_id": record.case_id,
            "source_type": record.source_type,
            "origin": record.origin,
            "source_path": record.source_path,
            "source_name": record.source_name,
        },
    }


def mistral_training_record_from_text(text: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
            {"role": "user", "content": TRAINING_USER_PROMPT},
            {"role": "assistant", "content": _normalize_whitespace(text)},
        ]
    }


def mistral_training_record_from_case(record: CaseRecord) -> dict[str, Any]:
    return mistral_training_record_from_text(record.text)


def write_case_corpus(records: list[CaseRecord], output_dir: str | Path) -> dict[str, Any]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    raw_path = target_dir / "e2e_cases.jsonl"
    train_path = target_dir / "e2e_cases_train.jsonl"
    train_mistral_path = target_dir / "e2e_cases_train_mistral.jsonl"
    manifest_path = target_dir / "manifest.json"
    summary_path = target_dir / "README.md"

    with raw_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

    with train_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_training_record(record), ensure_ascii=False) + "\n")

    with train_mistral_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(mistral_training_record_from_case(record), ensure_ascii=False) + "\n")

    manifest = build_manifest(records)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_lines = [
        "# Corpus E2E succession",
        "",
        f"- total: {manifest['total_cases']}",
    ]
    for source_type, count in manifest["by_source_type"].items():
        summary_lines.append(f"- {source_type}: {count}")
    summary_lines.extend(
        [
            "",
            "Fichiers générés :",
            f"- `{raw_path.name}` : corpus brut des énoncés",
            f"- `{train_path.name}` : JSONL prêt pour un fine-tuning orienté génération d'énoncés",
            f"- `{train_mistral_path.name}` : JSONL strictement compatible Mistral/OpenAI (`messages` uniquement)",
            f"- `{manifest_path.name}` : statistiques de couverture",
        ]
    )
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return manifest
