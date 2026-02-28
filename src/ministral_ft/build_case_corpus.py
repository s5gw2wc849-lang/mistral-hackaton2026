from __future__ import annotations

import argparse
import json
from pathlib import Path

from ministral_ft.e2e_case_corpus import collect_e2e_case_records, write_case_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rapatrie les énoncés de succession utilisés par les tests E2E de ../w5."
    )
    parser.add_argument("--w5-root", default="../w5")
    parser.add_argument("--output-dir", default="data/succession_e2e")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = collect_e2e_case_records(Path(args.w5_root))
    manifest = write_case_corpus(records, Path(args.output_dir))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
