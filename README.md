# Ministral Fine-Tuning

Scaffold Python pour fine-tuner `mistralai/Ministral-3-3B-Base-2512` avec un adapter LoRA.

For hackathon handoff and ongoing English status notes, see `HACKATHON_STATUS.md`.

Le périmètre de cette première version est volontairement simple :
- fine-tuning text-only
- dataset JSONL
- partie vision gelée
- export d'un adapter PEFT dans `runs/`

## Prérequis

- Python 3.12+
- une machine Linux avec GPU NVIDIA recommandé
- accès Hugging Face au modèle `mistralai/Ministral-3-3B-Base-2512`

## Installation

```bash
cd /root/projects/ministral-fine-tuning
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
cp .env.example .env
```

Remplis ensuite `HF_TOKEN` dans `.env`.

## Format du dataset

Chaque ligne du JSONL peut prendre une des formes suivantes :

```json
{"prompt":"USER: Corrige ce résumé.\nASSISTANT:","response":"Voici une version corrigée."}
{"messages":[{"role":"system","content":"Réponds en français."},{"role":"user","content":"Fais un titre."},{"role":"assistant","content":"Titre proposé"}]}
{"text":"Texte complet déjà prêt à apprendre tel quel."}
```

## Lancer un entraînement

```bash
source .venv/bin/activate
./scripts/train.sh
```

Ou directement :

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
python -m ministral_ft.train \
  --train-file data/examples/train.jsonl \
  --eval-file data/examples/valid.jsonl \
  --output-dir runs/ministral-3b-lora
```

## Paramètres utiles

```bash
python -m ministral_ft.train --help
```

Options importantes :
- `--no-4bit` pour désactiver le chargement quantifié
- `--max-length` pour contrôler la longueur de séquence
- `--per-device-batch-size` et `--gradient-accumulation-steps` pour ajuster la mémoire
- `--resume-from-checkpoint` pour reprendre un run

## Sorties

Après entraînement :
- l'adapter LoRA est sauvegardé dans `runs/...`
- un `training_summary.json` résume le run

## Corpus E2E succession

Le dépôt peut maintenant rapatrier les énoncés utilisés par les tests E2E du projet `../w5`.

Commande :

```bash
./scripts/build_succession_corpus.sh
```

Par défaut, cela génère dans `data/succession_e2e/` :
- `e2e_cases.jsonl` : corpus brut des énoncés E2E
- `e2e_cases_train.jsonl` : JSONL prêt pour fine-tuning orienté génération d'énoncés
- `e2e_cases_train_mistral.jsonl` : JSONL strictement compatible Mistral/OpenAI (`messages` uniquement)
- `manifest.json` : stats de couverture

Le corpus inclut :
- les cas `ENONCE.md` chargés depuis `cases/succession97` et `cases/succession` par les specs E2E
- les prompts inline des specs E2E
- les scénarios `succession-search-bar` réellement utilisés en E2E

## Serveur de consignes

Un serveur HTTP local peut distribuer des consignes de génération de cas, en équilibrant plusieurs dimensions :
- persona
- voix narrative
- forme du message
- niveau de bruit (fautes, abréviations, ambiguïtés)
- densité chiffrée
- complexité, y compris `hard negatives`
- thème juridique principal et secondaire

Commande :

```bash
./scripts/run_case_instruction_server.sh
```

Par défaut :
- hôte : `127.0.0.1`
- port : `8765`
- état persistant : `data/case_instruction_server/`
- schéma maître : `../w5/glinerExtract/schema/schema.full.json`
- cible totale : `5000` cas d'entraînement
- objectif de génération : calculé automatiquement (`cible totale - corpus seed`)

Endpoints :
- `GET /health`
- `GET /dashboard`
- `GET /next-instruction`
- `POST /next-instruction`
- `POST /submit-case`

Exemple :

```bash
curl -s http://127.0.0.1:8765/next-instruction
```

Ou :

```bash
curl -s \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":"agent-01"}' \
  http://127.0.0.1:8765/next-instruction
```

Soumission d'un cas généré :

```bash
curl -s \
  -H 'Content-Type: application/json' \
  -d '{"instruction_id":"INS-0001","agent_id":"agent-01","case_text":"..."}' \
  http://127.0.0.1:8765/submit-case
```

Mode TOON-first (recommandé) :

```bash
curl -s \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":"agent-01"}' \
  http://127.0.0.1:8765/next-instruction
```

Le serveur renvoie une instruction minimaliste contenant :
- `instruction_id`
- `target_toon` (TOON sparse schema-driven)
- `prompt` (consigne "TOON -> énoncé")

L'agent soumet uniquement `case_text` + `instruction_id` :

```bash
curl -s \
  -H 'Content-Type: application/json' \
  -d '{"instruction_id":"INS-0001","agent_id":"agent-01","case_text":"..."}' \
  http://127.0.0.1:8765/submit-case
```

Le `target_toon` interne est généré en mode schema-driven :
- validation stricte des chemins/types/enums contre `schema.full.json`
- sortie sparse stricte (pas de `null`, pas d'objet/liste vide)
- contraintes d'intégrité métier minimales (cohérence statut/lien, présence d'éléments attendus selon le thème)
- alignement strict topic <-> contenu du TOON (si le sujet est `assurance_vie`, le TOON contient bien une assurance-vie, etc.)

À chaque émission ou soumission, le serveur met à jour :
- `issued_instructions.jsonl`
- `generated_cases.jsonl`
- `generated_cases_train_mistral.jsonl`
- `full_training_cases_mistral.jsonl`
- `summary.json`
- `summary.md`
- un fichier par instruction dans `instructions/`
- un fichier par soumission dans `submissions/`

Note : `data/case_instruction_server/` est un dossier de runtime (état et exports) et est gitignoré.

## Notes

- Le script fine-tune les couches langage classiques (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- La partie vision est explicitement gelée dans ce scaffold.
- Si tu veux une vraie pipeline multimodale ensuite, il faudra ajouter un préprocessing image et un format de batch différent.
