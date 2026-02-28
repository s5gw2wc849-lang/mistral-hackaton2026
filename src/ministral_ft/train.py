from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from ministral_ft.data import JsonlSupervisedDataset, SupervisedDataCollator


DEFAULT_MODEL_ID = "mistralai/Ministral-3-3B-Base-2512"
DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass(slots=True)
class TrainConfig:
    model_id: str
    train_file: str
    eval_file: str | None
    output_dir: str
    max_length: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    num_epochs: float
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    seed: int
    load_in_4bit: bool
    gradient_checkpointing: bool
    resume_from_checkpoint: str | None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="Fine-tune Ministral 3 3B Base with a LoRA adapter."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--disable-gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-from-checkpoint", default=None)

    args = parser.parse_args()
    return TrainConfig(
        model_id=args.model_id,
        train_file=args.train_file,
        eval_file=args.eval_file,
        output_dir=args.output_dir,
        max_length=args.max_length,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        load_in_4bit=not args.no_4bit,
        gradient_checkpointing=not args.disable_gradient_checkpointing,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )


def _freeze_vision_parameters(model: Any) -> int:
    frozen = 0
    for name, param in model.named_parameters():
        lower_name = name.lower()
        if "vision" in lower_name or "image" in lower_name:
            param.requires_grad = False
            frozen += param.numel()
    return frozen


def _detect_precision(torch_module: Any) -> tuple[Any, bool, bool]:
    if torch_module.cuda.is_available():
        if torch_module.cuda.is_bf16_supported():
            return torch_module.bfloat16, True, False
        return torch_module.float16, False, True
    return torch_module.float32, False, False


def _make_model_and_tokenizer(config: TrainConfig) -> tuple[Any, Any, bool, bool]:
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Mistral3ForConditionalGeneration, MistralCommonBackend

    hf_token = _load_hf_token()
    dtype, use_bf16, use_fp16 = _detect_precision(torch)
    if config.load_in_4bit and not torch.cuda.is_available():
        raise SystemExit("4-bit quantization requires CUDA. Re-run with --no-4bit.")

    common_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    if hf_token:
        common_kwargs["token"] = hf_token

    if torch.cuda.is_available():
        common_kwargs["device_map"] = "auto"
    common_kwargs["torch_dtype"] = dtype

    if config.load_in_4bit:
        common_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model = Mistral3ForConditionalGeneration.from_pretrained(
        config.model_id,
        **common_kwargs,
    )

    tokenizer_kwargs: dict[str, Any] = {}
    if hf_token:
        tokenizer_kwargs["token"] = hf_token
    tokenizer = MistralCommonBackend.from_pretrained(
        config.model_id,
        mode="finetuning",
        **tokenizer_kwargs,
    )

    model.config.use_cache = False
    _freeze_vision_parameters(model)

    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        enable_inputs = getattr(model, "enable_input_require_grads", None)
        if callable(enable_inputs):
            enable_inputs()

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=DEFAULT_TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer, use_bf16, use_fp16


def _load_hf_token() -> str | None:
    import os

    token = os.getenv("HF_TOKEN", "").strip()
    return token or None


def _build_trainer(
    config: TrainConfig,
    model: Any,
    tokenizer: Any,
    use_bf16: bool,
    use_fp16: bool,
) -> Any:
    from transformers import Trainer, TrainingArguments, set_seed

    set_seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = JsonlSupervisedDataset(
        path=config.train_file,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    eval_dataset = None
    if config.eval_file:
        eval_path = Path(config.eval_file)
        if eval_path.exists():
            eval_dataset = JsonlSupervisedDataset(
                path=eval_path,
                tokenizer=tokenizer,
                max_length=config.max_length,
            )

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset is not None else "no",
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=config.gradient_checkpointing,
        optim="adamw_torch",
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=SupervisedDataCollator(tokenizer),
    )


def _write_summary(config: TrainConfig, output_dir: Path) -> None:
    summary = {
        "model_id": config.model_id,
        "train_file": config.train_file,
        "eval_file": config.eval_file,
        "output_dir": config.output_dir,
        "config": asdict(config),
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    load_dotenv()
    config = parse_args()
    model, tokenizer, use_bf16, use_fp16 = _make_model_and_tokenizer(config)
    trainer = _build_trainer(config, model, tokenizer, use_bf16=use_bf16, use_fp16=use_fp16)

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model(config.output_dir)

    save_pretrained = getattr(tokenizer, "save_pretrained", None)
    if callable(save_pretrained):
        save_pretrained(config.output_dir)

    _write_summary(config, Path(config.output_dir))


if __name__ == "__main__":
    main()
