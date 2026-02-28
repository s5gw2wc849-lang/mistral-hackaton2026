from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _render_messages(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = _normalize_text(message.get("role") or "user").upper()
        content = _normalize_text(message.get("content"))
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines).strip()


def _record_to_training_text(record: dict[str, Any]) -> tuple[str | None, str | None]:
    prompt = _normalize_text(record.get("prompt"))
    response = _normalize_text(record.get("response"))
    if prompt and response:
        prompt_prefix = prompt if prompt.endswith((" ", "\n")) else f"{prompt}\n"
        return prompt_prefix, response

    if "text" in record:
        text = _normalize_text(record.get("text"))
        if text:
            return None, text

    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        typed_messages = [item for item in messages if isinstance(item, dict)]
        last_assistant_index = -1
        for index in range(len(typed_messages) - 1, -1, -1):
            role = _normalize_text(typed_messages[index].get("role")).lower()
            if role == "assistant":
                last_assistant_index = index
                break

        if last_assistant_index >= 0:
            prompt_messages = typed_messages[:last_assistant_index]
            response_text = _normalize_text(typed_messages[last_assistant_index].get("content"))
            prompt_text = _render_messages(prompt_messages)
            prefix = "ASSISTANT:"
            if prompt_text:
                prefix = f"{prompt_text}\nASSISTANT:"
            prompt_prefix = prefix if prefix.endswith((" ", "\n")) else f"{prefix} "
            if response_text:
                return prompt_prefix, response_text

        rendered = _render_messages(typed_messages)
        if rendered:
            return None, rendered

    raise ValueError(
        "Unsupported JSONL record. Expected {prompt,response}, {messages}, or {text}."
    )


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected a JSON object on line {line_number} of {path}")
            records.append(payload)
    if not records:
        raise ValueError(f"No usable records found in {path}")
    return records


def _resolve_pad_token_id(tokenizer: Any) -> int:
    for attribute in ("pad_token_id", "eos_token_id"):
        candidate = getattr(tokenizer, attribute, None)
        if candidate is not None:
            return int(candidate)
    return 0


@dataclass(slots=True)
class EncodedExample:
    input_ids: torch.Tensor
    labels: torch.Tensor


class JsonlSupervisedDataset(Dataset[EncodedExample]):
    def __init__(self, path: str | Path, tokenizer: Any, max_length: int) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records = load_jsonl_records(self.path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> EncodedExample:
        record = self.records[index]
        prompt_text, target_text = _record_to_training_text(record)
        if prompt_text is None:
            input_ids = self.tokenizer.encode(target_text, return_tensors="pt")[0].to(torch.long)
            labels = input_ids.clone()
            return self._trim(input_ids, labels)

        prompt_ids = self.tokenizer.encode(prompt_text, return_tensors="pt")[0].to(torch.long)
        response_ids = self.tokenizer.encode(target_text, return_tensors="pt")[0].to(torch.long)
        input_ids = torch.cat((prompt_ids, response_ids), dim=0)
        prompt_mask = torch.full_like(prompt_ids, fill_value=-100)
        labels = torch.cat((prompt_mask, response_ids.clone()), dim=0)
        return self._trim(input_ids, labels)

    def _trim(self, input_ids: torch.Tensor, labels: torch.Tensor) -> EncodedExample:
        if input_ids.shape[0] > self.max_length:
            overflow = input_ids.shape[0] - self.max_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]
        return EncodedExample(input_ids=input_ids, labels=labels)


class SupervisedDataCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.pad_token_id = _resolve_pad_token_id(tokenizer)

    def __call__(self, features: list[EncodedExample]) -> dict[str, torch.Tensor]:
        input_ids = [item.input_ids for item in features]
        labels = [item.labels for item in features]
        padded_input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = padded_input_ids.ne(self.pad_token_id).to(torch.long)
        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "labels": padded_labels,
        }
