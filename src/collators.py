"""
Simplified data collators for mixed-task training.

Two specialized collators:
- FIMCollator: For fill-in-the-middle autocomplete
- Comment2CodeCollator: For comment-to-code generation (masks prompt tokens)

Plus a dispatcher MixedTaskCollator that routes to the appropriate one.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch


FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


def format_fim_text(prefix: str, suffix: str, middle: str) -> str:
    """Format FIM text with special tokens."""
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"


def _build_fim_text(ex: Dict[str, Any]) -> str:
    """Build FIM text from structured fields or fallback to text field."""
    txt = ex.get("text")
    if txt:
        return txt
    return format_fim_text(
        ex.get("prefix", ""),
        ex.get("suffix", ""),
        ex.get("middle", "")
    )


@dataclass
class FIMCollator:
    """Collator for FIM (Fill-in-the-Middle) autocomplete tasks.
    
    Standard causal LM training: all tokens are supervised (labels = input_ids).
    """
    tokenizer: Any
    max_length: int = 1024
    pad_to_multiple_of: Optional[int] = 8

    def __post_init__(self):
        self.pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []

        for ex in batch:
            text = _build_fim_text(ex)
            
            if not text.strip():
                # Empty sample - use fallback with masked loss
                ids = [self.pad_id]
                labels = [-100]
            else:
                ids = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.max_length,
                )["input_ids"]
                
                if len(ids) == 0:
                    ids = [self.pad_id]
                    labels = [-100]
                else:
                    # Standard causal LM: predict all tokens
                    labels = ids[:]

            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        attention_mask = (input_ids != self.pad_id).long()

        # Optional: pad to multiple
        if self.pad_to_multiple_of:
            seq_len = input_ids.shape[1]
            target = ((seq_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
            if target != seq_len:
                pad_len = target - seq_len
                input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=self.pad_id)
                labels = torch.nn.functional.pad(labels, (0, pad_len), value=-100)
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class Comment2CodeCollator:
    """Collator for comment-to-code tasks.
    
    Masks prompt tokens (labels = -100), only supervises completion tokens.
    This teaches the model to generate code from comments/docstrings.
    """
    tokenizer: Any
    max_length: int = 1024
    pad_to_multiple_of: Optional[int] = 8
    reserve_completion_tokens: int = 64
    min_supervised_tokens: int = 8

    def __post_init__(self):
        self.pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids_list = []
        labels_list = []

        for ex in batch:
            prompt = (ex.get("prompt") or "").strip()
            completion = (ex.get("completion") or "").strip()

            # Validate both prompt and completion exist
            if not prompt or not completion:
                ids = [self.pad_id]
                labels = [-100]
            else:
                # Tokenize completion first (reserve space)
                comp_max = min(self.reserve_completion_tokens, self.max_length)
                comp_ids = self.tokenizer(
                    completion,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=comp_max,
                )["input_ids"]

                # Check minimum supervised tokens
                if len(comp_ids) < self.min_supervised_tokens:
                    ids = [self.pad_id]
                    labels = [-100]
                else:
                    # Tokenize prompt with remaining space
                    prompt_max = max(self.max_length - len(comp_ids), 0)
                    prompt_ids = self.tokenizer(
                        prompt,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=prompt_max,
                    )["input_ids"]

                    # Concatenate: prompt (masked) + completion (supervised)
                    ids = prompt_ids + comp_ids
                    labels = ([-100] * len(prompt_ids)) + comp_ids

            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))

        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=self.pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        attention_mask = (input_ids != self.pad_id).long()

        # Optional: pad to multiple
        if self.pad_to_multiple_of:
            seq_len = input_ids.shape[1]
            target = ((seq_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
            if target != seq_len:
                pad_len = target - seq_len
                input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=self.pad_id)
                labels = torch.nn.functional.pad(labels, (0, pad_len), value=-100)
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class MixedTaskCollator:
    """Dispatcher collator that routes to task-specific collators.
    
    Automatically detects task type and uses the appropriate collator:
    - autocomplete_fim -> FIMCollator
    - comment2code -> Comment2CodeCollator
    
    For homogeneous batches (all same task), uses specialized collator directly.
    For mixed batches, processes each sample individually and combines.
    """
    tokenizer: Any
    max_length: int = 1024
    pad_to_multiple_of: Optional[int] = 8
    reserve_completion_tokens: int = 64
    min_supervised_tokens: int = 8

    def __post_init__(self):
        self.fim_collator = FIMCollator(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        self.c2c_collator = Comment2CodeCollator(
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            reserve_completion_tokens=self.reserve_completion_tokens,
            min_supervised_tokens=self.min_supervised_tokens,
        )

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate batch by task
        fim_batch = [ex for ex in batch if ex.get("task") == "autocomplete_fim"]
        c2c_batch = [ex for ex in batch if ex.get("task") == "comment2code"]
        
        # If batch is homogeneous, use specialized collator (faster)
        if len(fim_batch) == len(batch):
            return self.fim_collator(batch)
        if len(c2c_batch) == len(batch):
            return self.c2c_collator(batch)
        
        # Mixed batch: process separately and combine
        # (This is rare if you batch by task, but handles edge cases)
        all_input_ids = []
        all_labels = []
        
        for ex in batch:
            if ex.get("task") == "comment2code":
                result = self.c2c_collator([ex])
            else:
                result = self.fim_collator([ex])
            
            all_input_ids.append(result["input_ids"][0])
            all_labels.append(result["labels"][0])
        
        # Pad combined results
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            all_labels, batch_first=True, padding_value=-100
        )
        attention_mask = (input_ids != pad_id).long()
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
