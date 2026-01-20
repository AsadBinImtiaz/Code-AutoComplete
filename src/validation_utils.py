"""
Validation utilities for checking sample quality before training/evaluation.
"""

from typing import Dict, Any


FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


def is_valid_sample(ex: Dict[str, Any]) -> bool:
    """Basic sanity check: sample has enough content to be meaningful.
    
    Args:
        ex: Sample dictionary with task, language, and content fields
        
    Returns:
        True if sample has valid content, False otherwise
    """
    task = ex.get("task")
    
    if task == "autocomplete_fim":
        # FIM needs at least one of: text, prefix, suffix, or middle
        return bool(
            (ex.get("text") or "").strip()
            or (ex.get("prefix") or "").strip()
            or (ex.get("suffix") or "").strip()
            or (ex.get("middle") or "").strip()
        )
    
    if task == "comment2code":
        # Comment2code needs BOTH prompt AND completion
        # Using 'and' is correct - both must exist
        prompt = (ex.get("prompt") or "").strip()
        completion = (ex.get("completion") or "").strip()
        return bool(prompt and completion)
    
    # Fallback for unknown tasks
    return bool((ex.get("text") or "").strip())


def is_loss_valid_for_eval(
    ex: Dict[str, Any],
    tokenizer,
    max_length: int,
    min_completion_tokens: int = 8,
) -> bool:
    """Stricter check for evaluation: ensures sample will produce valid loss.
    
    For comment2code, ensures completion tokens will fit (supervised tokens exist).
    For FIM/plain LM, ensures tokenization yields non-empty ids.
    
    Args:
        ex: Sample dictionary
        tokenizer: Tokenizer to use for checking token counts
        max_length: Maximum sequence length
        min_completion_tokens: Minimum supervised tokens required for comment2code
        
    Returns:
        True if sample will produce valid loss, False otherwise
    """
    task = ex.get("task")

    if task == "autocomplete_fim":
        # Try to get text
        text = (ex.get("text") or "").strip()
        if not text:
            # Try reconstructing from parts
            prefix = ex.get("prefix", "")
            suffix = ex.get("suffix", "")
            middle = ex.get("middle", "")
            text = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}".strip()
        
        if not text:
            return False
        
        # Check tokenization produces non-empty result
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        return len(ids) > 0

    if task == "comment2code":
        prompt = (ex.get("prompt") or "").strip()
        completion = (ex.get("completion") or "").strip()
        
        # Both must exist
        if not prompt or not completion:
            return False

        # Check prompt doesn't consume entire window
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(prompt_ids) >= max_length:
            return False

        # Check completion has tokens
        comp_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        if len(comp_ids) == 0:
            return False

        # Check enough space remains for minimum supervised tokens
        remaining = max_length - len(prompt_ids)
        return remaining >= min_completion_tokens

    # Fallback for unknown tasks
    text = (ex.get("text") or "").strip()
    if not text:
        return False
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return len(ids) > 0
