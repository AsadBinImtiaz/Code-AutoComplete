"""
Training utilities for iterative training and data sampling.
"""
import random
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import Dataset


def sample_dataset(ds: Dataset, n: int, seed: int = 42) -> Dataset:
    """
    Sample n examples from a dataset.
    
    Args:
        ds: HuggingFace Dataset
        n: Number of samples
        seed: Random seed
        
    Returns:
        Sampled dataset
    """
    if n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def create_iterative_training_splits(
    dataset: List[Dict],
    iterations: int = 3,
    samples_per_iteration: int = 5000,
    seed: int = 42
) -> List[List[Dict]]:
    """
    Create multiple random splits for iterative training.
    
    Args:
        dataset: Full dataset
        iterations: Number of training iterations
        samples_per_iteration: Samples per iteration
        seed: Random seed
        
    Returns:
        List of dataset splits
    """
    rng = random.Random(seed)
    splits = []
    
    for i in range(iterations):
        # Use different seed for each iteration
        iter_seed = seed + i
        rng.seed(iter_seed)
        
        # Sample with replacement to allow overlaps
        if samples_per_iteration >= len(dataset):
            split = dataset.copy()
        else:
            split = rng.choices(dataset, k=samples_per_iteration)
        
        rng.shuffle(split)
        splits.append(split)
    
    return splits


def save_jsonl(data: List[Dict], file_path: Path):
    """Save data to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def tokenize_with_masked_labels(
    examples: List[Dict],
    tokenizer,
    max_length: int = 512
) -> Dict:
    """
    Tokenize examples with label masking for comment-to-code.
    
    Args:
        examples: List of dicts with 'text', 'task', optionally 'prefix_length'
        tokenizer: Model tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Dict with 'input_ids', 'attention_mask', 'labels'
    """
    texts = [ex["text"] for ex in examples]
    tasks = [ex.get("task", "autocomplete_fim") for ex in examples]
    prefix_lengths = [ex.get("prefix_length", 0) for ex in examples]
    
    # Tokenize all texts
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt"
    )
    
    # Create labels with masking for comment-to-code
    labels = []
    for i, (ids, task, prefix_len) in enumerate(zip(encoded["input_ids"], tasks, prefix_lengths)):
        if task == "comment2code" and prefix_len > 0:
            # Mask the prefix (instruction + docstring)
            prefix_ids = tokenizer(
                texts[i][:prefix_len],
                truncation=True,
                max_length=max_length,
                padding=False
            )["input_ids"]
            
            lab = [-100] * len(ids)
            start = min(len(prefix_ids), len(ids))
            for j in range(start, len(ids)):
                lab[j] = ids[j].item()
            labels.append(lab)
        else:
            # No masking for FIM
            labels.append(ids.tolist())
    
    encoded["labels"] = labels
    return encoded


def collate_training_batch(features: List[Dict]) -> Dict:
    """
    Collate function for training batches.
    
    Args:
        features: List of feature dicts
        
    Returns:
        Batched tensors
    """
    import torch
    return {
        k: torch.tensor([f[k] for f in features], dtype=torch.long)
        for k in features[0].keys()
    }

def get_latest_checkpoint(output_dir: Path) -> Optional[str]:
    """Return newest checkpoint path in output_dir, else None."""
    if not output_dir.exists():
        return None
    checkpoints = [p for p in output_dir.glob('checkpoint-*') if p.is_dir()]
    if not checkpoints:
        return None
    def _step(p: Path) -> int:
        try:
            return int(p.name.split('-')[-1])
        except Exception:
            return -1
    checkpoints.sort(key=_step)
    return str(checkpoints[-1])

def save_iteration_indices(out_dir: Path, indices: List[int]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'sample_indices.json', 'w', encoding='utf-8') as f:
        json.dump(indices, f)

def load_iteration_indices(out_dir: Path) -> Optional[List[int]]:
    p = out_dir / 'sample_indices.json'
    if not p.exists():
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)
