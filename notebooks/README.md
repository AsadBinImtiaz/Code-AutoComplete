# Code Autocomplete Assistant - Notebooks

This directory contains Jupyter notebooks for training and evaluating the Code Autocomplete Assistant.

## Shared Python Modules

All notebooks use shared modules in `src/` for consistency:

- **`src/data_processing.py`** - FIM generation, comment-to-code formatting, data cleaning
- **`src/evaluation.py`** - Standard evaluation prompts, code generation, syntax validation
- **`src/training_utils.py`** - Dataset sampling, iterative training, tokenization

## Notebook Sequence

### Data Preparation (Unified)

1. **01_dataset_preparation.ipynb** - Unified Data Preparation
   - Downloads CodeSearchNet (Python + JavaScript)
   - Clones AWS CDK repositories (aws-cdk-examples)
   - Extracts code examples from both sources
   - **Generates FIM samples** (80% of data) using `src.data_processing`
   - **Generates comment-to-code samples** (20% of data)
   - Creates training/validation splits
   - **Output**: 
     - `datasets/unified/train.jsonl` - Mixed FIM + comment-to-code
     - `datasets/unified/validation.jsonl`
     - `datasets/unified/cdk_train.jsonl` - CDK-specific (for Stage B)
     - `datasets/unified/cdk_validation.jsonl`

### Model Testing & Training

2. **03_model_inference_exploration.ipynb** - Base Model Testing
   - Loads base model (Qwen/Qwen2.5-Coder-7B or similar)
   - Tests FIM autocomplete using `src.data_processing.build_autocomplete_prompt()`
   - Evaluates using standard prompts from `src.evaluation`
   - Benchmarks latency
   - Saves baseline (M0) results
   - **Purpose**: Establish baseline performance for comparison

3. **04_training_stage_a.ipynb** - Stage A Training (FIM + Comment-to-Code)
   - Loads preprocessed FIM data from `datasets/unified/`
   - Uses `src.training_utils` for dataset sampling and iterative training
   - Trains on mixed objectives:
     - **FIM (Fill-in-the-Middle)**: 80% of training data
       - Format: `<|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>middle`
       - Simulates IDE autocomplete scenarios
     - **Comment-to-Code**: 20% of training data
       - Format: Instruction + docstring → code
       - Label masking: Only train on code generation part
   - Configures LoRA for efficient fine-tuning
   - Supports iterative training on random subsets
   - Evaluates using `src.evaluation.evaluate_model_on_prompts()`
   - Saves Stage A adapter
   - **Output**: `models/stage_a_adapter/`

### Stage B: CDK Domain Adaptation

4. **05_training_stage_b.ipynb** - Stage B Training (CDK Specialization)
   - Loads Stage A model as starting point
   - Trains on CDK dataset with same FIM + comment-to-code mix
   - Optionally mixes in 10-30% Stage A data to prevent forgetting
   - Saves Stage B adapter
   - **Output**: `models/stage_b_adapter/`

### Evaluation & Integration

6. **06_model_comparison_evaluation.ipynb** - Model Comparison
   - Compares M0 (base) vs M1 (Stage A) vs M2 (Stage B)
   - Tests on generic code generation
   - Tests on CDK-specific tasks
   - Measures syntax validity, latency, quality
   - **Output**: `evaluation/stage_comparison_report.json`

7. **07_inference_server_testing.ipynb** - API Server Testing
   - Tests FastAPI server endpoints
   - Validates FIM autocomplete API
   - Tests comment-to-code generation API
   - Measures API latency and throughput
   - **Output**: `evaluation/api_test_report.json`

8. **08_vscode_integration_demo.ipynb** - VS Code Integration
   - Configures Continue extension
   - Demonstrates autocomplete in real files
   - Shows CDK code generation
   - Creates demo scenarios
   - **Output**: Demo files and configuration

## Training Data Format

### Unified Format (from notebook 01)
All data is preprocessed in notebook 01 and saved in unified format:

```jsonl
{"text": "<|fim_prefix|>code_prefix<|fim_suffix|>code_suffix<|fim_middle|>code_middle", "task": "autocomplete_fim", "language": "python"}
{"text": "# Write Python code...\n\"\"\"\ndocstring\n\"\"\"\n\ncode", "task": "comment2code", "language": "python", "prefix_length": 50}
```

### FIM Training Format (80% of data)
```
<|fim_prefix|>def calculate_sum(a, b):
    """Add two numbers."""
<|fim_suffix|>
    return result
<|fim_middle|>    result = a + b
```

### Comment-to-Code Training Format (20% of data)
```
# Write Python code for the following docstring
"""
Calculate the factorial of a number.
Args:
    n: A non-negative integer
Returns:
    The factorial of n
"""

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
```

Note: The `prefix_length` field is used for label masking during training - only the code part (after the prefix) is trained on.

## Key Design Decisions

1. **Unified Data Preparation**: All data preprocessing (FIM generation, comment-to-code formatting) happens in notebook 01, not during training
2. **FIM Training**: 80% of training focuses on autocomplete (FIM) to match IDE usage patterns
3. **Comment-to-Code**: 20% trains on generating code from docstrings/comments
4. **Label Masking**: Comment-to-code training masks the prompt part (labels = -100) to only train on code generation
5. **Mixed Objectives**: Both FIM and comment-to-code in same training run for versatility
6. **Two-Stage Training**: Stage A (general) → Stage B (CDK domain adaptation)
7. **Shared Modules**: All notebooks use `src/` modules for consistency
8. **Iterative Training**: Notebook 04 supports training on random subsets multiple times

## Running the Notebooks

### Prerequisites
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure you have enough disk space
# - CodeSearchNet: ~20GB
# - AWS CDK repos: ~5GB  
# - Model weights: ~15GB (7B model)
```

### Execution Order
1. Run notebook 01 to prepare all data (CodeSearchNet + CDK, with FIM generation)
2. Run notebook 03 to test base model and establish baseline
3. Run notebook 04 to train Stage A (takes several hours)
4. Run notebook 05 to train Stage B (takes several hours)
5. Run notebooks 06, 07, 08 for evaluation and demo

### Hardware Requirements
- **Minimum**: 16GB RAM, Apple Silicon M1/M2 or NVIDIA GPU with 8GB VRAM
- **Recommended**: 32GB RAM, Apple Silicon M2 Pro/Max or NVIDIA GPU with 16GB+ VRAM
- **Training Time**: 
  - Stage A: 4-8 hours (depending on hardware)
  - Stage B: 2-4 hours

## Outputs

### Models
- `models/stage_a_adapter/` - LoRA adapter for Stage A
- `models/stage_b_adapter/` - LoRA adapter for Stage B

### Datasets
- `datasets/unified/train.jsonl` - Mixed FIM + comment-to-code training data
- `datasets/unified/validation.jsonl` - Mixed validation data
- `datasets/unified/cdk_train.jsonl` - CDK-specific training data
- `datasets/unified/cdk_validation.jsonl` - CDK validation data

### Evaluation
- `evaluation/baseline_m0_results.json` - Base model (M0) results
- `evaluation/stage_comparison_report.json` - Model comparison results
- `evaluation/api_test_report.json` - API performance metrics
- `models/stage_a_adapter/training_curves.png` - Training visualizations
- `models/stage_b_adapter/training_curves.png` - Training visualizations

## Notes

- Notebook 01 handles all data preparation including FIM generation
- Notebooks 03 and 04 use shared evaluation module for consistency
- Notebook 04 loads preprocessed FIM data, doesn't generate it during training
- This allows flexibility to experiment with different training formats without re-downloading data
- Iterative training in notebook 04 enables training on random subsets multiple times
