"""
Evaluation utilities shared across notebooks.
Provides consistent evaluation for model testing and training.
"""
import ast
import time
import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EvaluationPrompt:
    """Standard evaluation prompt."""
    name: str
    prompt: str
    language: str = "python"
    expected_keywords: List[str] = None


# ============================================================================
# Standard Evaluation Prompts
# ============================================================================

GENERIC_EVAL_PROMPTS = [
    EvaluationPrompt(
        name="Factorial Function",
        prompt="""# Write Python code for the following docstring
\"\"\"
Calculate the factorial of a number using recursion.
Args:
    n: A non-negative integer
Returns:
    The factorial of n
\"\"\"

# Code:
""",
        language="python",
        expected_keywords=["def", "factorial", "if", "return", "n"]
    ),
    EvaluationPrompt(
        name="List Filter",
        prompt="""# Write Python code for the following docstring
\"\"\"
Filter a list to keep only even numbers.
Args:
    numbers: A list of integers
Returns:
    A list containing only even numbers
\"\"\"

# Code:
""",
        language="python",
        expected_keywords=["def", "filter", "even", "%", "2"]
    ),
    EvaluationPrompt(
        name="String Reversal",
        prompt="""# Write Python code for the following docstring
\"\"\"
Reverse a string.
Args:
    text: The input string
Returns:
    The reversed string
\"\"\"

# Code:
""",
        language="python",
        expected_keywords=["def", "reverse", "return", "[::-1]"]
    ),
]


CDK_EVAL_PROMPTS = [
    EvaluationPrompt(
        name="S3 Bucket with Encryption",
        prompt="""# Write Python code for the following docstring
\"\"\"
Create an S3 bucket with:
- Encryption enabled (KMS)
- Versioning enabled
- Block public access
\"\"\"

# Code:
""",
        language="python",
        expected_keywords=["s3", "Bucket", "encryption", "versioning", "block_public_access"]
    ),
    EvaluationPrompt(
        name="Lambda Function",
        prompt="""# Write Python code for the following docstring
\"\"\"
Create a Lambda function with:
- Python 3.11 runtime
- Proper IAM role
- Environment variables
\"\"\"

# Code:
""",
        language="python",
        expected_keywords=["lambda", "Function", "Role", "runtime", "environment"]
    ),
]


# ============================================================================
# Code Generation
# ============================================================================

def generate_code(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.2,
    top_p: float = 0.9,
    device: str = "cpu"
) -> Tuple[str, float]:
    """
    Generate code completion from prompt.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        device: Device to run on
        
    Returns:
        Tuple of (generated_text, latency_ms)
    """
    start_time = time.time()
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    
    # Decode only the generated part
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text, latency_ms


# ============================================================================
# Syntax Validation
# ============================================================================

def validate_python_syntax(code: str) -> Tuple[bool, str]:
    """
    Validate Python code syntax.
    
    Args:
        code: Python code string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, "Valid"
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def validate_typescript_syntax(code: str) -> Tuple[bool, str]:
    """
    Basic TypeScript/JavaScript syntax validation.
    
    Args:
        code: TypeScript/JavaScript code string
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Basic checks for common syntax errors
    if code.count("{") != code.count("}"):
        return False, "Mismatched braces"
    if code.count("(") != code.count(")"):
        return False, "Mismatched parentheses"
    if code.count("[") != code.count("]"):
        return False, "Mismatched brackets"
    
    return True, "Valid (basic check)"


def validate_code_syntax(code: str, language: str) -> Tuple[bool, str]:
    """
    Validate code syntax based on language.
    
    Args:
        code: Code string
        language: Programming language
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    lang = language.lower().strip()
    
    if lang in ["python", "py"]:
        return validate_python_syntax(code)
    elif lang in ["typescript", "ts", "javascript", "js"]:
        return validate_typescript_syntax(code)
    else:
        return True, "Unknown language - skipped validation"


# ============================================================================
# Batch Evaluation
# ============================================================================

def evaluate_model_on_prompts(
    model,
    tokenizer,
    prompts: List[EvaluationPrompt],
    device: str = "cpu",
    max_tokens: int = 100
) -> List[Dict[str, Any]]:
    """
    Evaluate model on a list of prompts.
    
    Args:
        model: Language model
        tokenizer: Model tokenizer
        prompts: List of evaluation prompts
        device: Device to run on
        max_tokens: Maximum tokens to generate
        
    Returns:
        List of evaluation results
    """
    results = []
    
    for eval_prompt in prompts:
        generated, latency = generate_code(
            model=model,
            tokenizer=tokenizer,
            prompt=eval_prompt.prompt,
            max_tokens=max_tokens,
            device=device
        )
        
        is_valid, error_msg = validate_code_syntax(generated, eval_prompt.language)
        
        # Check for expected keywords
        keywords_found = []
        if eval_prompt.expected_keywords:
            keywords_found = [
                kw for kw in eval_prompt.expected_keywords
                if kw.lower() in generated.lower()
            ]
        
        results.append({
            "name": eval_prompt.name,
            "prompt": eval_prompt.prompt,
            "generated": generated,
            "latency_ms": latency,
            "syntax_valid": is_valid,
            "syntax_error": error_msg if not is_valid else None,
            "keywords_found": keywords_found,
            "keywords_expected": eval_prompt.expected_keywords or [],
            "keyword_match_rate": (
                len(keywords_found) / len(eval_prompt.expected_keywords)
                if eval_prompt.expected_keywords else 0
            )
        })
    
    return results


def summarize_evaluation_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarize evaluation results.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Summary statistics
    """
    if not results:
        return {}
    
    valid_count = sum(1 for r in results if r["syntax_valid"])
    total_count = len(results)
    
    latencies = [r["latency_ms"] for r in results]
    keyword_rates = [r["keyword_match_rate"] for r in results]
    
    return {
        "total_samples": total_count,
        "syntax_valid_count": valid_count,
        "syntax_validity_percent": 100 * valid_count / total_count,
        "mean_latency_ms": sum(latencies) / len(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
        "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
        "mean_keyword_match_rate": sum(keyword_rates) / len(keyword_rates),
    }


def print_evaluation_summary(results: List[Dict[str, Any]]):
    """Print formatted evaluation summary."""
    summary = summarize_evaluation_results(results)
    
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total samples: {summary['total_samples']}")
    print(f"Syntax valid: {summary['syntax_valid_count']}/{summary['total_samples']} "
          f"({summary['syntax_validity_percent']:.1f}%)")
    print(f"\nLatency:")
    print(f"  Mean: {summary['mean_latency_ms']:.1f}ms")
    print(f"  P50:  {summary['p50_latency_ms']:.1f}ms")
    print(f"  P95:  {summary['p95_latency_ms']:.1f}ms")
    print(f"\nKeyword match rate: {summary['mean_keyword_match_rate']:.1%}")
    print("=" * 60)
    
    print("\nDetailed Results:")
    for r in results:
        status = "✓" if r["syntax_valid"] else "✗"
        print(f"{status} {r['name']}: {r['latency_ms']:.0f}ms, "
              f"keywords: {len(r['keywords_found'])}/{len(r['keywords_expected'])}")
