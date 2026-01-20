# Utility functions and data models for code completion and generation service.
import torch
from pydantic import BaseModel
from typing import Optional, List
import sys
from pathlib import Path


# Add project root to path
project_root = Path.cwd().parent
sys.path.append(str(project_root))

print("Project root added to path:", project_root)


FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"


"""
    This function detects the available device for PyTorch computations.
"""
def detect_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

"""
    Data model for code completion request.
"""
class CompleteRequest(BaseModel):
    language: str
    prefix: str
    suffix: Optional[str] = None
    max_tokens: int = 64
    temperature: float = 0.2
    top_p: float = 0.95
    n: int = 1

"""
    Data model for code generation request.
"""
class GenerateRequest(BaseModel):
    language: str
    comment: str
    surrounding: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95

# Data models for code completion and generation responses.
class Suggestion(BaseModel):
    text: str

# Data models for code completion and generation responses.
class CompleteResponse(BaseModel):
    suggestions: List[Suggestion]

# Data models for code completion and generation responses.
class GenerateResponse(BaseModel):
    code: str


def build_fim_prompt(prefix: str, suffix: str) -> str:
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"



