import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from fastapi import FastAPI


# Add project root to path
project_root = Path.cwd().parent
sys.path.append(str(project_root))

from src.config import MODEL_NAME, BEST_ADAPTER_DIR

from src.util import detect_device, build_fim_prompt, CompleteResponse, CompleteRequest, Suggestion, GenerateResponse, GenerateRequest

device = detect_device()

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
    local_files_only=True,
    fix_mistral_regex=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if device == "cuda" else None,
    device_map=None,
    local_files_only=True,
).to(device)

model = PeftModel.from_pretrained(
    base,
    str(BEST_ADAPTER_DIR)
).to(device)
model.eval()


@app.post("/complete", response_model=CompleteResponse)
def complete(req: CompleteRequest):
    prompt = build_fim_prompt(req.prefix, req.suffix) if req.suffix else req.prefix

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=req.temperature > 0,
            temperature=req.temperature,
            top_p=req.top_p,
            num_return_sequences=req.n,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    suggestions = []

    for seq in outputs:
        decoded = tokenizer.decode(seq, skip_special_tokens=False)

        # return only the newly generated continuation
        gen = decoded[len(prompt):]

        # remove <|endoftext|> if present
        gen = gen.split("<|endoftext|>")[0].rstrip()

        # optionally cut at stop tokens/newlines
        suggestions.append(Suggestion(text=gen))

    return CompleteResponse(suggestions=suggestions)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    prompt = ""
    if req.language.lower().startswith("py"):
        prompt = "\n".join([
            "# Write Python code for the following docstring",
            '"""',
            req.comment.strip(),
            '"""',
            "",
        ])
    else:
        prompt = "\n".join([
            "// Write TypeScript/JavaScript code for the following comment",
            "/**",
            req.comment.strip(),
            "*/",
            "",
        ])

    if req.surrounding:
        prompt = req.surrounding + "\n\n" + prompt

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=req.temperature > 0,
            temperature=req.temperature,
            top_p=req.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(out[0], skip_special_tokens=False)
    code = decoded[len(prompt):]
    code = code.split("<|endoftext|>")[0].rstrip()
    return GenerateResponse(code=code)

