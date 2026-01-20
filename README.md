# Code Autocomplete Assistant

A local, offline-first code completion and generation assistant specialized for Python, TypeScript, and AWS CDK patterns.

## Features

- **Local Code Completion**: Fast, context-aware code completion for Python and TypeScript
- **Comment-to-Code Generation**: Convert natural language comments into executable code
- **AWS CDK Specialization**: Optimized for AWS CDK patterns and best practices
- **Offline Operation**: Works entirely locally without internet connectivity
- **Configurable**: Project-specific and global configuration support
- **VS Code Integration**: Works with Continue extension or custom VS Code extension

## Quick Start

### 1. Environment Setup

```bash
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Directory Structure

```
code-autocomplete-assistant/
├── server/           # FastAPI server and core logic
├── models/           # Model files (base + LoRA adapters)
├── datasets/         # Training datasets and manifests
├── scripts/          # Training and evaluation scripts
├── evaluation/       # Benchmarking and evaluation utilities
├── config/           # Configuration management
├── vscode/           # VS Code integration files
└── logs/             # Application logs
```

### 3. Configuration

The system supports both global and project-specific configurations:

- **Global config**: `~/.code-assistant/config.yaml`
- **Project config**: `.code-assistant.yaml` in project root

Create a project configuration template:

```bash
python -c "from config.settings import config_manager; config_manager.create_project_config_template('.')"
```

### 4. Model Setup

Place your model files in the `models/` directory:

```
models/
├── base/                 # Base transformer model
├── trained_adapter/      # LoRA adapter for training data
```

To download a pre-trained base model (e.g., Llama 2 7B):

```bash
# Example command to download a base model
# (Replace with actual download instructions as needed)
mkdir -p models/base/Qwen
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
m='Qwen/Qwen2.5-Coder-3B'; \
AutoTokenizer.from_pretrained(m).save_pretrained('models/base/Qwen2.5-Coder-3B'); \
AutoModelForCausalLM.from_pretrained(m).save_pretrained('models/base/Qwen2.5-Coder-3B')"
```

### 5. Running the Server

```bash
# Start the FastAPI server
python -m uvicorn src.server:app --host 127.0.0.1 --port 8000
```

## API Endpoints

- `POST /complete` - Code completion
- `POST /generate` - Comment-to-code generation  
- `GET /health` - Health check                   # TODO: implement
- `GET /metrics` - Performance metrics           # TODO: implement
- `POST /config/reload` - Reload configuration   # TODO: implement

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=server --cov=config --cov=scripts

# Run property-based tests
pytest -k "property" -v
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## Dataset Preparation

The project includes Jupyter notebooks for dataset preparation:

### Stage A: CodeSearchNet Dataset
```bash
# Run the notebook to extract Python and JavaScript examples
jupyter notebook notebooks/01_dataset_preparation.ipynb
```

### Stage B: AWS CDK Dataset

The Stage B dataset combines examples from two sources:

1. **aws-cdk-examples**: Real-world CDK examples from AWS samples
2. **aws-cdk library**: Official CDK library source code with constructs

```bash
# Extract from aws-cdk-examples repository
jupyter notebook notebooks/02_aws_examples_cdk_extraction.ipynb

# Extract from aws-cdk library source (TypeScript + Python)
python scripts/extract_cdk_lib_data.py

# Check dataset statistics
python scripts/check_stage_b_data.py
```

**Current Stage B Dataset:**
- Total samples: 1,758
- Training: 1,494 samples
- Validation: 264 samples
- Sources: aws-cdk-examples (1,118) + aws-cdk-lib (640)
- Languages: TypeScript (753), Python (20), mixed (985)

## Training Pipeline

The system supports two-stage training:

1. **Stage A**: Fine-tune on CodeSearchNet for general comment-to-code capability
2. **Stage B**: Domain adaptation on AWS CDK examples

See `scripts/` directory for training scripts.

## Create VS Code Extension

```(bash)
npm install -g yo generator-code
yo code
```

Choose:
- "New Extension (TypeScript)" 
- ✔ What type of extension do you want to create? New Extension (TypeScript)
- ✔ What's the name of your extension? code-autocomplete-assistant
- ✔ What's the identifier of your extension? code-autocomplete-assistant
- ✔ What's the description of your extension? A local, offline-first code completion and generation assistant specialized for Python, TypeScript, and AWS CDK patterns.
- ✔ Initialize a git repository? No
- ✔ Which bundler to use? unbundled
- ✔ Which package manager to use? npm
- Activation: "on command"

```
cd code-autocomplete-assistant
npm install axios
```

### Adding commands

* codeAutocompleteAssistant.complete
* codeAutocompleteAssistant.commentToCode

#### src/extension.ts
```typescript
import * as vscode from "vscode";

async function postJson(url: string, body: any) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json();
}

function getPrefixSuffix(doc: vscode.TextDocument, pos: vscode.Position) {
  const fullText = doc.getText();
  const offset = doc.offsetAt(pos);

  const prefix = fullText.slice(Math.max(0, offset - 4000), offset);
  const suffix = fullText.slice(offset, Math.min(fullText.length, offset + 2000));

  return { prefix, suffix };
}

function findNearestCommentBlock(doc: vscode.TextDocument, pos: vscode.Position): string | null {
  const maxLines = 20;
  let line = pos.line;

  const out: string[] = [];
  for (let i = 0; i < maxLines && line >= 0; i++, line--) {
    const text = doc.lineAt(line).text.trim();
    if (text.startsWith("#")) out.push(text.replace(/^#\s?/, ""));
    else if (text.startsWith("//")) out.push(text.replace(/^\/\/\s?/, ""));
    else if (text === "") continue;
    else break;
  }

  if (out.length === 0) return null;
  return out.reverse().join(" ").trim();
}

export function activate(context: vscode.ExtensionContext) {
  const baseUrl = "http://127.0.0.1:8000";

  // Command A: autocomplete
  const completeCmd = vscode.commands.registerCommand("codeAutocompleteAssistant.complete", async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const doc = editor.document;
    const pos = editor.selection.active;

    const { prefix, suffix } = getPrefixSuffix(doc, pos);
    const language = doc.languageId; // "python", "typescript", etc.

    vscode.window.withProgress(
      { location: vscode.ProgressLocation.Window, title: "Code Autocomplete Assistant: completing…" },
      async () => {
        const resp = await postJson(`${baseUrl}/complete`, {
          language,
          prefix,
          suffix,
          max_tokens: 64,
          temperature: 0.2,
          top_p: 0.95,
          n: 1,
        });

        const suggestion = resp.suggestions?.[0]?.text ?? "";
        await editor.edit((edit) => {
          edit.insert(pos, suggestion);
        });
      }
    );
  });

  // Command B: comment → code
  const commentCmd = vscode.commands.registerCommand("codeAutocompleteAssistant.commentToCode", async () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor) return;

    const doc = editor.document;
    const pos = editor.selection.active;
    const language = doc.languageId;

    // Either selected text OR nearest preceding comment
    const sel = editor.selection;
    let comment = doc.getText(sel).trim();
    if (!comment) {
      const block = findNearestCommentBlock(doc, pos);
      if (!block) {
        vscode.window.showWarningMessage("No comment block found. Select a comment or place cursor under one.");
        return;
      }
      comment = block;
    }

    const resp = await postJson(`${baseUrl}/generate`, {
      language,
      comment,
      max_tokens: 256,
      temperature: 0.2,
      top_p: 0.95,
    });

    const code = resp.code ?? "";
    await editor.edit((edit) => {
      edit.insert(pos, code);
    });
  });

  context.subscriptions.push(completeCmd, commentCmd);
}

export function deactivate() {}
```
### Packaging the Extension

#### package.json
```json
"commands": [
  { "command": "codeAutocompleteAssistant.complete", "title": "Local Assistant: Autocomplete" },
  { "command": "codeAutocompleteAssistant.commentToCode", "title": "Local Assistant: Comment → Code" }
],
"keybindings": [
  {
    "command": "codeAutocompleteAssistant.complete",
    "key": "ctrl+shift+a",
    "when": "editorTextFocus"
  },
  {
    "command": "localCodeAssistant.commentToCode",
    "key": "ctrl+shift+z",
    "when": "editorTextFocus"
  }
]
```

## License

MIT License - see LICENSE file for details.