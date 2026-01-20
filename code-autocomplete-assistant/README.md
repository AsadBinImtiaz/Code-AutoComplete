# Code Autocomplete Assistant

**Code Autocomplete Assistant** is a local, offline-first VS Code extension that provides intelligent code completion and comment-to-code generation for **Python** and **TypeScript**, with a special focus on **AWS CDK** coding patterns.

It connects to a **locally running inference server** (no cloud, no telemetry) and is designed for privacy-sensitive and enterprise environments.

---

## Features

### ✨ Autocomplete (Fill-in-the-Middle)
- Context-aware code completion at the cursor
- Supports Python and TypeScript
- Uses prefix + suffix (FIM) for higher-quality suggestions
- Optimized for infrastructure-as-code patterns (AWS CDK)

**Shortcut**
- **macOS**: `Ctrl + Shift + 1`

---

### ✍️ Comment → Code Generation
- Generate code from:
  - Selected comments, or
  - The nearest preceding comment block
- Produces clean scaffolds with idiomatic Python/TypeScript
- AWS services mentioned in comments bias CDK-style output

**Shortcut**
- **macOS**: `Ctrl + Shift + 2`

---

## How It Works

1. A local FastAPI server hosts a fine-tuned code LLM (PEFT / LoRA)
2. The VS Code extension sends editor context via HTTP
3. All inference runs **entirely on your machine**
4. No code or prompts ever leave the system

---

## Requirements

### 1. Local inference server (required)

You must run the local server before using the extension:

```bash
python -m uvicorn src.server:app --host 127.0.0.1 --port 8000
```

The server must be reachable at: 
```
http://127.0.0.1:8000
```

### 2. Supported languages

- Python
- TypeScript / JavaScript

### 3. VS Code

VS Code 1.106.x or newer

# Usage

## Autocomplete

1. Open a Python or TypeScript file
2. Place the cursor where completion is needed
3. Press Ctrl + Shift + 1
4. The suggestion is inserted at the cursor

## Comment → Code

1. Write or select a comment describing the code
2. Place the cursor below it
3. Press Ctrl + Shift + 2
4. Generated code is inserted at the cursor

## Extension Commands

| Command | Description |
|------|------------|
| `Local Assistant: Autocomplete` | Trigger local code completion at the cursor |
| `Local Assistant: Comment → Code` | Generate code from a natural-language comment |

You can invoke these commands via the **Command Palette** or their configured keyboard shortcuts.

---

## Configuration

The extension supports configuration via VS Code settings.

### Available Settings

```json
{
  "codeAutocompleteAssistant.serverUrl": "http://127.0.0.1:8000"
}
```

## Known Limitations

- The local inference server must be running before using the extension
- Generated code is single-shot (no streaming responses)
- Outputs are suggestions and should be reviewed before production use
- The extension does not perform multi-file refactoring

## Privacy & Security
- Runs fully offline
- No telemetry or usage tracking
- No code or prompts are sent outside the local machine
- Suitable for enterprise and privacy-sensitive environments

## Release Notes

### 0.0.1

* Initial internal release
* Local autocomplete using Fill-in-the-Middle (FIM)
* Comment-to-code generation
* Integration with local FastAPI inference server

## Internal Use

This extension is intended for internal distribution via .vsix files and is not published on the VS Code Marketplace.

# Support

For issues, improvements, or feature requests, please contact the internal maintainers or refer to the internal repository documentation.
