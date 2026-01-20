// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from "vscode";

type CompleteResponse = {
  suggestions: Array<{ text: string; score?: number }>;
  diagnostics?: any;
};

type GenerateResponse = {
  code: string;
  diagnostics?: any;
};

async function getFetch(): Promise<typeof fetch> {
  // Prefer native fetch if available
  const g = globalThis as any;
  if (typeof g.fetch === "function") { return g.fetch.bind(globalThis); }

  // Fallback: dynamically import ESM-only node-fetch
  const mod: any = await import("node-fetch");
  return (mod.default ?? mod) as typeof fetch;
}

async function postJson<T>(url: string, body: any): Promise<T> {
  const fetchFn = await getFetch();

  const res = await fetchFn(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text}`);
  }

  return (await res.json()) as T;
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
        const resp = await postJson<CompleteResponse>(`${baseUrl}/complete`, {
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

    const resp = await postJson<GenerateResponse>(`${baseUrl}/generate`, {
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
  vscode.window.showInformationMessage("Code Autocomplete Assistant activated");
}

export function deactivate() {}
