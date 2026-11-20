/**
 * Phase 12A Part B — JavaScript parser front end for the corpus audit.
 *
 * Compiles each corpus snippet with V8 to decide whether it is syntactically
 * valid, and describes its structure. Compilation is NOT execution: a snippet
 * is handed to `new vm.Script(...)` / `new vm.SourceTextModule(...)` and the
 * resulting object is discarded without ever being run. No corpus function is
 * called, no global is touched, and the input is only ever read from stdin.
 *
 * Input  (stdin, one JSON object per line):  {"id": "...", "code": "..."}
 * Output (stdout, one JSON object per line): identical `id` plus the fields
 *         documented in `describe()`.
 */

import vm from "node:vm";
import { createInterface } from "node:readline";

const TRIVIA = /^(\s+|\/\/[^\n]*|\/\*[\s\S]*?\*\/)*/;

/** Index just past leading whitespace and comments. */
function skipTrivia(code, i = 0) {
  let j = i;
  for (;;) {
    while (j < code.length && /\s/.test(code[j])) j++;
    if (code.startsWith("//", j)) {
      const nl = code.indexOf("\n", j);
      j = nl === -1 ? code.length : nl + 1;
      continue;
    }
    if (code.startsWith("/*", j)) {
      const end = code.indexOf("*/", j + 2);
      j = end === -1 ? code.length : end + 2;
      continue;
    }
    return j;
  }
}

/** Index just past a string, template literal or comment starting at `i`. */
function skipOpaque(code, i) {
  const ch = code[i];
  if (ch === "/" && code[i + 1] === "/") {
    const nl = code.indexOf("\n", i + 2);
    return nl === -1 ? code.length : nl + 1;
  }
  if (ch === "/" && code[i + 1] === "*") {
    const end = code.indexOf("*/", i + 2);
    return end === -1 ? code.length : end + 2;
  }
  if (ch === "'" || ch === '"' || ch === "`") {
    let j = i + 1;
    while (j < code.length) {
      if (code[j] === "\\") {
        j += 2;
        continue;
      }
      if (code[j] === ch) return j + 1;
      j++;
    }
    return code.length;
  }
  return i + 1;
}

/**
 * Scan from an opening bracket to its match, ignoring brackets inside strings,
 * comments and templates. Returns the index of the closing bracket, or -1.
 */
function findClose(code, start, open, close) {
  if (code[start] !== open) return -1;
  let depth = 0;
  let i = start;
  while (i < code.length) {
    const ch = code[i];
    if (ch === "'" || ch === '"' || ch === "`" || ch === "/") {
      const next = skipOpaque(code, i);
      if (next !== i + 1 || ch === "'" || ch === '"' || ch === "`") {
        i = next;
        continue;
      }
    }
    if (ch === open) depth++;
    else if (ch === close) {
      depth--;
      if (depth === 0) return i;
    }
    i++;
  }
  return -1;
}

const HEADER_PATTERNS = [
  { kind: "function_decl", re: /^export\s+default\s+(?:async\s+)?function\b\s*(\w*)/ },
  { kind: "function_decl", re: /^export\s+(?:async\s+)?function\b\s*(\w*)/ },
  { kind: "function_decl", re: /^(?:async\s+)?function\b\s*(\w*)/ },
  { kind: "function_assignment", re: /^module\.exports\s*=\s*(?:async\s+)?function\b\s*(\w*)/ },
  { kind: "function_assignment", re: /^exports\.(\w+)\s*=\s*(?:async\s+)?function\b\s*(\w*)/ },
  { kind: "function_assignment", re: /^(\w+)\s*=\s*(?:async\s+)?function\b\s*(\w*)/ },
  { kind: "arrow_assignment", re: /^(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*=>/ },
  { kind: "arrow_assignment", re: /^(\w+)\s*=\s*(?:async\s+)?(\w+)\s*=>/ },
  { kind: "class", re: /^class\b\s*(\w*)/ },
];

const DECLARATOR = /^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*([\s\S]*)$/;

/** Structural description of one snippet, without executing anything. */
function describe(code) {
  const start = skipTrivia(code);
  const head = code.slice(start);
  let kind = "other";
  let name = null;
  let params = null;
  let bodyKind = null;
  let complete = false;

  for (const { kind: candidate, re } of HEADER_PATTERNS) {
    const match = head.match(re);
    if (!match) continue;
    kind = candidate;
    const groups = match.slice(1).filter((g) => g !== undefined);
    name = groups.find((g) => /^\w+$/.test(g) && !["async", "function"].includes(g)) || null;
    params = match[2] && candidate === "arrow_assignment" ? match[2] : params;
    break;
  }

  // `const f = function (...)` / `const f = (...) =>` / `const f = class`
  const declarator = head.match(DECLARATOR);
  if (kind === "other" && declarator) {
    const [, declared, rhs] = declarator;
    const fn = rhs.match(/^(?:async\s+)?function\b\s*(\w*)\s*(\([\s\S]*)$/);
    const arrow = rhs.match(/^(?:async\s+)?(\([^)]*\))\s*=>([\s\S]*)$/);
    const bareArrow = rhs.match(/^(?:async\s+)?(\w+)\s*=>([\s\S]*)$/);
    if (fn) {
      kind = "function_assignment";
      name = fn[1] || declared;
      params = null;
      bodyKind = "block";
    } else if (arrow) {
      kind = "arrow_assignment";
      name = declared;
      params = arrow[1].slice(1, -1);
      bodyKind = arrow[2].trimStart().startsWith("{") ? "block" : "expression";
    } else if (bareArrow) {
      kind = "arrow_assignment";
      name = declared;
      params = bareArrow[1];
      bodyKind = bareArrow[2].trimStart().startsWith("{") ? "block" : "expression";
    } else if (/^class\b/.test(rhs)) {
      kind = "class";
      name = declared;
    }
  }

  if (kind === "function_decl" || kind === "function_assignment") {
    const paren = code.indexOf("(", start);
    if (paren !== -1) {
      const close = findClose(code, paren, "(", ")");
      if (close !== -1) {
        params = code.slice(paren + 1, close);
        const after = skipTrivia(code, close + 1);
        if (code[after] === "{") {
          const bodyEnd = findClose(code, after, "{", "}");
          bodyKind = "block";
          complete = bodyEnd !== -1 && skipTrivia(code, bodyEnd + 1) >= code.replace(/;?\s*$/, "").length;
        }
      }
    }
  } else if (kind === "arrow_assignment") {
    const arrowIndex = code.indexOf("=>", start);
    if (arrowIndex !== -1) {
      const after = skipTrivia(code, arrowIndex + 2);
      if (code[after] === "{") {
        bodyKind = "block";
        const bodyEnd = findClose(code, after, "{", "}");
        complete = bodyEnd !== -1 && skipTrivia(code, bodyEnd + 1) >= code.replace(/;?\s*$/, "").length;
      } else {
        bodyKind = "expression";
        complete = code.slice(after).trim().length > 0;
      }
    }
  } else if (kind === "class") {
    const braceIndex = code.indexOf("{", start);
    if (braceIndex !== -1) {
      const bodyEnd = findClose(code, braceIndex, "{", "}");
      complete = bodyEnd !== -1 && skipTrivia(code, bodyEnd + 1) >= code.replace(/;?\s*$/, "").length;
    }
    bodyKind = "class";
  }

  if (params !== null) params = params.trim().slice(0, 300);

  return {
    kind,
    name,
    params,
    body_kind: bodyKind,
    complete,
    has_require: /\brequire\s*\(/.test(code),
    has_import_export: /^\s*(import|export)\b/m.test(code) || /\bmodule\.exports\b/.test(code),
    uses_this: /\bthis\b/.test(code),
    uses_host_global:
      /\b(document|window|global|process|navigator|localStorage|fetch|XMLHttpRequest)\b/.test(code),
    console_calls: (code.match(/\bconsole\.\w+/g) || []).length,
  };
}

/** Compile-only syntax validation; nothing is executed. */
function parseStatus(code) {
  try {
    new vm.Script(code, { filename: "<corpus>" });
    return { parse: "valid_script", error: null };
  } catch (error) {
    const message = String((error && error.message) || error).split("\n")[0].slice(0, 200);
    const moduleSyntax = /import statement|Unexpected token 'export'|import\.meta|Cannot use import/i.test(
      message,
    );
    if (moduleSyntax && typeof vm.SourceTextModule === "function") {
      try {
        new vm.SourceTextModule(code, { identifier: "<corpus>" });
        return { parse: "valid_module", error: null };
      } catch (moduleError) {
        return {
          parse: "invalid",
          error: String((moduleError && moduleError.message) || moduleError).split("\n")[0].slice(0, 200),
        };
      }
    }
    return { parse: "invalid", error: message };
  }
}

const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
const out = [];
let pending = 0;

function flush() {
  if (out.length) process.stdout.write(out.join(""));
  out.length = 0;
}

for await (const line of input) {
  if (!line.trim()) continue;
  const record = JSON.parse(line);
  const code = String(record.code ?? "");
  let described;
  try {
    described = describe(code);
  } catch (error) {
    described = { kind: "scan_failed", name: null, params: null, body_kind: null, complete: false };
  }
  const status = parseStatus(code);
  out.push(JSON.stringify({ id: record.id, ...status, ...described }) + "\n");
  pending += 1;
  if (pending >= 500) {
    flush();
    pending = 0;
  }
}
flush();
