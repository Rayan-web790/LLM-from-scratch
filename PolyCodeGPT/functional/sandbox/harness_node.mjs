// In-container Node harness (unprivileged user, no network).
//
// Reads one JSON payload from stdin:
//   {token, entry, code, cases: [{name, kind, args, expected | expect_error}]}
//
// Writes token-prefixed JSON lines to stderr so candidate output cannot corrupt
// the protocol. Every candidate evaluation runs inside `vm` with a hard
// synchronous timeout (the outer container timeout is the second line of
// defence).
import fs from "node:fs";
import vm from "node:vm";

const payload = JSON.parse(fs.readFileSync(0, "utf8"));
const token = payload.token;

function emit(obj) {
  process.stderr.write(token + JSON.stringify(obj) + "\n");
}

function show(value, limit = 300) {
  let text;
  try {
    text = typeof value === "string" ? JSON.stringify(value) : String(JSON.stringify(value));
  } catch {
    text = `<unserializable ${typeof value}>`;
  }
  if (text === undefined) text = String(value);
  return text.length <= limit ? text : text.slice(0, limit) + "...";
}

function normalize(value) {
  // JSON round-trip so cross-realm objects compare structurally.
  try {
    return JSON.parse(JSON.stringify(value === undefined ? null : value));
  } catch {
    return undefined;
  }
}

const sandbox = Object.create(null);
sandbox.console = { log() {}, error() {}, warn() {}, info() {} };
sandbox.Math = Math;
sandbox.JSON = JSON;
sandbox.Number = Number;
sandbox.String = String;
sandbox.Array = Array;
sandbox.Object = Object;
sandbox.Boolean = Boolean;
sandbox.parseInt = parseInt;
sandbox.parseFloat = parseFloat;
sandbox.isNaN = isNaN;
sandbox.isFinite = isFinite;
sandbox.undefined = undefined;

const context = vm.createContext(sandbox);

try {
  vm.runInContext(payload.code, context, { timeout: 2000, displayErrors: false });
} catch (err) {
  if (err && err.code === "ERR_SCRIPT_EXECUTION_TIMEOUT") {
    emit({ type: "timeout", phase: "load" });
  } else {
    emit({ type: "exec_error", error: err && err.name ? err.name : "Error", message: String(err && err.message).slice(0, 500) });
  }
  process.exit(0);
}

const fn = context[payload.entry];
if (typeof fn !== "function") {
  emit({ type: "entry_missing", entry: payload.entry });
  process.exit(0);
}

emit({ type: "entry_found" });

for (const c of payload.cases) {
  const wantError = c.expect_error;
  context.__ph_args = c.args;
  context.__ph_fn = fn;
  let value = null;
  let threw = null;
  try {
    value = vm.runInContext("__ph_fn.apply(undefined, __ph_args)", context, { timeout: 2000, displayErrors: false });
  } catch (err) {
    if (err && err.code === "ERR_SCRIPT_EXECUTION_TIMEOUT") {
      emit({ type: "case", name: c.name, kind, status: "timeout", error: "Timeout",
             message: "case exceeded the in-VM execution budget" });
      continue;
    }
    threw = err;
  }

  const kind = c.kind || "normal";

  if (threw) {
    const name = threw && threw.name ? threw.name : "Error";
    if (wantError) {
      const ok = Array.isArray(wantError)
        ? wantError.includes(name)
        : wantError === "any" || name === wantError;
      emit({
        type: "case", name: c.name, kind,
        status: ok ? "pass" : "fail",
        got: `raised ${name}: ${String(threw.message || "").slice(0, 200)}`,
        expected: `raises ${wantError}`,
      });
    } else {
      emit({ type: "case", name: c.name, kind, status: "exception", error: name, message: String(threw.message || "").slice(0, 300) });
    }
    continue;
  }

  const got = normalize(value);
  if (wantError) {
    emit({ type: "case", name: c.name, kind, status: "fail", got: show(value), expected: `raises ${wantError}` });
    continue;
  }

  const ok = JSON.stringify(got) === JSON.stringify(c.expected);
  emit({
    type: "case", name: c.name, kind,
    status: ok ? "pass" : "fail",
    got: show(got),
    expected: show(c.expected),
  });
}

emit({ type: "done" });
