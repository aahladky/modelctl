/* Every class name a component asks for must exist in the stylesheet.
 *
 * This is the check for the one defect of 2026-08-04 that was genuinely
 * the front end's fault: the wizard screens were written against a CSS
 * vocabulary nobody had defined -- btn, btn-primary, form-grid, field,
 * error, warning -- so their buttons rendered as bare links and their
 * error messages rendered as unstyled body text. Invisible as errors,
 * which is the worst way for an error to render. No test framework
 * catches that; a component test asserts the class is on the element,
 * not that the class means anything.
 *
 * Deliberately conservative. It reads only literal class names -- a
 * template hole or a computed name is skipped rather than guessed at --
 * because a check that cries wolf gets disabled, and then the real
 * defect ships again. Under-reporting is the safe direction here.
 *
 * No dependencies: node tools/check-classes.mjs
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const SRC = new URL("../src/", import.meta.url).pathname;
const SHEET = join(SRC, "tokens.css");

/* Names that are never defined in the sheet because something else owns
   them. Each needs a reason; an unexplained entry here is how a check
   quietly stops checking. */
const NOT_OURS = new Set([
  /* Set on the root element by the theme toggle and matched by an
     attribute selector, not a class rule. */
  "dark", "light",
]);

function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) out.push(...walk(path));
    else if (/\.(tsx|ts)$/.test(entry) && !entry.endsWith(".test.ts")) {
      out.push(path);
    }
  }
  return out;
}

/* Class rules the sheet defines. Includes names appearing anywhere in a
   selector, so `.widget .sub` and `.a.b` both register. */
function definedClasses(css) {
  const found = new Set();
  for (const [, name] of css.matchAll(/\.(-?[_a-zA-Z][\w-]*)/g)) {
    found.add(name);
  }
  return found;
}

/* Literal class names a source file asks for.
   Handles  class="a b",  class={"a b"},  class={`a ${x}`}  and the
   conditional  class={c ? "a" : "b"}, by taking every string literal in
   the attribute value and ignoring everything computed. */
function requestedClasses(source) {
  const asked = new Map();
  const attr = /\bclass(?:Name)?=(?:"([^"]*)"|'([^']*)'|\{([^}]*)\})/g;
  for (const match of source.matchAll(attr)) {
    const [, dq, sq, brace] = match;
    const line = source.slice(0, match.index).split("\n").length;
    let literals = [];
    if (dq !== undefined) literals = [dq];
    else if (sq !== undefined) literals = [sq];
    else if (brace !== undefined) {
      /* Only the quoted parts. A ${...} hole or a bare identifier is a
         computed name and is not guessed at. */
      literals = [...brace.matchAll(/["'`]([^"'`]*)["'`]/g)].map((m) => m[1]);
    }
    for (const literal of literals) {
      for (const name of literal.split(/\s+/)) {
        if (!name || name.includes("$") || NOT_OURS.has(name)) continue;
        if (!/^-?[_a-zA-Z][\w-]*$/.test(name)) continue;
        if (!asked.has(name)) asked.set(name, line);
      }
    }
  }
  return asked;
}

const defined = definedClasses(readFileSync(SHEET, "utf8"));
const missing = [];
for (const file of walk(SRC)) {
  for (const [name, line] of requestedClasses(readFileSync(file, "utf8"))) {
    if (!defined.has(name)) {
      missing.push(`${file.replace(SRC, "src/")}:${line}  .${name}`);
    }
  }
}

if (missing.length) {
  console.error("class names used but never defined in tokens.css:\n");
  for (const row of missing.sort()) console.error(`  ${row}`);
  console.error(`\n${missing.length} undefined class name(s). A screen that `
    + "asks for a class nothing defines renders unstyled -- which is how "
    + "an error message ships invisible.");
  process.exit(1);
}
console.log(`ok: every class name resolves (${defined.size} defined)`);
