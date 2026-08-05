/* What the home screen's trouble band shows, as a pure function so the
 * band is testable without a DOM.
 *
 * Two kinds of trouble, with different lifetimes:
 *
 * - Current-state facts (a disabled-but-registered model, an absent
 *   node, a fleet section that errored) show for as long as they are
 *   true. Fixing the thing clears the row; nothing to expire.
 * - Job history is not state. A failed bench from three days ago is
 *   the Record's business, so job-derived rows expire after a window.
 *   Without one, "N things to look at" becomes the steady state and
 *   the calm line never renders -- seen live 2026-08-05, six stale
 *   failures from two days prior filling the band.
 *
 * Details are sentences, never tracebacks. The job error's first line
 * already says what happened in words ("this plan needs 10.2 GiB on
 * RPC:... but only 0.0 GiB is free"); everything from "Traceback" on
 * is forensics and belongs on the Record.
 */
import type { FleetView, JobRow, ModelRow } from "./types";

export interface TroubleItem {
  what: string;
  detail: string;
}

/* Job failures older than this stay off the home screen. */
export const JOB_TROUBLE_WINDOW_S = 24 * 3600;

const DETAIL_MAX_CHARS = 200;

export function failureSummary(error: string): string {
  /* An exception with an empty message arrives as "\nTraceback (...)":
     nothing before the marker means there are no words to show, and
     falling back to the raw text would serve the traceback header --
     the exact defect this function exists to prevent. */
  const beforeTrace = error.split("Traceback (most recent call last)")[0].trim();
  if (!beforeTrace) return "failed — the job says why";
  const line = beforeTrace.split("\n")[0].trim();
  return line.length > DETAIL_MAX_CHARS
    ? line.slice(0, DETAIL_MAX_CHARS - 1).trimEnd() + "…"
    : line;
}

/* Everything the operator would want to have been told, from whichever
   source knows it. Best-effort by design: a source that cannot be
   reached contributes nothing rather than taking the page down.
   `nowS` is epoch seconds, matching the job rows' clock. */
export function troubles(models: ModelRow[], jobs: JobRow[],
                         fleet: FleetView | null, nowS: number): TroubleItem[] {
  const out: TroubleItem[] = [];
  const cutoff = nowS - JOB_TROUBLE_WINDOW_S;
  for (const j of jobs) {
    const when = j.finished ?? j.started ?? j.created;
    if (when < cutoff) continue;
    if (j.status === "failed") {
      out.push({ what: j.title || j.type,
                 detail: j.error ? failureSummary(j.error)
                                 : "failed — the job says why" });
    } else if (j.router_reloaded === false) {
      out.push({ what: j.title || j.type,
                 detail: "applied, but the router did not reload — the "
                         + "previous placement is still serving" });
    }
  }
  for (const m of models) {
    if (m.registered && !m.enabled) {
      out.push({ what: m.name,
                 detail: "registered but disabled — the API will not serve it" });
    }
    if (!m.registered && m.enabled) {
      out.push({ what: m.name,
                 detail: "enabled but not registered with the router" });
    }
  }
  for (const n of fleet?.nodes ?? []) {
    if (n.enabled && n.location === "remote"
        && n.presence.state !== "PRESENT") {
      out.push({ what: n.name,
                 detail: `${n.presence.state} — ${n.presence.detail
                          || "no recent probe"}; its memory is out of the `
                         + "planner's reach" });
    }
  }
  for (const [where, message] of Object.entries(fleet?.errors ?? {})) {
    out.push({ what: where, detail: String(message) });
  }
  return out;
}
