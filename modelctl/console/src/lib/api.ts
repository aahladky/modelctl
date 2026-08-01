import type { CancelResult, JobRow, ModelRow } from "./types";

/* Session-cookie auth: an expired session answers 401 on /api/*; the
   only recovery is the login page, so go there instead of rendering a
   dead console. */
async function get<T>(path: string): Promise<T> {
  const r = await fetch(path, { headers: { Accept: "application/json" } });
  if (r.status === 401) {
    location.href = "/login?next=" + encodeURIComponent(location.pathname);
    throw new Error("unauthorized");
  }
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

export const fetchModels = () => get<ModelRow[]>("/api/v2/models");
export const fetchJobs = () => get<JobRow[]>("/api/v2/jobs");

export async function cancelJob(id: string): Promise<CancelResult> {
  const r = await fetch(`/api/v2/jobs/${id}/cancel`, { method: "POST" });
  if (r.status === 401) {
    location.href = "/login?next=" + encodeURIComponent(location.pathname);
    throw new Error("unauthorized");
  }
  if (!r.ok) throw new Error(`cancel ${id} -> ${r.status}`);
  return r.json();
}

export function fmtGiB(bytes: number | null | undefined, digits = 1): string {
  if (bytes == null) return "—";
  return (bytes / 2 ** 30).toFixed(digits);
}

export function fmtAgo(epochSeconds: number | null | undefined): string {
  if (!epochSeconds) return "";
  const s = Math.max(0, Math.floor(Date.now() / 1000 - epochSeconds));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function fmtClock(msEpoch: number): string {
  const d = new Date(msEpoch);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

export function fmtUp(startedEpoch: number): string {
  const s = Math.max(0, Math.floor(Date.now() / 1000 - startedEpoch));
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 86400) % 3600 / 60)}m`;
  return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
}
