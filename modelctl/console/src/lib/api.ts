import type {
  AdmissionPreview, CancelResult, ConfigSaveResult, GgufAnalysis, HistoryRow,
  JobRow, LogTail, ModelDetail, ModelRow, PlanRow, RegisterData, RepoContents,
  SearchResult, WizardDetail, WizardSummary,
} from "./types";

/* Server refusals (gates, scratch-safe mode, validation) answer with a
   JSON body naming the reason; surface that instead of a bare status so
   the UI can show the server's own words. */
export class ApiError extends Error {
  status: number;
  body: Record<string, unknown>;
  constructor(status: number, body: Record<string, unknown>, path: string) {
    super(String(body.error ?? body.reason ?? `${path} -> ${status}`));
    this.status = status;
    this.body = body;
  }
}

async function fail(r: Response, path: string): Promise<never> {
  let body: Record<string, unknown> = {};
  try {
    body = await r.json();
  } catch {
    /* non-JSON error body: the status alone will have to do */
  }
  throw new ApiError(r.status, body, path);
}

/* Session-cookie auth: an expired session answers 401 on /api/*; the
   only recovery is the login page, so go there instead of rendering a
   dead console. */
export async function get<T>(path: string): Promise<T> {
  const r = await fetch(path, { headers: { Accept: "application/json" } });
  if (r.status === 401) {
    location.href = "/login?next=" + encodeURIComponent(location.pathname);
    throw new Error("unauthorized");
  }
  if (!r.ok) return fail(r, path);
  return r.json();
}

export async function postJson<T>(path: string, body: unknown = {}): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body ?? {}),
  });
  if (r.status === 401) {
    location.href = "/login?next=" + encodeURIComponent(location.pathname);
    throw new Error("unauthorized");
  }
  if (!r.ok) return fail(r, path);
  return r.json();
}

export const fetchModels = () => get<ModelRow[]>("/api/v2/models");
export const fetchJobs = () => get<JobRow[]>("/api/v2/jobs");

/* ---- model hub ---- */
const enc = encodeURIComponent;

export const fetchModelDetail = (name: string) =>
  get<ModelDetail>(`/api/v2/models/${enc(name)}`);
export const fetchModelPlans = (name: string) =>
  get<PlanRow[]>(`/api/v2/models/${enc(name)}/plans`);
export const fetchModelHistory = (name: string) =>
  get<HistoryRow[]>(`/api/v2/models/${enc(name)}/history`);
export const fetchLogTail = (name: string) =>
  get<LogTail>(`/api/v2/models/${enc(name)}/logtail`);

export function admissionQuery(ctx?: number | null,
                               budgets?: Record<string, number> | null,
                               moeMode?: string | null): string {
  const q = new URLSearchParams();
  if (ctx != null) q.set("ctx", String(ctx));
  if (budgets) {
    for (const [dev, bytes] of Object.entries(budgets)) {
      q.set(`budget_bytes.${dev}`, String(Math.round(bytes)));
    }
  }
  if (moeMode != null) q.set("moe_mode", moeMode);
  const s = q.toString();
  return s ? `?${s}` : "";
}

export const fetchAdmission = (name: string, ctx?: number | null,
                               budgets?: Record<string, number> | null,
                               moeMode?: string | null) =>
  get<AdmissionPreview>(
    `/api/v2/models/${enc(name)}/admission${admissionQuery(ctx, budgets, moeMode)}`);

export const saveModelConfig = (name: string, body: {
  updates: Record<string, string | boolean>;
  budgets_bytes?: Record<string, number> | null;
  moe_mode?: string | null;
  accept_structural?: boolean;
}) => postJson<ConfigSaveResult>(`/api/v2/models/${enc(name)}/config`, body);

/* ---- add wizard ---- */

export const fetchWizards = () => get<WizardSummary[]>("/api/v2/wizards");
export const createWizard = () => postJson<WizardDetail>("/api/v2/wizards");
export const fetchWizard = (id: string) =>
  get<WizardDetail>(`/api/v2/wizard/${enc(id)}`);
export const wizardSource = (id: string, body: {
  source_type: string; repo_id?: string; local_path?: string;
}) => postJson<WizardDetail>(`/api/v2/wizard/${enc(id)}/source`, body);
export const fetchWizardInspect = (id: string) =>
  get<{ contents: RepoContents | null; error: string }>(
    `/api/v2/wizard/${enc(id)}/inspect`);
export const wizardInspectSubmit = (id: string, quant: string) =>
  postJson<WizardDetail>(`/api/v2/wizard/${enc(id)}/inspect`, { quant });
export const wizardDownloadNext = (id: string) =>
  postJson<WizardDetail>(`/api/v2/wizard/${enc(id)}/download/next`);
export const fetchWizardAnalyze = (id: string) =>
  get<{ profile: ModelDetail | null; analysis: GgufAnalysis | null }>(
    `/api/v2/wizard/${enc(id)}/analyze`);
export const wizardAnalyzeNext = (id: string) =>
  postJson<WizardDetail>(`/api/v2/wizard/${enc(id)}/analyze/next`);
export const fetchWizardPlans = (id: string) =>
  get<{ plans: PlanRow[]; selected_plan_id: string }>(
    `/api/v2/wizard/${enc(id)}/plans`);
export const wizardPlansSubmit = (id: string, plan_id: string, action: string) =>
  postJson<WizardDetail>(`/api/v2/wizard/${enc(id)}/plans`, { plan_id, action });
export const wizardTestRun = (id: string) =>
  postJson<WizardDetail>(`/api/v2/wizard/${enc(id)}/test/run`);
export const wizardTestNext = (id: string) =>
  postJson<WizardDetail>(`/api/v2/wizard/${enc(id)}/test/next`);
export const fetchWizardRegister = (id: string, ctx?: number | null,
                                    budgets?: Record<string, number> | null,
                                    moeMode?: string | null) =>
  get<RegisterData>(
    `/api/v2/wizard/${enc(id)}/register${admissionQuery(ctx, budgets, moeMode)}`);
export const wizardRegisterSubmit = (id: string, body: Record<string, unknown>) =>
  postJson<WizardDetail>(`/api/v2/wizard/${enc(id)}/register`, body);
export const wizardRetry = (id: string, step: string) =>
  postJson<WizardDetail>(`/api/v2/wizard/${enc(id)}/retry/${enc(step)}`);
export const wizardDelete = (id: string) =>
  postJson<{ deleted: string }>(`/api/v2/wizard/${enc(id)}/delete`);
export const hfSearch = (q: string) =>
  get<{ results: SearchResult[]; error: string }>(
    `/api/v2/hf/search?q=${enc(q)}`);

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
