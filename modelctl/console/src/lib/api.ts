import type {
  AdmissionPreview, BudgetSubmitted, CacheResetResult, CancelResult,
  ConfigSaveResult, FleetView, GgufAnalysis, HistoryRow, JobRow, JobSubmitted,
  LogTail, ModelDetail, ModelRow, PlacementApplyResult, PlacementSelection,
  PlacementView, PlanRow, ProbeResult, RegisterData,
  RepoContents, RoutingMatrix, RunCommand, RuntimePolicyView, SaveResult,
  SearchResult, SettingsOverview, TierApplyResult, WizardDetail, WizardSummary,
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

/* No auth: the console is deliberately LAN-open (owner decision
   2026-08-03), so there is no 401 path to handle. */
export async function get<T>(path: string): Promise<T> {
  const r = await fetch(path, { headers: { Accept: "application/json" } });
  if (!r.ok) return fail(r, path);
  return r.json();
}

export async function postJson<T>(path: string, body: unknown = {}): Promise<T> {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body ?? {}),
  });
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

/* Runtime actions: the change queue answers with the job id, and the
   tick stream carries the rest. */
export const loadModel = (name: string) =>
  postJson<{ job_id: string }>(`/api/v2/models/${enc(name)}/load`);
export const unloadModel = (name: string) =>
  postJson<{ job_id: string }>(`/api/v2/models/${enc(name)}/unload`);

export const saveModelConfig = (name: string, body: {
  updates: Record<string, string | boolean>;
  budgets_bytes?: Record<string, number> | null;
  moe_mode?: string | null;
  accept_structural?: boolean;
}) => postJson<ConfigSaveResult>(`/api/v2/models/${enc(name)}/config`, body);

/* ---- phase 4: operations ----
   Every write here is the typed re-home of a control the phase-3 cutover
   removed. They answer with the job id; the tick stream carries the rest,
   so nothing on these paths polls. */

export const restartModel = (name: string) =>
  postJson<JobSubmitted>(`/api/v2/models/${enc(name)}/restart`);
export const resetModelCache = (name: string) =>
  postJson<CacheResetResult>(`/api/v2/models/${enc(name)}/cache/reset`);
export const deleteModel = (name: string) =>
  postJson<JobSubmitted>(`/api/v2/models/${enc(name)}/delete`);
export const benchModel = (name: string, max_tokens: number, runs: number) =>
  postJson<JobSubmitted & { max_tokens: number; runs: number }>(
    `/api/v2/models/${enc(name)}/bench`, { max_tokens, runs });
export const smokeModel = (name: string) =>
  postJson<JobSubmitted>(`/api/v2/models/${enc(name)}/smoke`);
export const autotuneModel = (name: string, objective: string) =>
  postJson<JobSubmitted>(`/api/v2/models/${enc(name)}/autotune`, { objective });
export const unloadAll = () =>
  postJson<JobSubmitted>("/api/v2/runtime/unload-all");

export type PlanAction = "select" | "enable" | "disable" | "test";
export const planAction = (name: string, planId: string, action: PlanAction) =>
  postJson<JobSubmitted>(
    `/api/v2/models/${enc(name)}/plans/${enc(planId)}/${action}`);

export const applyTier = (name: string, accept: boolean) =>
  postJson<TierApplyResult>(`/api/v2/models/${enc(name)}/tier/apply`,
                            { accept_tier_change: accept });

/* ---- placement ----
   The selection goes to the planner and the planner answers where the
   weights land. Preview and apply send the SAME selection, so the layout
   on screen is the layout the apply computes. */

export function placementQuery(selection: PlacementSelection): string {
  const q = new URLSearchParams();
  for (const [dev, choice] of Object.entries(selection)) {
    if (choice.on != null) q.set(`on.${dev}`, choice.on ? "1" : "0");
    if (choice.ceiling_bytes != null) {
      q.set(`ceiling.${dev}`, String(Math.round(choice.ceiling_bytes)));
    }
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

/* fresh re-reads the machine instead of the profile's recorded snapshot.
   On the GET it changes nothing on disk; on the POST it is what makes the
   apply record the machine the operator was just shown. */
export const fetchPlacement = (name: string, selection: PlacementSelection,
                               fresh = false) =>
  get<PlacementView>(
    `/api/v2/models/${enc(name)}/placement${placementQuery(selection)}`
    + (fresh ? (placementQuery(selection) ? "&" : "?") + "fresh=1" : ""));

export const applyPlacement = (name: string, selection: PlacementSelection,
                               accept: boolean, fresh = false) =>
  postJson<PlacementApplyResult>(`/api/v2/models/${enc(name)}/placement`,
                                 { selection, accept_tier_change: accept,
                                   fresh });

export const fetchRuntimePolicy = (name: string) =>
  get<RuntimePolicyView>(`/api/v2/models/${enc(name)}/runtime-policy`);
export const saveRuntimePolicy = (name: string, body: Record<string, unknown>) =>
  postJson<JobSubmitted & { mode: string }>(
    `/api/v2/models/${enc(name)}/runtime-policy`, body);

export const fetchRunCommand = (name: string) =>
  get<RunCommand>(`/api/v2/models/${enc(name)}/run-command`);

export const fetchJob = (id: string) => get<JobRow>(`/api/v2/jobs/${enc(id)}`);

/* Retry for a job that reported router_reloaded=false: re-runs the
   backend sync as its own job and answers with the new job's id. */
export const retrySync = () => postJson<{ job_id: string }>("/api/v2/sync");

/* ---- fleet ----
   The read opens no socket; refreshing presence is the explicit POST,
   which is why the page fires it on open rather than relying on the
   GET to be fresh. The budget write rides the same mutation entry every
   other phase-4 control does and answers with a job id. */

export const fetchFleet = () => get<FleetView>("/api/v2/fleet");
export const probeFleet = () => postJson<ProbeResult>("/api/v2/fleet/probe");
export const probeNode = (node: string) =>
  postJson<ProbeResult>(`/api/v2/fleet/nodes/${enc(node)}/probe`);
export const setNodeBudget = (node: string, device: string,
                              budget_bytes: number) =>
  postJson<BudgetSubmitted>(
    `/api/v2/fleet/nodes/${enc(node)}/devices/${enc(device)}/budget`,
    { budget_bytes });

export const fetchRouting = () => get<RoutingMatrix>("/api/v2/settings/routing");
export const applyRouting = () =>
  postJson<JobSubmitted>("/api/v2/settings/routing/apply");

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

/* ---- settings ---- */

export const fetchSettings = () => get<SettingsOverview>("/api/v2/settings");
export const saveDefaults = (updates: Record<string, string | number>) =>
  postJson<SaveResult>("/api/v2/settings/defaults", { updates });
export const saveHardware = (body: {
  devices?: Record<string, Record<string, unknown>>;
  ram?: { reserve_bytes: number };
}) => postJson<SaveResult>("/api/v2/settings/hardware", body);
export const reprobeCapabilities = () =>
  postJson<{ ok: boolean; message: string }>(
    "/api/v2/settings/capabilities/reprobe");
export const calibrateStorage = () =>
  postJson<{ job_id: string }>("/api/v2/settings/hardware/calibrate");
export async function cancelJob(id: string): Promise<CancelResult> {
  const r = await fetch(`/api/v2/jobs/${id}/cancel`, { method: "POST" });
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

/* Plain words for llama-swap worker states. The raw value stays in the
   API and filters; only what people read changes. */
export const stateLabel = (state: string): string =>
  (({
    ready: "running",
    loading: "starting",
    queued: "waiting to start",
    unloading: "stopping",
    stopped: "idle",
    failed: "failed",
    unregistered: "not serving",
  } as Record<string, string>)[state] ?? state);
