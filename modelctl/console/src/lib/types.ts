/* Typed mirror of the /api/v2 JSON surface (modelctl_web/telemetry.py).
   Every endpoint the SPA touches is described here -- if the backend
   shape changes, this file is where the compiler starts complaining. */

export interface ServiceStatus {
  ok: boolean;
  latency_ms: number | null;
  detail: string;
}

export interface Services {
  swap: ServiceStatus;
  api: ServiceStatus;
  console_started: number;
}

export interface GpuRow {
  device: string;
  name: string;
  total_bytes: number;
  free_bytes: number;
  used_bytes: number;
}

export interface RamRow {
  total_bytes: number;
  available_bytes: number;
  used_bytes: number;
}

export interface CacheStats {
  hit_ratio: number | null;
  learning: boolean | null;
  hits: number;
  misses: number;
  devices: string[];
}

export interface ModelRow {
  name: string;
  file: string;
  backend: string;
  placement: string;
  state: string;
  state_class: string;
  registered: boolean;
  running: boolean;
  enabled: boolean;
  port: number | null;
  size_bytes: number | null;
  moe_cache_mode: string;
  tok_s: number | null;
  tok_s_avg: number | null;
  cache: CacheStats | null;
}

export type JobStatus =
  | "queued"
  | "running"
  | "done"
  | "failed"
  | "cancelled"
  | "interrupted";

export interface JobRow {
  id: string;
  type: string;
  title: string;
  lane: string;
  status: JobStatus;
  progress: number;
  detail: string;
  error: string;
  result_tail: string;
  cancellable: boolean;
  created: number;
  started: number | null;
  finished: number | null;
}

export interface Tick {
  ts: number;
  services: Services;
  gpus: GpuRow[];
  ram: RamRow;
  models: ModelRow[];
  jobs: JobRow[];
}

export interface CancelResult {
  id: string;
  status: JobStatus;
  cancelled: boolean;
  reason: string | null;
}

/* ---- phase 2: model hub ---- */

export interface ModelConfig {
  device: string;
  split_mode: string;
  tensor_split: string;
  ctx: number | string;
  cache_type_k: string;
  cache_type_v: string;
  flash_attn: string;
  ttl: number | string;
  mtp: string;
  fit: string;
  extra: string;
  binary: string;
}

export interface ModelDetail {
  name: string;
  repo_id: string;
  file: string;
  model_path: string;
  backend: string;
  binary: string;
  enabled: boolean;
  size_bytes: number | null;
  config: ModelConfig;
  moe_cache: { mode: string; budgets_bytes: Record<string, number> };
  runtime: {
    state: string;
    state_class: string;
    running: boolean;
    registered: boolean;
    port: number | null;
    pid: number | null;
    started: number | null;
  };
  planning: { recorded_at: string; has_stored_inputs: boolean };
  gpus: { device: string; name: string; total_bytes: number }[];
}

export interface PlanMeasured {
  generation_tps?: number;
  prompt_tps?: number;
  load_seconds?: number;
  ttft_seconds?: number;
  actual_context?: number;
  cache_state: string;
  measured_at: number | null;
  stale: boolean;
}

export interface PlanRow {
  id: string;
  label: string;
  source: string;
  category: string;
  category_label: string;
  tested: boolean;
  stale: boolean;
  pinned: boolean;
  disabled: boolean;
  estimated: Record<string, unknown>;
  measured: PlanMeasured | null;
  warnings: string[];
  reason: string;
  admission: AdmissionRecord | null;
}

export interface HistoryRow {
  started_at: number | null;
  plan_id: string;
  run_kind: string;
  success: boolean;
  failure_class: string;
  generation_tps: number | null;
  prompt_tps: number | null;
  load_seconds: number | null;
  ttft_seconds: number | null;
  actual_context: number | null;
  cache_state: string;
  bottleneck: string;
  bottleneck_why: string;
}

export interface LogTail {
  source: string;
  tail: string;
  error: string;
}

/* The tier planner's admission record (plan["admission"]): requested /
   assumed / chosen plus per-device math and the degradation trail. */
export interface AdmissionDevice {
  capacity_bytes: number;
  usable_bytes: number;
  weights_bytes: number;
  kv_bytes: number;
  overhead_bytes: number;
  pinned_expert_bytes: number;
  cache_bytes: number;
  compute_reserve_bytes: number;
  demand_bytes: number;
  fits: boolean;
}

export interface AdmissionRecord {
  fits: boolean;
  devices?: Record<string, AdmissionDevice>;
  requested: Record<string, unknown>;
  assumed: Record<string, unknown>;
  chosen: Record<string, unknown>;
  degradations: {
    action: string;
    device: string;
    requested_bytes?: number;
    chosen_bytes?: number;
    reason?: string;
  }[];
}

export interface TierPlan {
  tier: number;
  config: { device: string; split_mode: string; tensor_split: string; extra: string };
  layout: [string, number, string][];
  warnings: string[];
  analysis: Record<string, unknown>;
  cache_budgets: Record<string, number> | null;
  admission: AdmissionRecord;
}

export interface Gate {
  kind: "none" | "pins" | "structural";
  changes: string[];
  requires_accept: boolean;
}

export interface AdmissionPreview {
  plan: TierPlan | null;
  planning_inputs: Record<string, unknown> | null;
  planning_inputs_source: string;
  gate: Gate | null;
  error?: string;
}

export interface ConfigSaveResult {
  jobs: { config: string | null; moe_cache: string | null };
  gate: Gate;
}

/* ---- phase 2: add wizard ---- */

export type WizardStep =
  | "source" | "inspect" | "download" | "analyze"
  | "plans" | "test" | "register" | "done";

export interface StepOutcome {
  ok: boolean;
  job_id: string;
  status: string;
  messages: string[];
  warnings: string[];
  error: string;
  at: number;
}

export interface WizardDetail {
  wizard_id: string;
  step: WizardStep;
  steps: WizardStep[];
  source_type: string;
  repo_id: string;
  local_path: string;
  selected_quant: string;
  download_job_id: string;
  download_complete: boolean;
  analysis: GgufAnalysis | Record<string, never>;
  selected_plan_id: string;
  test_job_id: string;
  test_observations: Record<string, Record<string, number | string>>;
  profile_name: string;
  registration_complete: boolean;
  endpoint: string;
  registration_error: string;
  source_verification: {
    ok?: boolean; messages?: string[]; warnings?: string[];
    data?: Record<string, unknown>;
  };
  command_fingerprint: string;
  measured: Record<string, number | string>;
  step_outcomes: Record<string, StepOutcome>;
  errors: { time: number; message: string }[];
  step_gates: Record<string, { blocking_reason: string; outcome: Partial<StepOutcome> }>;
  jobs: Record<string, JobRow>;
  created_at: number;
  updated_at: number;
}

export interface WizardSummary {
  wizard_id: string;
  step: WizardStep;
  source_type: string;
  repo_id: string;
  local_path: string;
  profile_name: string;
  updated_at: number;
  created_at: number;
}

export interface GgufAnalysis {
  arch: string;
  name: string;
  model_max_ctx: number | null;
  block_count: number | null;
  embedding_length: number | null;
  expert_count: number | null;
  is_moe: boolean;
  weight_bytes: number | null;
  kv_bytes_per_token: number | null;
}

export interface RepoContents {
  quant_groups: { label: string; files: string[]; sharded: boolean; total_size: number }[];
  mmproj_files: { name: string; size: number }[];
  mtp_files: { name: string; size: number }[];
}

export interface SearchResult {
  repo_id: string;
  downloads?: number;
  likes?: number;
  is_gguf?: boolean;
}

export interface RegisterData {
  profile: ModelDetail;
  analysis: GgufAnalysis | null;
  admission: AdmissionPreview;
  measured: Record<string, number | string>;
  selected_plan_id: string;
  test_gate: { blocking_reason: string; outcome: Partial<StepOutcome> };
}
