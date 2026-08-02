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
  /* section -> why it is missing. A section that threw server-side yields
     its empty shape, which is indistinguishable from the truth of a
     machine with no GPUs; this is how a region tells the two apart and
     degrades itself instead of rendering the fallback as fact. */
  errors?: Record<string, string>;
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

/* ---- phase 4: operations ---- */

/* Every phase-4 write answers with the job it queued, so the caller can
   name it in the toast and the jobs page can be deep-linked to it. */
export interface JobSubmitted {
  job_id: string;
}

export interface CacheResetResult {
  ok: boolean;
  name: string;
  port: number;
}

export interface TierApplyResult {
  job_id: string;
  gate: Gate;
}

export interface RuntimePolicy {
  mode: string;
  objective: string;
  pinned_plan_id: string | null;
  allow_fallback: boolean;
  allow_untested: boolean;
  minimum_context: number | null;
  maximum_cpu_bytes: number | null;
  maximum_storage_tier: number;
  disabled_plan_ids: string[];
}

/* The form's select is built from `objectives`, which is the same tuple
   the write validates against -- the console cannot offer an objective
   the endpoint rejects, or hide one it accepts. */
export interface RuntimePolicyView {
  policy: RuntimePolicy | null;
  objectives: string[];
  plans: { id: string; label: string }[];
}

export interface RunCommand {
  argv: string[];
  run_sh: string;
  command_fingerprint: string;
  resolved_binary: string;
  pinned_binary: string;
  ok: boolean;
  messages: string[];
  warnings: string[];
  error: string;
}

export interface RoutingRow {
  key: string;
  managed: boolean;
  before: string | null;
  after: string | null;
  change: "added" | "removed" | "changed" | "unchanged";
  evict_cost: number | null;
}

export interface RoutingMatrix {
  config_path: string;
  existing: Record<string, unknown>;
  generated: {
    vars: Record<string, string>;
    evict_costs: Record<string, number>;
    sets: Record<string, string>;
    claims: Record<string, Record<string, number>>;
    excluded: { name: string; reason: string }[];
  } | null;
  merged: Record<string, unknown> | null;
  preview: string;
  rows: RoutingRow[];
  errors: Record<string, string>;
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

/* ---- settings (phase 3) ---- */

export interface DefaultField {
  name: string;
  kind: "int" | "text" | "choice";
  min: number | null;
  max: number | null;
  unit: string;
  choices: string[];
}

export interface DefaultsSection {
  values: Record<string, string | number>;
  fields: DefaultField[];
  /* field -> the MODELCTL_DEFAULT_* var that overrides it in the running
     service, so the form can say why a save would not take effect */
  shadowed: Record<string, string>;
  error: string;
}

export interface HwDevice {
  device: string;
  name: string;
  total_bytes: number;
  free_bytes: number;
  reserve_bytes: number;
  enabled: boolean;
  role: string;
  bandwidth_gbs: number | null;
  bandwidth_overridden: boolean;
  pci_address: string;
  pcie_width: number | null;
}

export interface HwStorage {
  path: string;
  kind: string;
  mount_point: string;
  filesystem: string;
  transport: string;
  allow_mmap: boolean;
  total_bytes: number;
  free_bytes: number;
  measured_sequential_read_bps: number | null;
  measured_random_read_bps: number | null;
  measurement_age_seconds: number | null;
}

export interface HardwareSection {
  devices: HwDevice[];
  ram: { total_bytes: number; available_bytes: number; reserve_bytes: number };
  storage: HwStorage[];
  roles: string[];
  fingerprint: string;
  error: string;
}

export interface AccessSection {
  bind: string;
  bind_source: string;
  bind_editable: boolean;
  token_path: string;
  token_source: string;
  token_rotatable: boolean;
  secure_cookie: boolean;
}

export interface PathRow {
  label: string;
  value: string;
  source: string;
  overridden: boolean;
}

export interface DiagnosticsSection {
  manifest: {
    path?: string;
    present?: boolean;
    ok?: boolean;
    error?: string;
    modelctl_commit?: string;
    validated_modelctl_commit?: string;
    validated_llama_commit?: string;
    upstream_base?: string;
    validation_report?: string;
    newer_than_validated?: boolean;
    submodule_pinned?: string;
    submodule_checked_out?: string;
    working_tree_dirty?: boolean;
    mismatches?: string[];
    notes?: string[];
  };
  capabilities: {
    binary?: string;
    candidates?: string[];
    probe?: Record<string, unknown> | null;
    error?: string;
    capability_fingerprint?: string;
  };
  environment: {
    platform?: Record<string, string>;
    paths?: Record<string, string>;
    oneapi_env_scripts?: string[];
    benchmark_modes?: string[];
    modelctl_env?: Record<string, string>;
  };
  errors: string[];
}

export interface ReadinessCheck {
  id: string;
  title: string;
  severity: "ok" | "warning" | "error";
  detail: string;
  fix: string;
  fix_url: string;
  fix_command: string;
}

export interface ReadinessSection {
  ready: boolean;
  first_run: boolean;
  checks: ReadinessCheck[];
  error: string;
}

export interface SettingsOverview {
  readiness: ReadinessSection;
  defaults: DefaultsSection;
  hardware: HardwareSection;
  access: AccessSection;
  paths: PathRow[];
  diagnostics: DiagnosticsSection;
}

export interface SaveResult {
  ok: boolean;
  applied: Record<string, unknown> | number;
  values?: Record<string, string | number>;
  warnings: string[];
  messages: string[];
}

/* ---- fleet: the local node and the remote RPC nodes, one shape ----
   Presence is tri-state and not a boolean on purpose: PIN_MISMATCH is a
   node that answers a handshake immediately while being unusable, and
   collapsing it into "up" is exactly the render this view exists to
   prevent (see modelctl_web/fleet.py). */

export type PresenceState = "PRESENT" | "STALE" | "PIN_MISMATCH";

export interface FleetDeviceRow {
  name: string;
  kind: string;
  label?: string;
  budget_bytes: number;
  total_bytes: number;
  cap_bytes: number;
  ceiling_bytes: number;
  ceiling_basis: string;
  admission_key: string;
  editable: boolean;
  edit_note: string;
}

export interface FleetNodeRow {
  name: string;
  location: "local" | "remote";
  host: string;
  port: number | null;
  endpoint: string;
  variant: string;
  enabled: boolean;
  note: string;
  pin: { node: string; expected: string; agrees: boolean };
  presence: {
    state: PresenceState;
    detail: string;
    reachable: boolean;
    protocol: string;
    probed_at: number | null;
    ttl_seconds: number;
  };
  devices: FleetDeviceRow[];
}

export interface NightLaneRow {
  id: string;
  title: string;
  enabled: boolean;
  mode: string;
  registered: string;
  requires_nodes: string[];
  arms: { name: string; profile: string; requires_nodes: string[] }[];
}

export interface StaleProfileRow {
  name: string;
  changed: Record<string, { recorded: number | null; live: number | null }>;
}

export interface FleetView {
  nodes: FleetNodeRow[];
  night_lane: NightLaneRow[];
  stale_profiles: StaleProfileRow[];
  local_pin: string;
  errors: Record<string, string>;
}

export interface ProbeResult {
  probed: {
    node: string;
    endpoint: string;
    reachable: boolean;
    protocol: string;
    pin: string;
    pin_agrees: boolean;
    detail: string;
    probed_at: number;
  }[];
}

export interface BudgetSubmitted {
  job_id: string;
  budget_bytes: number;
  ceiling_bytes: number;
}
