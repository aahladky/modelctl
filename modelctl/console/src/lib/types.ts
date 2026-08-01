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
