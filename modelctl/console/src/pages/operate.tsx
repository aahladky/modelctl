/* overview: organized around "is anything running, and is it healthy."
   Active workloads first (running models as cards, in-flight loads as a
   persistent pipeline wired to the real job), hardware second, the
   library of stopped profiles last, behind search. Everything renders
   from the tick stream; when the stream drops, every live region flips
   to the stale treatment instead of blanking. */
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import type { ComponentChildren } from "preact";
import { useStream } from "../lib/stream";
import { Spark, push } from "../lib/spark";
import { Info } from "../lib/info";
import { Meter } from "../lib/meter";
import { ConfirmButton, submitAction } from "../lib/actions";
import { cancelJob, fmtAgo, fmtClock, fmtGiB, fmtUp, loadModel, probeFleet,
  unloadAll, unloadModel,
  stateLabel } from "../lib/api";
import { EmptyState, Pipeline, PlacementChips, SectionHead, StatBlock }
  from "../lib/ui";
import { toast } from "../lib/toasts";
import type { JobRow, ModelRow, NodeStatRow, NodeStats, Tick }
  from "../lib/types";

type Series = Record<string, number[]>;

function useSeries(tick: Tick | null): Series {
  const ref = useRef<Series>({});
  const [, bump] = useState(0);
  useEffect(() => {
    if (!tick) return;
    const s = ref.current;
    for (const g of tick.gpus) {
      s[`gpu:${g.device}`] = push(s[`gpu:${g.device}`] ?? [], g.used_bytes / 2 ** 30);
    }
    s["ram"] = push(s["ram"] ?? [], tick.ram.used_bytes / 2 ** 30);
    for (const m of tick.models) {
      if (m.tok_s != null) s[`tok:${m.name}`] = push(s[`tok:${m.name}`] ?? [], m.tok_s);
      if (m.cache?.hit_ratio != null) {
        s[`hit:${m.name}`] = push(s[`hit:${m.name}`] ?? [], m.cache.hit_ratio * 100);
      }
    }
    bump((n) => n + 1);
  }, [tick]);
  return ref.current;
}

/* A region marks itself only when its data is not trustworthy. The
   header's global badge already says the stream is live; a healthy
   region repeating it is noise, so it renders nothing. */
function LiveMark({ stale }: { stale: boolean }) {
  return stale
    ? <span class="live stale"><span class="dot"></span>stale</span>
    : null;
}

function StaleSub({ lastAt, retryIn }: { lastAt: number | null; retryIn: number | null }) {
  return (
    <div class="sub">
      last known {lastAt != null ? fmtClock(lastAt) : "—"} ·{" "}
      <span>{retryIn != null ? `retrying in ${retryIn}s` : "reconnecting…"}</span>
    </div>
  );
}

const DASH = "—";

/* One remote node inside the fleet card. Every number here can be
   absent, and absent renders as an em dash — never 0. */
function NodeBlock({ n, first }: { n: NodeStatRow; first: boolean }) {
  const sep = first
    ? undefined
    : "border-top:1px solid var(--border);margin-top:.7rem;padding-top:.7rem";
  const gpu = n.kind === "gpu";
  const title = (
    <span>{n.name} <span class="sub">{gpu ? n.device : "cpu"}</span></span>
  );

  if (!n.polled) {
    return (
      <div style={sep}>
        <div class="label">
          {title}
          <span class="num">{fmtGiB(n.budget_bytes)} GiB budget</span>
        </div>
        <div class="sub">
          {n.present ? "present" : "not a planning target"} · no ssh host
          recorded — presence and budget only
        </div>
      </div>
    );
  }

  const value = gpu ? n.gpu_used_bytes : n.unit_memory_bytes;
  const max = gpu ? n.gpu_total_bytes : n.unit_memory_max_bytes;
  const what = gpu ? "VRAM" : "cgroup memory";
  const subLine = gpu
    ? ["vram",
       `gpu ${n.gpu_util_pct == null ? DASH : `${n.gpu_util_pct}%`}`,
       `${n.gpu_temp_c == null ? DASH : n.gpu_temp_c} C`]
    : ["cgroup",
       `load ${n.host_load1 == null ? DASH : n.host_load1.toFixed(2)}`,
       `${n.host_nproc == null ? DASH : n.host_nproc} threads`];
  const secondLine = gpu
    ? `unit ${fmtGiB(n.unit_memory_bytes, 2)} / `
      + `${fmtGiB(n.unit_memory_max_bytes, 0)} GiB cap`
    : `host ${fmtGiB(n.host_mem_available_bytes)} / `
      + `${fmtGiB(n.host_mem_total_bytes, 0)} GiB available`;

  return (
    <div style={sep}>
      <div class="label">
        {title}
        <span class="num">{fmtGiB(value)} / {fmtGiB(max, 0)} GiB</span>
      </div>
      <Meter value={value} max={max} label={`${n.name} ${what}`}
             valuetext={`${fmtGiB(value)} of ${fmtGiB(max, 0)} GiB `
                        + `${what} on ${n.name}`} />
      <div class="sub">{subLine.join(" · ")}</div>
      <div class="sub">
        {secondLine}
        {n.present ? "" : " · not a planning target"}
      </div>
    </div>
  );
}

function RemoteFleet({ ns, stale, note }:
                     { ns: NodeStats; stale: boolean; note: ComponentChildren }) {
  if (ns.nodes.length === 0) return null;
  const hosts = [...new Set(ns.nodes.map((n) => n.host).filter(Boolean))];
  return (
    <div class={stale ? "widget stale" : "widget"}>
      <div class="label">
        <span>remote fleet <span class="sub">{hosts.join(" · ") || DASH}</span></span>
        <LiveMark stale={stale} />
      </div>
      <div class={ns.pins_agree ? "sub" : "msg error"} style="margin-top:.3rem">
        {ns.present} present · {ns.pins_agree
          ? "versions match"
          : "VERSION MISMATCH — rebuild the node to use it"} · RPC {ns.protocol || DASH}
      </div>
      {ns.nodes.map((n, i) => (
        <NodeBlock key={n.name} n={n} first={i === 0} />
      ))}
      {note}
    </div>
  );
}

function UnloadAll({ running, stale }: { running: number; stale: boolean }) {
  const [busy, setBusy] = useState(false);
  if (running === 0) return null;
  return (
    <ConfirmButton
      label="unload all"
      confirmLabel={`yes, unload ${running}`}
      busy={busy}
      disabled={stale}
      consequences={<>
        Unloads every resident model ({running} running). In-flight
        requests to them fail; the next request reloads from disk.
      </>}
      onConfirm={() => {
        setBusy(true);
        submitAction(unloadAll, "unload all").finally(() => setBusy(false));
      }} />
  );
}

/* ---- active workloads ------------------------------------------------ */

/* The newest load/restart job for a model, by title match: the tick's
   job rows don't carry a payload, but every runtime-lane title is
   "<verb> <name>" by construction (modelctl_web/mutate.py). */
function loadJobFor(jobs: JobRow[], name: string): JobRow | null {
  const mine = jobs.filter((j) =>
    (j.type === "load" || j.type === "restart")
    && j.title === `${j.type} ${name}`);
  if (mine.length === 0) return null;
  return mine.reduce((a, b) => (b.created > a.created ? b : a));
}

function fmtElapsed(started: number | null): string {
  if (!started) return DASH;
  const s = Math.max(0, Math.round(Date.now() / 1000 - started));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function RunningCard({ m, spark, stale }:
                     { m: ModelRow; spark: number[]; stale: boolean }) {
  const [busy, setBusy] = useState(false);
  const hit = m.cache?.hit_ratio;
  return (
    <div class={stale ? "widget stale" : "widget"}
         style="border-left:3px solid var(--ok)">
      <div style="display:flex;gap:1.2rem;align-items:center;flex-wrap:wrap">
        <div style="flex:1;min-width:220px">
          <h3 style="font-size:1.02rem;margin:0">
            <a href={`/v2/models/${encodeURIComponent(m.name)}`}>{m.name}</a>
            <span style={`color:var(--${m.state_class === "ok" || m.running ? "ok" : "muted"});font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;margin-left:.6em`}>
              {stateLabel(m.state)}
            </span>
          </h3>
          <div class="sub num" style="margin:.15rem 0 .4rem">
            {[m.port ? `:${m.port}` : "",
              m.size_bytes != null ? `${fmtGiB(m.size_bytes)} GiB` : "",
              m.backend !== "llama-cpp" ? m.backend : ""]
              .filter(Boolean).join(" · ") || m.file}
          </div>
          <PlacementChips summary={m.placement_summary} fallback={m.placement} />
        </div>
        <div class="stats">
          <StatBlock value={m.tok_s != null ? m.tok_s.toFixed(1) : DASH}
                     unit="tok/s" />
          {hit != null && (
            <StatBlock value={(hit * 100).toFixed(1)} unit="% cache hit" />
          )}
          {spark.length > 1 && (
            <div style="width:120px;align-self:center">
              <Spark data={spark} height={30} label={`${m.name} tok/s`} />
            </div>
          )}
        </div>
        <div class="actions">
          <button type="button" class={busy ? "btn-danger busy" : "btn-danger"}
                  disabled={busy || stale}
                  aria-label={`unload ${m.name}`}
                  onClick={() => {
                    setBusy(true);
                    submitAction(() => unloadModel(m.name), `unload ${m.name}`)
                      .finally(() => setBusy(false));
                  }}>unload</button>
        </div>
      </div>
    </div>
  );
}

/* An in-flight load as a persistent operation, not a button and a toast.
   The stages are only the ones the backend truly has: llama-swap's warm
   load is one opaque call, so the pipe is queued → loading → ready and
   the card leans on elapsed time, the job link, and its log — not on
   invented percentages. */
function LoadingCard({ m, job, stale }:
                     { m: ModelRow; job: JobRow; stale: boolean }) {
  const [busy, setBusy] = useState(false);
  const at = job.status === "queued" ? 0 : 1;
  const cancel = async () => {
    setBusy(true);
    try {
      const res = await cancelJob(job.id);
      if (!res.cancelled) {
        toast("err", `✗ cancel failed — ${job.title}`,
              res.reason || `job is ${res.status}`, 9000);
      } else {
        toast("ok", `${job.title} cancelled`, "");
      }
    } catch (e) {
      toast("err", `✗ cancel failed — ${job.title}`, String(e), 9000);
    } finally {
      setBusy(false);
    }
  };
  return (
    <div class={stale ? "widget stale" : "widget"}
         style="border-left:3px solid var(--accent)">
      <div style="display:flex;gap:1.2rem;align-items:center;flex-wrap:wrap">
        <div style="flex:1;min-width:240px">
          <h3 style="font-size:1.02rem;margin:0">
            <a href={`/v2/models/${encodeURIComponent(m.name)}`}>{m.name}</a>
            <span style="color:var(--accent);font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;margin-left:.6em">
              {job.type === "restart" ? "restarting" : "loading"}
            </span>
          </h3>
          <div class="sub num" style="margin:.15rem 0 0">
            {[m.size_bytes != null ? `${fmtGiB(m.size_bytes)} GiB` : "", m.file]
              .filter(Boolean).join(" · ")}
            {" · job "}
            <a href={`/v2/jobs/${encodeURIComponent(job.id)}`}>
              {job.id.slice(0, 8)}
            </a>
          </div>
          <Pipeline stages={["queued", "loading", "ready"]} at={at}
                    label={`${m.name} ${job.status === "queued"
                      ? "queued to load" : "loading"}`} />
          {job.detail && <div class="sub">{job.detail}</div>}
        </div>
        <div class="stats">
          <StatBlock value={fmtElapsed(job.started)} unit="elapsed" />
        </div>
        <div class="actions" style="flex-direction:column;align-items:stretch">
          <a class="btn" style="text-align:center"
             href={`/v2/jobs/${encodeURIComponent(job.id)}`}>logs</a>
          {job.cancellable && (
            <button type="button" class={busy ? "btn-danger busy" : "btn-danger"}
                    disabled={busy || stale} onClick={cancel}>cancel</button>
          )}
        </div>
      </div>
    </div>
  );
}

/* A failed load stays on the page in the error treatment with a retry —
   a toast that evaporated was the whole complaint. It clears when a
   newer load job replaces it or the model comes up. */
function FailedLoadCard({ m, job, stale }:
                        { m: ModelRow; job: JobRow; stale: boolean }) {
  const [busy, setBusy] = useState(false);
  return (
    <div class="widget" style="border-left:3px solid var(--err)">
      <div style="display:flex;gap:1.2rem;align-items:center;flex-wrap:wrap">
        <div style="flex:1;min-width:240px">
          <h3 style="font-size:1.02rem;margin:0">
            <a href={`/v2/models/${encodeURIComponent(m.name)}`}>{m.name}</a>
            <span style="color:var(--err);font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;margin-left:.6em">
              load failed
            </span>
          </h3>
          <div class="msg error" style="margin:.25rem 0 0">
            {(job.error || "load failed").split("\n")[0]}
          </div>
          <div class="sub num">
            {job.finished ? `${fmtAgo(job.finished)} · ` : ""}
            job <a href={`/v2/jobs/${encodeURIComponent(job.id)}`}>
              {job.id.slice(0, 8)}</a> has the full log
          </div>
        </div>
        <div class="actions">
          <button type="button" class={busy ? "busy" : undefined}
                  disabled={busy || stale}
                  onClick={() => {
                    setBusy(true);
                    submitAction(() => loadModel(m.name), `load ${m.name}`)
                      .finally(() => setBusy(false));
                  }}>retry load</button>
        </div>
      </div>
    </div>
  );
}

/* ---- library --------------------------------------------------------- */

function LibraryRow({ m, spark, stale, loadBlocked }: {
  m: ModelRow; spark: number[]; stale: boolean; loadBlocked: boolean;
}) {
  const [busy, setBusy] = useState(false);
  const chipClass = m.state_class ? `chip ${m.state_class}` : "chip";
  return (
    <tr class="rowhover">
      <td>
        <a href={`/v2/models/${encodeURIComponent(m.name)}`}>{m.name}</a>
        <div class="sub">
          {[m.file, m.backend !== "llama-cpp" ? m.backend : ""]
            .filter(Boolean).join(" · ")}
        </div>
      </td>
      <td class="opt">
        <PlacementChips summary={m.placement_summary} fallback={m.placement} />
      </td>
      <td class={m.size_bytes != null ? "num" : "num sub"}>
        {m.size_bytes != null ? `${fmtGiB(m.size_bytes)} GiB` : DASH}
      </td>
      <td class="opt">
        {m.state !== "stopped" || !m.enabled
          ? <span class={chipClass}><span class="dot"></span>
              {m.enabled ? m.state : "disabled"}</span>
          : null}
        {spark.length > 1
          ? <Spark data={spark} height={22} label={`${m.name} tok/s`} />
          : null}
      </td>
      <td style="width:90px">
        <div class="actions">
          {m.registered && m.enabled && !m.running
            ? <button type="button" class={busy ? "busy" : undefined}
                      disabled={busy || stale || loadBlocked}
                      aria-label={`load ${m.name}`}
                      onClick={() => {
                        setBusy(true);
                        submitAction(() => loadModel(m.name), `load ${m.name}`)
                          .finally(() => setBusy(false));
                      }}>load</button>
            : null}
        </div>
      </td>
    </tr>
  );
}

export function Operate() {
  const { tick, stale, lastAt, retryIn } = useStream();
  const series = useSeries(stale ? null : tick);
  const [q, setQ] = useState("");
  const [backend, setBackend] = useState("all");
  const [sort, setSort] = useState<"name" | "size">("name");

  /* A person opening the page is the probe trigger — same rule as the
     fleet page, so presence doesn't decay mid-glance. */
  useEffect(() => { probeFleet().catch(() => undefined); }, []);

  const active = useMemo(() => {
    if (!tick) return { running: [] as ModelRow[], loading: [] as
      { m: ModelRow; job: JobRow }[], failed: [] as { m: ModelRow; job: JobRow }[] };
    const running: ModelRow[] = [];
    const loading: { m: ModelRow; job: JobRow }[] = [];
    const failed: { m: ModelRow; job: JobRow }[] = [];
    for (const m of tick.models) {
      const job = loadJobFor(tick.jobs, m.name);
      if (m.running) {
        running.push(m);
      } else if (job && (job.status === "running" || job.status === "queued")) {
        loading.push({ m, job });
      } else if (job && job.status === "failed") {
        failed.push({ m, job });
      }
    }
    return { running, loading, failed };
  }, [tick]);

  if (!tick) {
    return (
      <div class="widget">
        <span class="sub">
          {stale
            ? <>stream unavailable · {retryIn != null ? `retrying in ${retryIn}s` : "reconnecting…"}</>
            : "connecting to telemetry…"}
        </span>
      </div>
    );
  }

  const { services, gpus, ram, models, jobs } = tick;
  const runningJobs = jobs.filter((j) => j.status === "running").length;
  const cacheModels = models.filter((m) => m.cache != null);
  const nodeStats = tick.node_stats
    ?? { nodes: [], age_seconds: null, ok: false, present: 0,
         pins_agree: true, protocol: "" };
  const err = tick.errors ?? {};
  const bad = (section: string) => stale || section in err;
  const cls = (section: string) => (bad(section) ? "widget stale" : "widget");
  const sectionNote = (section: string) =>
    err[section]
      ? <div class="sub stale-note">last reading is stale — {section} probe
          failed: {err[section]}</div>
      : stale
        ? <StaleSub lastAt={lastAt} retryIn={retryIn} />
        : null;

  /* the library is everything not currently active on the page above */
  const activeNames = new Set([
    ...active.running.map((m) => m.name),
    ...active.loading.map((x) => x.m.name),
  ]);
  const library = models
    .filter((m) => !activeNames.has(m.name))
    .filter((m) => backend === "all" || m.backend === backend)
    .filter((m) => {
      const needle = q.trim().toLowerCase();
      if (!needle) return true;
      return m.name.toLowerCase().includes(needle)
        || m.file.toLowerCase().includes(needle);
    })
    .sort((a, b) => sort === "size"
      ? (b.size_bytes ?? -1) - (a.size_bytes ?? -1)
      : a.name.localeCompare(b.name));
  const backends = [...new Set(models.map((m) => m.backend))].sort();
  const libraryTotal = models.length - activeNames.size;
  const nActive = active.running.length + active.loading.length;

  return (
    <>
      <div class="widget" style="display:flex;gap:1.2rem;align-items:center;flex-wrap:wrap">
        <span class={services.swap.ok ? "chip ok" : "chip err"}>
          <span class="dot"></span>
          model router · {services.swap.ok ? "running" : "not reachable"}
          {!services.swap.ok && services.swap.detail ? ` · ${services.swap.detail}` : ""}
          {services.swap.ok && services.swap.latency_ms != null ? ` · ${services.swap.latency_ms}ms` : ""}
        </span>
        <span class={services.api.ok ? "chip ok" : "chip err"}>
          <span class="dot"></span>
          console API · {services.api.ok ? "ok" : "down"}
          {services.api.latency_ms != null ? ` · ${services.api.latency_ms}ms` : ""}
        </span>
        {stale && (
          <span class="chip warn"><span class="dot"></span>telemetry · stream dropped</span>
        )}
        {Object.keys(err).map((section) => (
          <span class="chip warn" key={section}><span class="dot"></span>
            {section} · probe failed
          </span>
        ))}
        <span class="sub num">console restarted {fmtUp(services.console_started)} ago</span>
        <span class="grow" style="flex:1"></span>
        <a href="/v2/jobs" class="sub">
          {runningJobs} job{runningJobs === 1 ? "" : "s"} running →
        </a>
      </div>

      <SectionHead title="active workloads">
        {nActive} of {models.length} models
        {active.running.length > 1 && <>
          {" · "}
          <UnloadAll running={active.running.length} stale={stale} />
        </>}
      </SectionHead>
      {active.running.map((m) => (
        <RunningCard key={m.name} m={m}
                     spark={series[`tok:${m.name}`] ?? []} stale={stale} />
      ))}
      {active.loading.map(({ m, job }) => (
        <LoadingCard key={m.name} m={m} job={job} stale={stale} />
      ))}
      {active.failed.map(({ m, job }) => (
        <FailedLoadCard key={m.name} m={m} job={job} stale={stale} />
      ))}
      {nActive === 0 && active.failed.length === 0 && (
        <EmptyState>
          nothing is running — load a model from the library below
        </EmptyState>
      )}

      <SectionHead title="hardware" />
      <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(210px,1fr))">
        {gpus.length === 0 && (
          <div class={cls("gpus")}>
            <div class="label"><span>GPUs</span><LiveMark stale={bad("gpus")} /></div>
            <div class="sub" style="margin-top:.4rem">
              {err["gpus"]
                ? <span class="stale-note">device inventory unavailable —
                    the numbers that were here are gone, not zero</span>
                : "no GPUs reported by the backend"}
            </div>
          </div>
        )}
        {gpus.map((g) => {
          const used = g.used_bytes / 2 ** 30;
          const total = g.total_bytes / 2 ** 30;
          return (
            <div class={cls("gpus")} key={g.device}>
              <div class="label">
                <span>{g.name} <span class="sub num">{g.device}</span></span>
                <LiveMark stale={bad("gpus")} />
              </div>
              <Meter value={used} max={total} label={`${g.device} VRAM`}
                     valuetext={`${used.toFixed(1)} of ${total.toFixed(0)} GiB `
                                + `VRAM on ${g.device}`} />
              <div class="big">{used.toFixed(1)}<span class="sub"> / {total.toFixed(0)} GiB VRAM</span></div>
              <Spark data={series[`gpu:${g.device}`] ?? []} min={0} max={total}
                     label={`${g.device} VRAM used`} />
              {sectionNote("gpus")}
            </div>
          );
        })}

        <div class={cls("ram")}>
          <div class="label">
            <span>system RAM</span>
            <LiveMark stale={bad("ram")} />
          </div>
          <Meter value={err["ram"] ? null : ram.used_bytes}
                 max={err["ram"] ? null : ram.total_bytes}
                 label="system RAM"
                 valuetext={err["ram"]
                   ? "system RAM reading unavailable"
                   : `${fmtGiB(ram.used_bytes)} of `
                     + `${fmtGiB(ram.total_bytes, 0)} GiB system RAM used`} />
          <div class="big">
            {err["ram"] ? "—" : fmtGiB(ram.used_bytes)}
            <span class="sub"> / {err["ram"] ? "—" : fmtGiB(ram.total_bytes, 0)} GiB</span>
          </div>
          <Spark data={series["ram"] ?? []} min={0} max={ram.total_bytes / 2 ** 30}
                 label="RAM used" />
          {sectionNote("ram")}
        </div>

        <RemoteFleet ns={nodeStats}
                     stale={bad("node_stats") || !nodeStats.ok}
                     note={sectionNote("node_stats")} />

        {cacheModels.map((m) => {
          const c = m.cache!;
          const pct = c.hit_ratio != null ? c.hit_ratio * 100 : null;
          return (
            <div class={cls("models")} key={`cache-${m.name}`}>
              <div class="label">
                <span>
                  {m.name} · expert cache{" "}
                  <Info label="about the expert cache">
                    Expert-cache hit ratio for this model, scraped from its
                    runtime metrics. While the cache is still learning which
                    experts recur, the ratio is provisional; "learning done"
                    means placement has settled.
                  </Info>
                </span>
                <LiveMark stale={bad("models")} />
              </div>
              <Meter value={pct} max={pct == null ? null : 100}
                     label={`${m.name} expert cache hit ratio`}
                     valuetext={pct == null
                       ? `${m.name} cache hit ratio not reported`
                       : `${pct.toFixed(1)}% cache hit ratio for ${m.name}`} />
              <div class="big">
                {pct != null ? pct.toFixed(1) : DASH}
                <span class="sub"> % hit · {c.learning === null
                  ? "learning state unknown"
                  : c.learning ? "still learning" : "learning done"}</span>
              </div>
              <Spark data={series[`hit:${m.name}`] ?? []} min={0} max={100}
                     label={`${m.name} cache hit ratio`} />
              {sectionNote("models")}
            </div>
          );
        })}
      </div>
      {cacheModels.length === 0 && (
        <EmptyState>
          no expert cache active — cache meters appear here when a
          cache-enabled model runs
        </EmptyState>
      )}

      <SectionHead title="model library">
        {libraryTotal} profile{libraryTotal === 1 ? "" : "s"}
      </SectionHead>
      <div class={cls("models")}>
        <div class="frow" style="align-items:center;margin-bottom:.5rem">
          <input type="search" placeholder="search name or file…"
                 aria-label="search models" value={q}
                 style="flex:1 1 200px;max-width:320px"
                 onInput={(e) => setQ((e.target as HTMLInputElement).value)} />
          {backends.length > 1 && (
            <select aria-label="backend filter" style="width:auto"
                    value={backend}
                    onChange={(e) => setBackend((e.target as HTMLSelectElement).value)}>
              <option value="all">all backends</option>
              {backends.map((b) => <option key={b} value={b}>{b}</option>)}
            </select>
          )}
          <select aria-label="sort" style="width:auto" value={sort}
                  onChange={(e) => setSort((e.target as HTMLSelectElement).value as "name" | "size")}>
            <option value="name">by name</option>
            <option value="size">by size</option>
          </select>
          <span class="sub" style="margin-left:auto">
            {library.length === libraryTotal
              ? `${libraryTotal}`
              : `${library.length} of ${libraryTotal}`}
          </span>
        </div>
        {models.length === 0
          ? (err["models"]
              ? <p class="sub stale-note">could not read the profile store —
                  {" "}{err["models"]}. This is not an empty install.</p>
              : <p class="sub">no models yet — <a href="/v2/add">add one</a></p>)
          : library.length === 0
            ? <EmptyState>nothing matches</EmptyState>
            : (
              <div class="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>model</th><th class="opt">placement</th>
                      <th class="num">size</th>
                      <th class="opt"><span class="sr-only">state</span></th>
                      <th><span class="sr-only">actions</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    {library.map((m) => (
                      <LibraryRow key={m.name} m={m}
                                  spark={series[`tok:${m.name}`] ?? []}
                                  stale={stale}
                                  loadBlocked={active.loading.some((x) =>
                                    x.m.name === m.name)} />
                    ))}
                  </tbody>
                </table>
              </div>
            )}
        <p class="sub" style="margin:.5rem 0 0">
          each model name links to its detail page ·{" "}
          <a href="/v2/models">measurement tools live on models →</a>
        </p>
        {sectionNote("models")}
      </div>
    </>
  );
}
