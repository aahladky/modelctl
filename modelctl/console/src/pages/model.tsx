/* per-model page: overview + placement, compiled plans with
   measured/estimated tags, full measurement history from the store, log
   tail, and the typed configure form. The form's save path fetches the
   planner's admission preview (requested / assumed / chosen + warnings,
   the server's own wording) and surfaces the structural-change gate as
   an explicit confirm -- never a silent rewrite. */
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { useStream } from "../lib/stream";
import {
  ApiError, applyTier, autotuneModel, benchModel, deleteModel,
  fetchAdmission, fetchLogTail, fetchModelDetail, fetchModelHistory,
  fetchModelPlans, fetchRunCommand, fetchRuntimePolicy, fmtAgo, fmtClock,
  fmtGiB, planAction, resetModelCache, restartModel, saveModelConfig,
  saveRuntimePolicy, smokeModel,
} from "../lib/api";
import type { PlanAction } from "../lib/api";
import { ConfirmButton, submitAction } from "../lib/actions";
import { Info } from "../lib/info";
import { toast } from "../lib/toasts";
import type {
  AdmissionPreview, Gate, HistoryRow, LogTail as LogTailT, ModelDetail,
  ModelRow, PlanRow, RunCommand as RunCommandT, RuntimePolicyView,
} from "../lib/types";

function MeasuredTag({ measured }: { measured: boolean }) {
  return measured
    ? <span class="tag measured">measured</span>
    : <span class="tag estimated">estimated</span>;
}

function Overview({ d, live, stale, lastAt, retryIn }: {
  d: ModelDetail; live: ModelRow | null; stale: boolean;
  lastAt: number | null; retryIn: number | null;
}) {
  const rt = live ?? null;
  const state = rt ? rt.state : d.runtime.state;
  const cls = (rt ? rt.state_class : d.runtime.state_class) || "stopped";
  return (
    <div class={stale ? "widget stale" : "widget"}>
      <div class="label">
        <span>overview</span>
        {/* the mark used to say "live" whenever a runtime row existed, so a
            dropped stream showed frozen numbers under a pulsing green dot */}
        {rt && stale && (
          <span class="live stale"><span class="dot"></span>stale</span>
        )}
      </div>
      {stale && (
        <div class="sub stale-note">
          stream dropped — showing the last reading
          {lastAt != null ? ` from ${fmtClock(lastAt)}` : ""}
          {retryIn != null ? ` · retrying in ${retryIn}s` : " · reconnecting…"}
        </div>
      )}
      <table>
        <tbody>
          <tr><td class="sub">state</td>
            <td><span class={`chip ${cls}`}><span class="dot"></span>{state}</span>
              {rt?.tok_s != null && <span class="num"> · {rt.tok_s.toFixed(1)} tok/s</span>}
            </td></tr>
          <tr><td class="sub">source</td>
            <td>{d.repo_id || "(local import)"}<span class="sub"> · {d.file}</span></td></tr>
          <tr><td class="sub">weights</td>
            <td class="num">{d.size_bytes != null ? `${fmtGiB(d.size_bytes)} GiB on disk` : "—"}</td></tr>
          <tr><td class="sub">backend</td>
            <td>{d.backend}{d.binary ? <span class="sub"> · {d.binary}</span> : null}</td></tr>
          <tr><td class="sub">placement</td>
            <td>{d.config.tensor_split
              ? `split ${d.config.tensor_split} (${d.config.split_mode || "layer"})`
              : (d.config.device || "(backend default)")}
              {d.config.extra ? <div class="sub">{d.config.extra}</div> : null}</td></tr>
          <tr><td class="sub">planning inputs</td>
            <td>{d.planning.has_stored_inputs
              ? <>stored{d.planning.recorded_at ? <span class="sub"> · recorded {d.planning.recorded_at}</span> : null}</>
              : <span class="sub">not recorded — the next plan apply freezes them</span>}
              {" "}
              <Info label="about planning inputs">
                The planner reads recorded machine facts (RAM, GPU inventory,
                limits) instead of probing live on every replan, so planning
                twice gives the same answer. Inputs are recorded the first
                time a plan is applied.
              </Info>
            </td></tr>
        </tbody>
      </table>
    </div>
  );
}

/* ---- runtime actions ------------------------------------------------- */

function RuntimeActions({ d, live, onChanged }: {
  d: ModelDetail; live: ModelRow | null; onChanged: () => void;
}) {
  const [busy, setBusy] = useState("");
  const running = live ? live.running : d.runtime.running;
  const run = (key: string, fn: () => Promise<unknown>, label: string,
               okSub?: (r: never) => string) => {
    setBusy(key);
    submitAction(fn, label, onChanged, okSub as never)
      .finally(() => setBusy(""));
  };
  return (
    <div class="widget">
      <div class="label"><span>actions</span></div>
      <div class="actions" style="flex-wrap:wrap;gap:.6rem">
        <ConfirmButton
          label="restart"
          busy={busy === "restart"}
          disabled={!!busy || !running}
          consequences={<>
            Stops and reloads {d.name}. In-flight requests to it fail, and
            the reload pays the model's full load time again.
          </>}
          onConfirm={() => run("restart", () => restartModel(d.name),
                               `restart ${d.name}`)} />
        {/* Not behind a confirm: it is not destructive to anything but
            the cache's own learned placement, which rebuilds itself. It
            IS reported honestly, because a reset that quietly did nothing
            makes the next measurement a lie. */}
        <button type="button" class={busy === "cache" ? "busy" : undefined}
                disabled={!!busy || !running || d.moe_cache.mode === "off"}
                onClick={() => run("cache", () => resetModelCache(d.name),
                                   `MoE cache reset for ${d.name}`,
                                   ((r: { port: number }) =>
                                     `cleared on port ${r.port} · the cache relearns from the next request`) as never)}>
          reset MoE cache
        </button>
      </div>
      {!running && (
        <div class="sub" style="margin-top:.4rem">
          restart and cache reset need the model resident — it is
          currently {live ? live.state : d.runtime.state}
        </div>
      )}
      {running && d.moe_cache.mode === "off" && (
        <div class="sub" style="margin-top:.4rem">
          no MoE cache to reset — this profile's cache mode is off
        </div>
      )}
    </div>
  );
}

/* ---- measurement triggers (moved here from the catalog rows) --------- */

function BenchForm({ name, onDone }: { name: string; onDone: () => void }) {
  const [tokens, setTokens] = useState("256");
  const [runs, setRuns] = useState("3");
  const [busy, setBusy] = useState(false);
  return (
    <div class="frow" style="margin-top:.4rem;align-items:end">
      <div class="field" style="max-width:120px">
        <label for={`bench-tok-${name}`}>max tokens</label>
        <input id={`bench-tok-${name}`} type="number" min={1} max={4096}
               value={tokens}
               onInput={(e) => setTokens((e.target as HTMLInputElement).value)} />
      </div>
      <div class="field" style="max-width:100px">
        <label for={`bench-runs-${name}`}>runs</label>
        <input id={`bench-runs-${name}`} type="number" min={1} max={10}
               value={runs}
               onInput={(e) => setRuns((e.target as HTMLInputElement).value)} />
      </div>
      <div class="actions">
        <button type="button" onClick={onDone}>cancel</button>
        <button type="button" class={busy ? "btn-primary busy" : "btn-primary"}
                disabled={busy}
                onClick={() => {
                  setBusy(true);
                  submitAction(
                    () => benchModel(name, parseInt(tokens, 10) || 256,
                                     parseInt(runs, 10) || 3),
                    `benchmark ${name}`,
                    () => onDone(),
                    (r) => `job ${r.job_id} · ${r.max_tokens} tokens x ${r.runs}`
                  ).finally(() => setBusy(false));
                }}>
          run benchmark
        </button>
      </div>
    </div>
  );
}

function Measure({ name }: { name: string }) {
  const [benching, setBenching] = useState(false);
  const [busy, setBusy] = useState("");
  const run = (key: string, fn: () => Promise<{ job_id: string }>,
               label: string) => {
    setBusy(key);
    submitAction(fn, label).finally(() => setBusy(""));
  };
  return (
    <div class="widget">
      <div class="label"><span>measure{" "}
        <Info label="about the measurement triggers">
          A benchmark runs the speed harness and records the result; a
          test proves the model answers at all; autotune launches
          candidate plans and keeps the one that measures best. All three
          queue one at a time, so two measurements never share the
          machine.
        </Info></span>
      </div>
      <div class="actions" style="margin-top:.4rem">
        <button type="button" class={busy === "bench" ? "busy" : undefined}
                onClick={() => setBenching(!benching)}>benchmark</button>
        <button type="button" class={busy === "smoke" ? "busy" : undefined}
                disabled={!!busy}
                onClick={() => run("smoke", () => smokeModel(name),
                                   `test ${name}`)}>test</button>
        <button type="button" class={busy === "tune" ? "busy" : undefined}
                disabled={!!busy}
                onClick={() => run("tune", () => autotuneModel(name, "balanced"),
                                   `autotune ${name}`)}>autotune</button>
      </div>
      {benching && <BenchForm name={name} onDone={() => setBenching(false)} />}
    </div>
  );
}

/* ---- danger zone: destruction lives with configuration, quarantined --- */

function DangerZone({ d, onChanged }: {
  d: ModelDetail; onChanged: () => void;
}) {
  const [busy, setBusy] = useState(false);
  return (
    <div class="widget" style="border-color:var(--err);max-width:880px">
      <div class="label"><span style="color:var(--err)">danger zone</span></div>
      <div class="actions" style="margin-top:.4rem">
        <ConfirmButton
          label="delete profile"
          confirmLabel="yes, delete the profile"
          busy={busy}
          disabled={busy}
          consequences={<>
            Deletes the profile and its generated artifacts, then resyncs
            the backends so {d.name} is no longer served. The model file
            on disk ({d.model_path || "unknown path"}) is left alone.
          </>}
          onConfirm={() => {
            setBusy(true);
            submitAction(() => deleteModel(d.name), `delete ${d.name}`,
                         onChanged)
              .finally(() => setBusy(false));
          }} />
      </div>
    </div>
  );
}

/* ---- tier apply ------------------------------------------------------ */

function TierApply({ name, revision, onApplied }: {
  name: string; revision: number; onApplied: () => void;
}) {
  const [preview, setPreview] = useState<AdmissionPreview | null>(null);
  const [err, setErr] = useState("");
  const [gate, setGate] = useState<Gate | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    setErr("");
    /* No draft arguments: this is the planner's answer for the profile
       exactly as saved, which is what apply would compute. The same
       endpoint the configure form drafts against. */
    fetchAdmission(name).then(setPreview).catch((e) => setErr(String(e)));
  }, [name, revision]);

  const apply = (accept: boolean) => {
    setBusy(true);
    setGate(null);
    applyTier(name, accept)
      .then((r) => {
        toast("ok", `tier apply ${name} submitted`,
              `job ${r.job_id} · watch it on the jobs page`);
        onApplied();
      })
      .catch((e) => {
        if (e instanceof ApiError && e.status === 409 && e.body.gate) {
          setGate(e.body.gate as Gate);
        } else if (e instanceof ApiError && e.status === 405) {
          toast("err", "✗ tier apply refused",
                String(e.body.reason ?? e.message), 9000);
        } else if (e instanceof ApiError) {
          toast("err", `✗ tier apply ${name} failed`,
                String(e.body.error ?? e.message), 9000);
        } else {
          toast("err", `✗ tier apply ${name} failed`, String(e), 9000);
        }
      })
      .finally(() => setBusy(false));
  };

  const plan = preview?.plan ?? null;
  const admission = plan?.admission ?? null;
  return (
    <div class="widget">
      <div class="label"><span>tier plan{" "}
        <Info label="about tiers">
          The planner picks a placement tier from the recorded machine
          facts: tier 1 is GPU-only, 2 adds RAM, 3 adds storage. Applying
          writes the tier's placement into the profile and resyncs. This
          is the same plan and the same gate the CLI's tier apply uses.
        </Info></span>
      </div>
      {err && <p class="sub">planner unavailable: {err}</p>}
      {!preview && !err && <p class="sub">asking the planner…</p>}
      {preview && !plan && (
        <p class="sub">the planner could not analyze this model's layout
          {preview.error ? ` — ${preview.error}` : ""} — nothing to apply</p>
      )}
      {plan && (
        <>
          <div class="sub">
            tier {plan.tier} · inputs {preview?.planning_inputs_source}
            {admission && (admission.fits
              ? " · fits as chosen"
              : " · does not fit even degraded")}
          </div>
          <table>
            <tbody>
              {plan.layout.map(([label, gib, desc], i) => (
                <tr key={i}>
                  <td class="sub">{label}</td>
                  <td class="num">{gib.toFixed(1)} GiB</td>
                  <td class="sub">{desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {plan.warnings.map((w) => (
            <div class="msg warning" key={w}>⚠ {w}</div>
          ))}
          {preview?.gate && preview.gate.changes.length > 0 && (
            <div class="sub" style="margin-top:.4rem">
              this replan differs from what is saved:
              {preview.gate.changes.map((c) => (
                <div class="sub" key={c}>· {c}</div>
              ))}
            </div>
          )}
          {gate && (
            <fieldset style="border-color:var(--warn);margin-top:.6rem">
              <legend style="color:var(--warn)">
                tier change — confirm required
              </legend>
              <p class="sub">This replan changes tier, split, or placement
                beyond the profile's pins:</p>
              {gate.changes.map((c) => (
                <div class="msg warning" key={c}>⚠ {c}</div>
              ))}
              <div class="actions">
                <button type="button" onClick={() => setGate(null)}>cancel</button>
                <button type="button" class="btn-danger" disabled={busy}
                        onClick={() => apply(true)}>
                  apply the tier change
                </button>
              </div>
            </fieldset>
          )}
          <div class="actions" style="justify-content:flex-end;margin-top:.6rem">
            <button type="button" class={busy ? "busy" : undefined}
                    disabled={busy} onClick={() => apply(false)}>
              apply tier {plan.tier}
            </button>
          </div>
        </>
      )}
    </div>
  );
}

/* ---- plans, with their four actions ---------------------------------- */

function PlanActions({ name, plan, onChanged, onOptimistic }: {
  name: string; plan: PlanRow; onChanged: () => void;
  onOptimistic: (planId: string, disabled: boolean | null) => void;
}) {
  const [busy, setBusy] = useState("");
  const act = (action: PlanAction, label: string) => {
    setBusy(action);
    submitAction(() => planAction(name, plan.id, action),
                 `${label} ${plan.label}`, onChanged)
      .finally(() => setBusy(""));
  };
  /* enable/disable are the only reversible ones, so they are the only
     ones that reflect before the server answers -- and they un-happen
     loudly (the row snaps back and the toast carries the reason) when it
     refuses. select rewrites the profile's launch config and test starts
     a real server; neither is something to pretend has happened. */
  const toggle = (action: "enable" | "disable", label: string) => {
    setBusy(action);
    onOptimistic(plan.id, action === "disable");
    submitAction(() => planAction(name, plan.id, action),
                 `${label} ${plan.label}`, onChanged)
      .then((ok) => {
        if (!ok) onOptimistic(plan.id, null);
      })
      .finally(() => setBusy(""));
  };
  return (
    <span class="actions">
      <button type="button"
              class={busy === "select" ? "btn-primary busy" : "btn-primary"}
              disabled={!!busy || plan.pinned}
              onClick={() => act("select", "select plan")}>select</button>
      {plan.disabled
        ? <button type="button" class={busy === "enable" ? "busy" : undefined}
                  disabled={!!busy}
                  onClick={() => toggle("enable", "enable plan")}>enable</button>
        : <button type="button" class={busy === "disable" ? "busy" : undefined}
                  disabled={!!busy}
                  onClick={() => toggle("disable", "disable plan")}>disable</button>}
      <button type="button" class={busy === "test" ? "busy" : undefined}
              disabled={!!busy}
              onClick={() => act("test", "test plan")}>test</button>
    </span>
  );
}

function Plans({ name, revision, onChanged }: {
  name: string; revision: number; onChanged: () => void;
}) {
  const [plans, setPlans] = useState<PlanRow[] | null>(null);
  const [err, setErr] = useState("");
  /* null restores the server's value: the un-happen path must not invert
     the flag a second time, it has to forget the optimistic one. */
  const [pretend, setPretend] = useState<Record<string, boolean>>({});
  useEffect(() => {
    fetchModelPlans(name)
      .then((rows) => {
        setPlans(rows);
        // fresh truth outranks every optimistic guess still standing
        setPretend({});
      })
      .catch((e) => setErr(String(e)));
  }, [name, revision]);
  const onOptimistic = (planId: string, disabled: boolean | null) =>
    setPretend((p) => {
      if (disabled === null) {
        const { [planId]: _drop, ...rest } = p;
        return rest;
      }
      return { ...p, [planId]: disabled };
    });
  const rows = plans?.map((p) => (
    p.id in pretend ? { ...p, disabled: pretend[p.id] } : p));
  return (
    <div class="widget">
      <div class="label">
        <span>launch plans{" "}
          <Info label="about plans">
            Plans are recompiled deterministically from the profile and the
            hardware snapshot — they are not stored. A measured tag means a
            real benchmark of that exact plan; estimated numbers come from
            the GGUF layout math. Measurements outrank estimates.
          </Info>
        </span>
      </div>
      {err && <p class="sub">plans unavailable: {err}</p>}
      {!rows && !err && <p class="sub">compiling plans…</p>}
      {rows && rows.length === 0 && <p class="sub">no plans could be compiled</p>}
      {rows && rows.length > 0 && (
        <table>
          <thead>
            <tr><th>plan</th><th>status</th><th class="num">tok/s</th>
              <th class="num">load</th><th>numbers</th><th>actions{" "}
                <Info label="about the plan actions">
                  Select writes this plan's placement into the profile and
                  resyncs. Disable takes it out of the ranked candidate
                  set without deleting anything, and enable puts it back.
                  Test launches a real server on this exact plan and
                  records what it measures.
                </Info></th></tr>
          </thead>
          <tbody>
            {rows.map((p) => (
              <tr key={p.id}>
                <td>
                  {p.label}
                  {p.pinned && <span class="tag" style="margin-left:.4em">pinned</span>}
                  {p.disabled && <span class="tag" style="margin-left:.4em">disabled</span>}
                  {p.warnings.length > 0 && (
                    <div class="sub" style="color:var(--warn)">
                      {p.warnings.map((w) => <div key={w}>⚠ {w}</div>)}
                    </div>
                  )}
                </td>
                <td><span class="chip"><span class="dot"></span>{p.category_label}</span>
                  {p.stale && <div class="sub">stale — machine changed since</div>}</td>
                <td class="num">
                  {p.measured?.generation_tps != null
                    ? p.measured.generation_tps.toFixed(1)
                    : "—"}
                </td>
                <td class="num">
                  {p.measured?.load_seconds != null
                    ? `${p.measured.load_seconds.toFixed(0)}s`
                    : "—"}
                </td>
                <td>
                  <MeasuredTag measured={!!p.measured} />
                  {p.measured?.cache_state
                    ? <span class="sub"> {p.measured.cache_state}</span> : null}
                </td>
                <td>
                  <PlanActions name={name} plan={p} onChanged={onChanged}
                               onOptimistic={onOptimistic} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function History({ name, revision }: { name: string; revision: number }) {
  const [rows, setRows] = useState<HistoryRow[] | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    fetchModelHistory(name).then(setRows).catch((e) => setErr(String(e)));
  }, [name, revision]);
  return (
    <div class="widget">
      <div class="label"><span>measurement history{" "}
        <Info label="about history">
          Every recorded run from the measurement store, newest first —
          benchmarks and serving launches, successes and failures. The
          bottleneck column is judged from counters the run actually
          recorded; "unknown" means the evidence doesn't support a verdict.
        </Info></span>
      </div>
      {err && <p class="sub">history unavailable: {err}</p>}
      {!rows && !err && <p class="sub">loading history…</p>}
      {rows && rows.length === 0 && <p class="sub">no measurements recorded yet</p>}
      {rows && rows.length > 0 && (
        <table>
          <thead>
            <tr><th>when</th><th>plan</th><th>kind</th><th>result</th>
              <th class="num">tok/s</th><th class="num">load</th><th>bottleneck</th></tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td class="sub">{r.started_at ? fmtAgo(r.started_at) : "—"}</td>
                <td class="sub">{r.plan_id.slice(0, 8)}</td>
                <td class="sub">{r.run_kind}{r.cache_state ? ` · ${r.cache_state}` : ""}</td>
                <td>{r.success
                  ? <span class="chip ok"><span class="dot"></span>ok</span>
                  : <span class="chip err"><span class="dot"></span>{r.failure_class || "failed"}</span>}
                </td>
                <td class="num">{r.generation_tps != null ? r.generation_tps.toFixed(1) : "—"}</td>
                <td class="num">{r.load_seconds != null ? `${r.load_seconds.toFixed(0)}s` : "—"}</td>
                <td class="sub">
                  {r.bottleneck}
                  {/* the evidence behind the verdict was in a title
                      attribute -- unreachable by keyboard and invisible on
                      touch. Summoned help is the spec's affordance. */}
                  {r.bottleneck_why && (
                    <>{" "}<Info label={`why ${r.bottleneck}`}>
                      {r.bottleneck_why}
                    </Info></>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function Logs({ name }: { name: string }) {
  const [log, setLog] = useState<LogTailT | null>(null);
  // A polling region that hides its own failures is the worst kind of
  // live surface: the tail simply stops moving. Track the last success
  // and mark the region when polling starts failing.
  const [lastAt, setLastAt] = useState<number | null>(null);
  const [pollErr, setPollErr] = useState("");
  const ref = useRef<HTMLPreElement>(null);
  useEffect(() => {
    let alive = true;
    const pull = () => fetchLogTail(name)
      .then((l) => {
        if (!alive) return;
        setLog(l);
        setLastAt(Date.now());
        setPollErr("");
      })
      .catch((e) => { if (alive) setPollErr(String(e)); });
    pull();
    const t = setInterval(pull, 5000);
    return () => { alive = false; clearInterval(t); };
  }, [name]);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [log?.tail]);
  const stale = !!pollErr;
  return (
    <div class={stale ? "widget stale" : "widget"}>
      <div class="label"><span>log tail</span>
        {stale
          ? <span class="live stale"><span class="dot"></span>
              polling failed{lastAt != null ? ` · last ${fmtClock(lastAt)}` : ""}
            </span>
          : <span class="live"><span class="dot"></span>live · every 5 s</span>}
      </div>
      {log?.source && <div class="sub">from {log.source}</div>}
      {pollErr && <div class="sub stale-note">{pollErr}</div>}
      {!log && !pollErr && <p class="sub">fetching logs…</p>}
      {log && log.error && !log.tail && <p class="sub">{log.error}</p>}
      {log && log.tail && (
        <pre class="log" style="max-height:300px" ref={ref}>{log.tail}</pre>
      )}
    </div>
  );
}

/* ---- typed configure form ------------------------------------------- */

interface Draft {
  ctx: string;
  flash_attn: string;
  ttl: string;
  cache_type_k: string;
  cache_type_v: string;
  moe_mode: string;
  budgets: Record<string, string>; // GiB text per device
  enabled: boolean;
}

function draftFrom(d: ModelDetail): Draft {
  const budgets: Record<string, string> = {};
  for (const g of d.gpus) {
    const bytes = d.moe_cache.budgets_bytes[g.device];
    budgets[g.device] = bytes ? (bytes / 2 ** 30).toFixed(1) : "";
  }
  return {
    ctx: String(d.config.ctx || ""),
    flash_attn: String(d.config.flash_attn || "auto"),
    ttl: String(d.config.ttl ?? ""),
    cache_type_k: String(d.config.cache_type_k || "q8_0"),
    cache_type_v: String(d.config.cache_type_v || "q8_0"),
    moe_mode: d.moe_cache.mode,
    budgets,
    enabled: d.enabled,
  };
}

function budgetsBytes(draft: Draft): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [dev, gib] of Object.entries(draft.budgets)) {
    const v = parseFloat(gib);
    if (!Number.isNaN(v) && v > 0) out[dev] = Math.round(v * 2 ** 30);
  }
  return out;
}

export function Configure({ d, onSaved }:
                          { d: ModelDetail; onSaved: () => void }) {
  const [draft, setDraft] = useState<Draft>(() => draftFrom(d));
  const [preview, setPreview] = useState<AdmissionPreview | null>(null);
  const [gate, setGate] = useState<Gate | null>(null);
  const [busy, setBusy] = useState(false);
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);

  const set = (patch: Partial<Draft>) => setDraft((old) => ({ ...old, ...patch }));

  /* Live admission preview: what the planner would say about the draft.
     Debounced — this is a read, but not a free one. */
  useEffect(() => {
    if (debounce.current) clearTimeout(debounce.current);
    debounce.current = setTimeout(() => {
      const ctx = parseInt(draft.ctx, 10);
      // moe_mode rides along: the planner only charges budgets when the
      // mode is on, so previewing enable-the-cache needs the draft mode.
      fetchAdmission(d.name, Number.isNaN(ctx) ? null : ctx,
                     draft.moe_mode !== "off" ? budgetsBytes(draft) : null,
                     draft.moe_mode)
        .then(setPreview)
        .catch(() => setPreview(null));
    }, 400);
    return () => {
      if (debounce.current) clearTimeout(debounce.current);
    };
  }, [d.name, draft.ctx, draft.moe_mode, JSON.stringify(draft.budgets)]);

  const kvNote = useMemo(() => {
    const plan = preview?.plan;
    if (!plan) return "";
    const an = plan.analysis as Record<string, unknown>;
    const kv = an && typeof an["kv_gib"] === "number" ? an["kv_gib"] as number : null;
    return kv != null ? `KV cache ≈ ${kv.toFixed(1)} GiB at this setting` : "";
  }, [preview]);

  const admission = preview?.plan?.admission ?? null;
  const warnings = preview?.plan?.warnings ?? [];

  const save = async (accept: boolean) => {
    setBusy(true);
    setGate(null);
    const updates: Record<string, string | boolean> = {};
    const cur = draftFrom(d);
    for (const k of ["ctx", "flash_attn", "ttl", "cache_type_k", "cache_type_v"] as const) {
      if (draft[k] !== cur[k]) updates[k] = draft[k];
    }
    if (draft.enabled !== cur.enabled) updates["enabled"] = draft.enabled;
    const body = {
      updates,
      budgets_bytes: JSON.stringify(draft.budgets) !== JSON.stringify(cur.budgets)
        ? budgetsBytes(draft) : undefined,
      moe_mode: draft.moe_mode !== cur.moe_mode ? draft.moe_mode : undefined,
      accept_structural: accept,
    };
    if (Object.keys(updates).length === 0
        && body.budgets_bytes === undefined && body.moe_mode === undefined) {
      toast("ok", "nothing to save", "no field differs from the stored profile");
      setBusy(false);
      return;
    }
    try {
      const res = await saveModelConfig(d.name, body);
      toast("ok", `configure ${d.name} submitted`,
            "queued on the mutation lane · watch it on the jobs page");
      onSaved();
      void res;
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && e.body.gate) {
        setGate(e.body.gate as Gate); // the explicit confirm renders below
      } else if (e instanceof ApiError && e.status === 405) {
        toast("err", "✗ save refused", String(e.body.reason ?? e.message), 9000);
      } else {
        toast("err", `✗ configure ${d.name} failed`, String(e), 9000);
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <div class="widget" style="max-width:880px">
      <h2>configure</h2>
      <p class="sub" style="margin:.2rem 0 .9rem">
        Typed form, no JSON. The file on disk stays hand-editable from the
        shell; this is the only editor the console offers. Saving shows
        the planner's admission answer first.
      </p>
      <form style="display:grid;gap:1rem" onSubmit={(e) => e.preventDefault()}>
        <fieldset>
          <legend>profile</legend>
          <div class="frow">
            <div class="field">
              <div class="lbl">
                <label for="cfg-ctx">context length <span class="sub">tokens</span></label>
                <Info label="about context length">
                  KV cache grows linearly with this; it is the biggest
                  single VRAM lever. The planner sizes KV at exactly this
                  value during admission.
                </Info>
              </div>
              <input id="cfg-ctx" type="number" min={512} step={1024} value={draft.ctx}
                     onInput={(e) => set({ ctx: (e.target as HTMLInputElement).value })} />
              {kvNote && <div class="msg" style="color:var(--ok)">✓ {kvNote}</div>}
            </div>
            <div class="field">
              <div class="lbl">
                <label for="cfg-ttl">TTL <span class="sub">seconds</span></label>
                <Info label="about TTL">
                  How long llama-swap keeps the model resident after its
                  last request before unloading it.
                </Info>
              </div>
              <input id="cfg-ttl" type="number" min={0} value={draft.ttl}
                     onInput={(e) => set({ ttl: (e.target as HTMLInputElement).value })} />
            </div>
            <div class="check" style="align-self:end">
              <input type="checkbox" id="cfg-enabled" checked={draft.enabled}
                     onChange={(e) => set({ enabled: (e.target as HTMLInputElement).checked })} />
              <div><b><label for="cfg-enabled">enabled</label></b>
                <div class="help">disabled profiles are removed from the
                  swap config on the next sync</div>
              </div>
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>attention + KV cache</legend>
          <div class="frow">
            <div class="field">
              <div class="lbl"><label for="cfg-fa">flash attention</label>
                <Info label="about flash attention">
                  Faster prompt processing on SYCL. "auto" lets the backend
                  decide from its probed capabilities.
                </Info>
              </div>
              <select id="cfg-fa" value={draft.flash_attn}
                      onChange={(e) => set({ flash_attn: (e.target as HTMLSelectElement).value })}>
                <option value="auto">auto</option>
                <option value="on">on</option>
                <option value="off">off</option>
              </select>
            </div>
            <div class="field">
              <div class="lbl"><label for="cfg-ctk">KV cache type (K)</label>
                <Info label="about the K cache type">
                  Quantization for the K half of the KV cache. It multiplies
                  straight into the KV size the context length implies —
                  q8_0 is roughly half of f16 — so it is the second-biggest
                  VRAM lever after context.
                </Info>
              </div>
              <select id="cfg-ctk" value={draft.cache_type_k}
                      onChange={(e) => set({ cache_type_k: (e.target as HTMLSelectElement).value })}>
                {["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"].map((t) =>
                  <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
            <div class="field">
              <div class="lbl"><label for="cfg-ctv">KV cache type (V)</label>
                <Info label="about the V cache type">
                  Quantization for the V half. Some backends tolerate a
                  quantized K better than a quantized V; if generation
                  quality drops after changing this, put V back to f16
                  first.
                </Info>
              </div>
              <select id="cfg-ctv" value={draft.cache_type_v}
                      onChange={(e) => set({ cache_type_v: (e.target as HTMLSelectElement).value })}>
                {["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"].map((t) =>
                  <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>MoE expert cache</legend>
          <div class="frow">
            <div class="field" style="max-width:220px">
              <div class="lbl"><label for="cfg-moe">mode</label>
                <Info label="about the MoE cache">
                  Reserves a VRAM budget per GPU for hot experts. Learning
                  starts at load; watch the hit-ratio meter on operate.
                  Off means experts run where the placement puts them.
                </Info>
              </div>
              <select id="cfg-moe" value={draft.moe_mode}
                      onChange={(e) => set({ moe_mode: (e.target as HTMLSelectElement).value })}>
                <option value="off">off</option>
                <option value="auto">auto</option>
                <option value="manual">manual</option>
              </select>
            </div>
            {d.gpus.map((g) => {
              const dev = admission?.devices?.[g.device];
              const bytesTxt = draft.budgets[g.device] ?? "";
              const wanted = parseFloat(bytesTxt);
              const headroom = dev
                ? (dev.usable_bytes - (dev.demand_bytes - dev.cache_bytes)) / 2 ** 30
                : null;
              const over = headroom != null && !Number.isNaN(wanted)
                && wanted > Math.max(0, headroom);
              return (
                <div class={over ? "field warning" : "field"} key={g.device}>
                  <label for={`cfg-budget-${g.device}`}>{g.device} budget{" "}
                    <span class="sub">GiB</span></label>
                  <input id={`cfg-budget-${g.device}`} type="number" min={0}
                         step={0.5} value={bytesTxt}
                         disabled={draft.moe_mode === "off"}
                         onInput={(e) => set({
                           budgets: { ...draft.budgets,
                                      [g.device]: (e.target as HTMLInputElement).value } })} />
                  <div class="help">{g.name} · {fmtGiB(g.total_bytes, 0)} GiB</div>
                  {over && headroom != null && (
                    <div class="msg warning">
                      ⚠ exceeds the planner's admissible headroom
                      ({Math.max(0, headroom).toFixed(1)} GiB) — launch will
                      fall back to a smaller cache
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </fieldset>

        {/* admission preview: the server's own answer, inline and visible */}
        <fieldset>
          <legend>admission preview{" "}
            <Info label="about admission">
              Computed by the planner at plan time from recorded inputs:
              requested is what this form asks for, assumed is the machine
              facts the plan used, chosen is what would actually launch
              after any degradations (cache shrinks first, then pinned
              experts spill to CPU).
            </Info>
          </legend>
          {!preview && <p class="sub">asking the planner…</p>}
          {preview && !preview.plan && (
            <p class="sub">the planner could not analyze this model's layout
              {preview.error ? ` — ${preview.error}` : ""}</p>
          )}
          {preview?.plan && (
            <>
              <div class="sub">
                tier {preview.plan.tier} · inputs {preview.planning_inputs_source}
                {admission && (admission.fits
                  ? " · fits as chosen"
                  : " · does not fit even degraded")}
              </div>
              {admission && (
                <table>
                  <thead>
                    <tr><th></th><th>requested</th><th>chosen</th></tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td class="sub">ctx</td>
                      <td class="num">{String((admission.requested as Record<string, unknown>)["ctx"] ?? "—")}</td>
                      <td class="num">tier {String((admission.chosen as Record<string, unknown>)["tier"] ?? "")}</td>
                    </tr>
                    <tr>
                      <td class="sub">cache budgets</td>
                      <td class="num">{fmtBudgetMap((admission.requested as Record<string, unknown>)["cache_budgets_bytes"])}</td>
                      <td class="num">{fmtBudgetMap((admission.chosen as Record<string, unknown>)["cache_budgets_bytes"])}</td>
                    </tr>
                  </tbody>
                </table>
              )}
              {admission && admission.degradations.length > 0 && (
                <div class="sub" style="color:var(--warn)">
                  {admission.degradations.map((deg, i) => (
                    <div key={i}>⚠ {deg.action} on {deg.device}
                      {deg.requested_bytes != null && deg.chosen_bytes != null
                        ? `: ${fmtGiB(deg.requested_bytes)} → ${fmtGiB(deg.chosen_bytes)} GiB` : ""}
                      {deg.reason ? ` — ${deg.reason}` : ""}</div>
                  ))}
                </div>
              )}
              {warnings.map((w) => (
                <div class="msg warning" key={w}>⚠ {w}</div>
              ))}
            </>
          )}
        </fieldset>

        {/* structural gate: the explicit confirm */}
        {gate && (
          <fieldset style="border-color:var(--warn)">
            <legend style="color:var(--warn)">structural change — confirm required</legend>
            <p class="sub">This save changes placement or cache budgets, which
              rewrites where the model's bytes live:</p>
            {gate.changes.map((c) => <div class="msg warning" key={c}>⚠ {c}</div>)}
            <div class="actions">
              <button type="button" onClick={() => setGate(null)}>cancel</button>
              <button type="button" class="btn-danger" disabled={busy}
                      onClick={() => save(true)}>
                apply structural change
              </button>
            </div>
          </fieldset>
        )}

        <div class="actions" style="justify-content:flex-end">
          <button type="button" class="btn-primary" disabled={busy}
                  onClick={() => save(false)}>
            save configuration
          </button>
        </div>
      </form>
    </div>
  );
}

/* ---- runtime policy (typed form) ------------------------------------- */

function RuntimePolicyForm({ name, revision, onSaved }: {
  name: string; revision: number; onSaved: () => void;
}) {
  const [view, setView] = useState<RuntimePolicyView | null>(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [managed, setManaged] = useState(false);
  const [objective, setObjective] = useState("balanced");
  const [pinned, setPinned] = useState("");
  const [allowFallback, setAllowFallback] = useState(true);
  const [allowUntested, setAllowUntested] = useState(false);
  const [minCtx, setMinCtx] = useState("");
  const [maxCpuGiB, setMaxCpuGiB] = useState("");
  const [tier, setTier] = useState("3");

  useEffect(() => {
    fetchRuntimePolicy(name)
      .then((v) => {
        setView(v);
        const p = v.policy;
        setManaged(!!p && p.mode === "managed");
        setObjective(p?.objective || "balanced");
        setPinned(p?.pinned_plan_id || "");
        setAllowFallback(p ? !!p.allow_fallback : true);
        setAllowUntested(p ? !!p.allow_untested : false);
        setMinCtx(p?.minimum_context != null ? String(p.minimum_context) : "");
        setMaxCpuGiB(p?.maximum_cpu_bytes != null
          ? String(Math.round(p.maximum_cpu_bytes / 2 ** 30)) : "");
        setTier(String(p?.maximum_storage_tier ?? 3));
      })
      .catch((e) => setErr(String(e)));
  }, [name, revision]);

  const save = () => {
    setBusy(true);
    const body: Record<string, unknown> = managed
      ? {
          mode: "managed",
          objective,
          pinned_plan_id: pinned || null,
          allow_fallback: allowFallback,
          allow_untested: allowUntested,
          minimum_context: minCtx === "" ? null : parseInt(minCtx, 10),
          /* the form asks for GiB because that is the unit an operator
             thinks in; the profile stores bytes, and the conversion
             happens here rather than in a server-side "_gib" field that
             would have to exist forever */
          maximum_cpu_bytes: maxCpuGiB === ""
            ? null : Math.round(parseFloat(maxCpuGiB) * 2 ** 30),
          maximum_storage_tier: parseInt(tier, 10),
          disabled_plan_ids: view?.policy?.disabled_plan_ids ?? [],
        }
      : { mode: "fixed" };
    submitAction(() => saveRuntimePolicy(name, body),
                 `runtime policy for ${name}`, onSaved)
      .finally(() => setBusy(false));
  };

  return (
    <div class="widget" style="max-width:880px">
      <h2>runtime policy{" "}
        <Info label="about runtime policy">
          Fixed renders one command from the saved config, forever.
          Managed lets the planner rank the compiled plans at launch and
          pick the best one for the objective — it can only pick plans it
          has evidence for unless untested plans are allowed.
        </Info>
      </h2>
      {err && <p class="sub">policy unavailable: {err}</p>}
      {!view && !err && <p class="sub">loading policy…</p>}
      {view && (
        <form style="display:grid;gap:1rem" onSubmit={(e) => e.preventDefault()}>
          <fieldset>
            <legend>mode</legend>
            <div class="frow">
              <div class="field" style="max-width:260px">
                <label for="rp-mode">placement</label>
                <select id="rp-mode" value={managed ? "managed" : "fixed"}
                        onChange={(e) => setManaged(
                          (e.target as HTMLSelectElement).value === "managed")}>
                  <option value="fixed">fixed command</option>
                  <option value="managed">managed placement</option>
                </select>
                <div class="help">
                  {managed
                    ? "the planner chooses a plan at every launch"
                    : "saving fixed drops the policy and restores the profile's own command"}
                </div>
              </div>
            </div>
          </fieldset>

          {managed && (
            <>
              <fieldset>
                <legend>ranking</legend>
                <div class="frow">
                  <div class="field" style="max-width:260px">
                    <div class="lbl"><label for="rp-objective">objective</label>
                      <Info label="about objectives">
                        What the ranker maximizes. Every option here is one
                        the ranker actually scores differently — the list
                        comes from the server, so the form cannot offer an
                        objective the write rejects.
                      </Info>
                    </div>
                    <select id="rp-objective" value={objective}
                            onChange={(e) => setObjective(
                              (e.target as HTMLSelectElement).value)}>
                      {view.objectives.map((o) => (
                        <option key={o} value={o}>{o.replace(/_/g, " ")}</option>
                      ))}
                    </select>
                  </div>
                  <div class="field">
                    <label for="rp-pinned">pinned plan</label>
                    <select id="rp-pinned" value={pinned}
                            onChange={(e) => setPinned(
                              (e.target as HTMLSelectElement).value)}>
                      <option value="">(none — rank every candidate)</option>
                      {view.plans.map((p) => (
                        <option key={p.id} value={p.id}>{p.label}</option>
                      ))}
                    </select>
                    {pinned && !view.plans.some((p) => p.id === pinned) && (
                      <div class="msg warning">
                        ⚠ the pinned plan {pinned.slice(0, 8)} is not among the
                        plans that compile today
                      </div>
                    )}
                  </div>
                </div>
                <div class="frow">
                  <div class="check">
                    <input type="checkbox" id="rp-fallback" checked={allowFallback}
                           onChange={(e) => setAllowFallback(
                             (e.target as HTMLInputElement).checked)} />
                    <div><b><label for="rp-fallback">allow fallback</label></b>
                      <div class="help">when the ranked plan fails to launch,
                        try the next one instead of failing the request</div>
                    </div>
                  </div>
                  <div class="check">
                    <input type="checkbox" id="rp-untested" checked={allowUntested}
                           onChange={(e) => setAllowUntested(
                             (e.target as HTMLInputElement).checked)} />
                    <div><b><label for="rp-untested">allow untested plans</label></b>
                      <div class="help">let the ranker choose a plan no run has
                        ever measured</div>
                    </div>
                  </div>
                </div>
              </fieldset>

              <fieldset>
                <legend>limits</legend>
                <div class="frow">
                  <div class="field" style="max-width:180px">
                    <label for="rp-minctx">minimum context <span class="sub">tokens</span></label>
                    <input id="rp-minctx" type="number" min={0} value={minCtx}
                           placeholder="(no floor)"
                           onInput={(e) => setMinCtx(
                             (e.target as HTMLInputElement).value)} />
                  </div>
                  <div class="field" style="max-width:180px">
                    <label for="rp-maxcpu">max CPU offload <span class="sub">GiB</span></label>
                    <input id="rp-maxcpu" type="number" min={0} value={maxCpuGiB}
                           placeholder="(no limit)"
                           onInput={(e) => setMaxCpuGiB(
                             (e.target as HTMLInputElement).value)} />
                  </div>
                  <div class="field" style="max-width:220px">
                    <label for="rp-tier">max tier</label>
                    <select id="rp-tier" value={tier}
                            onChange={(e) => setTier(
                              (e.target as HTMLSelectElement).value)}>
                      <option value="1">GPU only</option>
                      <option value="2">GPU + RAM</option>
                      <option value="3">GPU + RAM + SSD</option>
                    </select>
                  </div>
                </div>
              </fieldset>
            </>
          )}

          <div class="actions" style="justify-content:flex-end">
            <button type="button" class={busy ? "btn-primary busy" : "btn-primary"}
                    disabled={busy} onClick={save}>
              save policy &amp; resync
            </button>
          </div>
        </form>
      )}
    </div>
  );
}

/* ---- the launch command, read-only ----------------------------------- */

function RunCommand({ name, revision }: { name: string; revision: number }) {
  const [cmd, setCmd] = useState<RunCommandT | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    fetchRunCommand(name).then(setCmd).catch((e) => setErr(String(e)));
  }, [name, revision]);
  const drifted = !!cmd && !!cmd.pinned_binary
    && cmd.resolved_binary !== cmd.pinned_binary;
  return (
    <div class="widget">
      <div class="label"><span>launch command{" "}
        <Info label="about the launch command">
          Exactly what a launch of this profile would run, compiled by the
          same code that writes run.sh and the llama-swap entry — so it
          cannot drift from either. argv[0] is the RESOLVED binary, which
          is the one surface where a profile whose backend moved becomes
          visible. Read-only: nothing on this card writes.
        </Info></span>
      </div>
      {err && <p class="sub">command unavailable: {err}</p>}
      {!cmd && !err && <p class="sub">compiling the command…</p>}
      {cmd && cmd.error && <p class="sub">{cmd.error}</p>}
      {cmd && !cmd.error && (
        <>
          <div class="sub">
            fingerprint {cmd.command_fingerprint.slice(0, 12) || "—"}
            {cmd.ok ? "" : " · the backend preflight is not clean"}
          </div>
          {drifted && (
            <div class="msg warning">
              ⚠ resolved binary is {cmd.resolved_binary}, the profile pins{" "}
              {cmd.pinned_binary}
            </div>
          )}
          {cmd.messages.map((m) => <div class="msg" key={m}>{m}</div>)}
          {cmd.warnings.map((w) => (
            <div class="msg warning" key={w}>⚠ {w}</div>
          ))}
          <pre class="log" style="max-height:260px">{cmd.run_sh}</pre>
        </>
      )}
    </div>
  );
}

function fmtBudgetMap(v: unknown): string {
  if (!v || typeof v !== "object") return "—";
  const entries = Object.entries(v as Record<string, number>);
  if (entries.length === 0) return "none";
  return entries.map(([dev, b]) => `${dev} ${fmtGiB(b)} GiB`).join(" · ");
}

const TABS = ["overview", "plans", "history", "logs", "configure"] as const;
type Tab = (typeof TABS)[number];

export function Model({ name }: { name: string }) {
  const { tick, stale, lastAt, retryIn } = useStream();
  const [detail, setDetail] = useState<ModelDetail | null>(null);
  const [err, setErr] = useState("");
  const [reloadN, setReloadN] = useState(0);
  /* The subtitle used to promise "overview · plans · measurements ·
     logs · configure" over one endless scroll; now they are real tabs.
     The hash deep-links a tab and survives reload. */
  const [tab, setTab] = useState<Tab>(() => {
    const h = location.hash.replace("#", "");
    return (TABS as readonly string[]).includes(h) ? (h as Tab) : "overview";
  });
  const pick = (t: Tab) => {
    setTab(t);
    history.replaceState(null, "", `#${t}`);
  };

  useEffect(() => {
    fetchModelDetail(name).then(setDetail).catch((e) => setErr(String(e)));
  }, [name, reloadN]);

  /* Plans and measurement history are the page's evidence, and they only
     change when a job finishes. Counting finished jobs off the shared tick
     gives them a refresh trigger without polling: bench a model from
     anywhere and this page updates in place, per "no manual refresh
     anywhere operational state is shown". */
  const finishedJobs = tick
    ? tick.jobs.filter((j) => !["running", "queued"].includes(j.status)).length
    : 0;

  const live = tick?.models.find((m) => m.name === name) ?? null;

  if (err) {
    return (
      <div class="widget">
        <p>couldn't load {name}: {err}</p>
        <p class="sub"><a href="/v2/models">back to models</a></p>
      </div>
    );
  }
  if (!detail) {
    return <div class="widget"><span class="sub">loading {name}…</span></div>;
  }
  const reload = () => setReloadN((n) => n + 1);
  /* Both triggers refresh the evidence: a finished job anywhere, and any
     action submitted from this page. Without the second one, a plan
     select made here would not show until some job finished. */
  const revision = finishedJobs + reloadN;
  return (
    <>
      <div class="tabs" role="tablist" aria-label={`${name} sections`}>
        {TABS.map((t) => (
          <button key={t} type="button" role="tab" aria-selected={tab === t}
                  onClick={() => pick(t)}>{t}</button>
        ))}
      </div>
      {tab === "overview" && <>
        <Overview d={detail} live={live} stale={stale} lastAt={lastAt}
                  retryIn={retryIn} />
        <RuntimeActions d={detail} live={live} onChanged={reload} />
        <Measure name={name} />
        <TierApply name={name} revision={revision} onApplied={reload} />
      </>}
      {tab === "plans" && <>
        <Plans name={name} revision={revision} onChanged={reload} />
        <RuntimePolicyForm name={name} revision={revision} onSaved={reload} />
      </>}
      {tab === "history" && <History name={name} revision={revision} />}
      {tab === "logs" && <Logs name={name} />}
      {tab === "configure" && <>
        <Configure d={detail} onSaved={reload} />
        <RunCommand name={name} revision={revision} />
        <DangerZone d={detail} onChanged={reload} />
      </>}
    </>
  );
}
