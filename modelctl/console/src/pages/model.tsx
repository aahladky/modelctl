/* per-model page: overview + placement, compiled plans with
   measured/estimated tags, full measurement history from the store, log
   tail, and the typed configure form. The form's save path fetches the
   planner's admission preview (requested / assumed / chosen + warnings,
   the server's own wording) and surfaces the structural-change gate as
   an explicit confirm -- never a silent rewrite. */
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { useStream } from "../lib/stream";
import {
  ApiError, fetchAdmission, fetchLogTail, fetchModelDetail,
  fetchModelHistory, fetchModelPlans, fmtAgo, fmtGiB, saveModelConfig,
} from "../lib/api";
import { Info } from "../lib/info";
import { toast } from "../lib/toasts";
import type {
  AdmissionPreview, Gate, HistoryRow, LogTail as LogTailT, ModelDetail,
  ModelRow, PlanRow,
} from "../lib/types";

function MeasuredTag({ measured }: { measured: boolean }) {
  return measured
    ? <span class="tag measured">measured</span>
    : <span class="tag estimated">estimated</span>;
}

function Overview({ d, live }: { d: ModelDetail; live: ModelRow | null }) {
  const rt = live ?? null;
  const state = rt ? rt.state : d.runtime.state;
  const cls = (rt ? rt.state_class : d.runtime.state_class) || "stopped";
  return (
    <div class="widget">
      <div class="label">
        <span>overview</span>
        {rt && <span class="live"><span class="dot"></span>live</span>}
      </div>
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

function Plans({ name }: { name: string }) {
  const [plans, setPlans] = useState<PlanRow[] | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    fetchModelPlans(name).then(setPlans).catch((e) => setErr(String(e)));
  }, [name]);
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
      {!plans && !err && <p class="sub">compiling plans…</p>}
      {plans && plans.length === 0 && <p class="sub">no plans could be compiled</p>}
      {plans && plans.length > 0 && (
        <table>
          <thead>
            <tr><th>plan</th><th>status</th><th class="num">tok/s</th>
              <th class="num">load</th><th>numbers</th></tr>
          </thead>
          <tbody>
            {plans.map((p) => (
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
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function History({ name }: { name: string }) {
  const [rows, setRows] = useState<HistoryRow[] | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    fetchModelHistory(name).then(setRows).catch((e) => setErr(String(e)));
  }, [name]);
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
                <td class="sub" title={r.bottleneck_why}>{r.bottleneck}</td>
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
  const ref = useRef<HTMLPreElement>(null);
  useEffect(() => {
    let alive = true;
    const pull = () => fetchLogTail(name)
      .then((l) => { if (alive) setLog(l); })
      .catch(() => { /* transient; keep last tail */ });
    pull();
    const t = setInterval(pull, 5000);
    return () => { alive = false; clearInterval(t); };
  }, [name]);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [log?.tail]);
  return (
    <div class="widget">
      <div class="label"><span>log tail</span>
        {log?.source && <span class="sub">from {log.source} · refreshes every 5 s</span>}
      </div>
      {!log && <p class="sub">fetching logs…</p>}
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
              <label>context length <span class="sub">tokens</span>{" "}
                <Info label="about context length">
                  KV cache grows linearly with this; it is the biggest
                  single VRAM lever. The planner sizes KV at exactly this
                  value during admission.
                </Info>
              </label>
              <input type="number" min={512} step={1024} value={draft.ctx}
                     onInput={(e) => set({ ctx: (e.target as HTMLInputElement).value })} />
              {kvNote && <div class="msg" style="color:var(--ok)">✓ {kvNote}</div>}
            </div>
            <div class="field">
              <label>TTL <span class="sub">seconds</span>{" "}
                <Info label="about TTL">
                  How long llama-swap keeps the model resident after its
                  last request before unloading it.
                </Info>
              </label>
              <input type="number" min={0} value={draft.ttl}
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
              <label>flash attention{" "}
                <Info label="about flash attention">
                  Faster prompt processing on SYCL. "auto" lets the backend
                  decide from its probed capabilities.
                </Info>
              </label>
              <select value={draft.flash_attn}
                      onChange={(e) => set({ flash_attn: (e.target as HTMLSelectElement).value })}>
                <option value="auto">auto</option>
                <option value="on">on</option>
                <option value="off">off</option>
              </select>
            </div>
            <div class="field">
              <label>KV cache type (K)</label>
              <select value={draft.cache_type_k}
                      onChange={(e) => set({ cache_type_k: (e.target as HTMLSelectElement).value })}>
                {["f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"].map((t) =>
                  <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
            <div class="field">
              <label>KV cache type (V)</label>
              <select value={draft.cache_type_v}
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
              <label>mode{" "}
                <Info label="about the MoE cache">
                  Reserves a VRAM budget per GPU for hot experts. Learning
                  starts at load; watch the hit-ratio meter on operate.
                  Off means experts run where the placement puts them.
                </Info>
              </label>
              <select value={draft.moe_mode}
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
                  <label>{g.device} budget <span class="sub">GiB</span></label>
                  <input type="number" min={0} step={0.5} value={bytesTxt}
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

function fmtBudgetMap(v: unknown): string {
  if (!v || typeof v !== "object") return "—";
  const entries = Object.entries(v as Record<string, number>);
  if (entries.length === 0) return "none";
  return entries.map(([dev, b]) => `${dev} ${fmtGiB(b)} GiB`).join(" · ");
}

export function Model({ name }: { name: string }) {
  const { tick } = useStream();
  const [detail, setDetail] = useState<ModelDetail | null>(null);
  const [err, setErr] = useState("");
  const [reloadN, setReloadN] = useState(0);

  useEffect(() => {
    fetchModelDetail(name).then(setDetail).catch((e) => setErr(String(e)));
  }, [name, reloadN]);

  const live = tick?.models.find((m) => m.name === name) ?? null;

  if (err) {
    return (
      <div class="widget">
        <p>couldn't load {name}: {err}</p>
        <p class="sub"><a href="/v2/models">back to the model hub</a></p>
      </div>
    );
  }
  if (!detail) {
    return <div class="widget"><span class="sub">loading {name}…</span></div>;
  }
  return (
    <>
      <Overview d={detail} live={live} />
      <Plans name={name} />
      <History name={name} />
      <Logs name={name} />
      <Configure d={detail} onSaved={() => setReloadN((n) => n + 1)} />
    </>
  );
}
