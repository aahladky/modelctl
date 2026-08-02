/* /v2/add/:id — the add-model wizard as an SPA: stepper state from the
   server's WizardState, inline job progress from the shared tick stream,
   blocked-advance rendering the server's reason verbatim, and
   single-shot retry that re-issues the step's own job with identical
   params. The register step is the structured editor: ctx max comes from
   the GGUF header read at analyze, cache budgets validate against the
   admission record, warning copy is the server's own. */
import { useEffect, useRef, useState } from "preact/hooks";
import { useStream } from "../lib/stream";
import {
  ApiError, fetchWizard, fetchWizardAnalyze, fetchWizardInspect,
  fetchWizardPlans, fetchWizardRegister, fmtGiB, wizardAnalyzeNext,
  wizardDownloadNext, wizardInspectSubmit, wizardPlansSubmit,
  wizardRegisterSubmit, wizardRetry, wizardSource, wizardTestNext,
  wizardTestRun,
} from "../lib/api";
import { Info } from "../lib/info";
import { toast } from "../lib/toasts";
import type {
  GgufAnalysis, JobRow, ModelDetail, PlanRow, RegisterData, RepoContents,
  WizardDetail, WizardStep,
} from "../lib/types";

const STEP_LABELS: Record<WizardStep, string> = {
  source: "source", inspect: "inspect", download: "download",
  analyze: "analyze", plans: "plans", test: "test",
  register: "register", done: "done",
};

function Stepper({ w }: { w: WizardDetail }) {
  const order = w.steps;
  const at = order.indexOf(w.step);
  return (
    <div class="stepper">
      {order.map((s, i) => (
        <>
          <span key={s}
                class={i < at ? "st done" : i === at ? "st current" : "st"}>
            {i < at ? "✓ " : i === at ? "● " : ""}{STEP_LABELS[s]}
          </span>
          {i < order.length - 1 && <span class="arrow">→</span>}
        </>
      ))}
    </div>
  );
}

function Blocked({ reason, onRetry, retryLabel }:
                 { reason: string; onRetry?: () => void; retryLabel?: string }) {
  if (!reason) return null;
  return (
    <div class="field error" style="margin:.6rem 0">
      <div class="msg error">✗ {reason}</div>
      {onRetry && (
        <div class="actions">
          <button type="button" onClick={onRetry}>{retryLabel ?? "retry this step"}</button>
        </div>
      )}
    </div>
  );
}

/* Inline job progress straight off the shared tick stream: the same rows
   the jobs page renders, scoped to this wizard's job id. */
function JobProgress({ jobId, fallback }:
                     { jobId: string; fallback?: JobRow | null }) {
  const { tick } = useStream();
  const job = tick?.jobs.find((j) => j.id === jobId) ?? fallback ?? null;
  const ref = useRef<HTMLPreElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [job?.result_tail]);
  if (!jobId) return null;
  if (!job) return <p class="sub">job {jobId} not found in the store</p>;
  const chip = job.status === "running" ? "chip active"
    : job.status === "done" ? "chip ok"
    : job.status === "failed" ? "chip err"
    : job.status === "queued" ? "chip" : "chip warn";
  return (
    <div style="margin:.5rem 0">
      <span class={chip}><span class="dot"></span>{job.status}</span>
      <span class="sub"> {job.title} · lane {job.lane}</span>
      <div class="meterbar" style="max-width:420px">
        <div class="fill" style={`width:${Math.round(job.progress * 100)}%`}></div>
      </div>
      <div class="sub num">{job.detail || `${Math.round(job.progress * 100)}%`}</div>
      {job.error && <div class="msg error">✗ {job.error.split("\n")[0]}</div>}
      {job.result_tail.trim() && (
        <pre class="log" style="max-width:560px;max-height:180px" ref={ref}>
          {job.result_tail.trimEnd()}
        </pre>
      )}
    </div>
  );
}

/* ---- steps ----------------------------------------------------------- */

function SourceStep({ w, apply }:
                    { w: WizardDetail; apply: (p: Promise<WizardDetail>) => void }) {
  const [kind, setKind] = useState(w.source_type || "hf_repo");
  const [repo, setRepo] = useState(w.repo_id);
  const [path, setPath] = useState(w.local_path);
  const [err, setErr] = useState("");
  const submit = () => {
    setErr("");
    apply(wizardSource(w.wizard_id, kind === "hf_repo"
      ? { source_type: "hf_repo", repo_id: repo }
      : { source_type: "local_file", local_path: path })
      .catch((e) => { setErr(e instanceof ApiError ? e.message : String(e)); throw e; }));
  };
  return (
    <div class="widget" style="max-width:720px">
      <h2>source</h2>
      <div class="field">
        <label>where does the model come from?{" "}
          <Info label="about sources">
            A Hugging Face repository is inspected first so you can pick a
            quantization. A local GGUF is verified in place — magic bytes,
            shard completeness, readability — before any profile exists.
          </Info>
        </label>
        <div class="check">
          <input type="radio" id="src-hf" checked={kind === "hf_repo"}
                 onChange={() => setKind("hf_repo")} />
          <div><b><label for="src-hf">Hugging Face repository</label></b></div>
        </div>
        <div class="check">
          <input type="radio" id="src-local" checked={kind === "local_file"}
                 onChange={() => setKind("local_file")} />
          <div><b><label for="src-local">local GGUF file</label></b></div>
        </div>
      </div>
      {kind === "hf_repo" ? (
        <div class="field">
          <div class="lbl"><label for="src-repo">repository id</label>
            <Info label="about the repository id">
              The Hugging Face repo holding the GGUF files, written
              owner/name — for example unsloth/Qwen3-30B-GGUF. The next
              step lists the quantizations that repo actually publishes.
            </Info>
          </div>
          <input id="src-repo" type="text" placeholder="org/model-name" value={repo}
                 spellcheck={false}
                 onInput={(e) => setRepo((e.target as HTMLInputElement).value)} />
        </div>
      ) : (
        <div class="field">
          <div class="lbl">
            <label for="src-path">absolute path on this machine</label>
            <Info label="about the local path">
              An absolute path to a .gguf already on this machine; nothing
              is copied. For a sharded model point at the first shard
              (…-00001-of-000NN.gguf) and the rest are found beside it.
            </Info>
          </div>
          <input id="src-path" type="text" placeholder="/path/to/model.gguf" value={path}
                 spellcheck={false}
                 onInput={(e) => setPath((e.target as HTMLInputElement).value)} />
        </div>
      )}
      {err && <div class="msg error">✗ {err}</div>}
      <div class="actions" style="justify-content:flex-end">
        <button type="button" class="btn-primary" onClick={submit}>continue →</button>
      </div>
    </div>
  );
}

function InspectStep({ w, apply }:
                     { w: WizardDetail; apply: (p: Promise<WizardDetail>) => void }) {
  const [contents, setContents] = useState<RepoContents | null>(null);
  const [err, setErr] = useState("");
  const [quant, setQuant] = useState(w.selected_quant);
  useEffect(() => {
    fetchWizardInspect(w.wizard_id)
      .then((r) => { setContents(r.contents); setErr(r.error); })
      .catch((e) => setErr(String(e)));
  }, [w.wizard_id]);
  return (
    <div class="widget" style="max-width:760px">
      <h2>inspect — {w.repo_id}</h2>
      {!contents && !err && <p class="sub">fetching repository contents…</p>}
      {err && <Blocked reason={err} />}
      {contents && (
        <>
          <div class="field">
            <label>quantization{" "}
              <Info label="about quantizations">
                Smaller quants fit more of the model in VRAM at some
                quality cost. The recommended pick balances the two for
                this machine; you can override it.
              </Info>
            </label>
            <table>
              <tbody>
                {contents.quant_groups.map((g) => (
                  <tr key={g.label}>
                    <td style="width:30px">
                      <input type="radio" checked={quant === g.label}
                             onChange={() => setQuant(g.label)} />
                    </td>
                    <td>{g.label}
                      <span class="sub"> · {fmtGiB(g.total_size)} GiB
                        {g.sharded ? " · sharded" : ""}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div class="actions" style="justify-content:flex-end">
            <button type="button" class="btn-primary"
                    onClick={() => apply(wizardInspectSubmit(w.wizard_id, quant))}>
              download →
            </button>
          </div>
        </>
      )}
    </div>
  );
}

function DownloadStep({ w, apply, retry }:
                      { w: WizardDetail;
                        apply: (p: Promise<WizardDetail>) => void;
                        retry: (step: string) => void }) {
  const [blocked, setBlocked] = useState("");
  const gate = w.step_gates["download"];
  const next = () => {
    setBlocked("");
    apply(wizardDownloadNext(w.wizard_id).catch((e) => {
      if (e instanceof ApiError && e.status === 409) setBlocked(e.message);
      throw e;
    }));
  };
  const failed = gate?.outcome?.status === "failed"
    || w.jobs["download"]?.status === "failed";
  return (
    <div class="widget" style="max-width:760px">
      <h2>download</h2>
      <p class="sub">
        {w.source_type === "local_file"
          ? "verifying and importing the local file — no bytes are copied unless asked"
          : `pulling ${w.repo_id}${w.selected_quant ? ` (${w.selected_quant})` : ""}`}
      </p>
      {w.download_job_id
        ? <JobProgress jobId={w.download_job_id} fallback={w.jobs["download"]} />
        : <p class="sub">no acquisition job yet</p>}
      <Blocked reason={blocked || (failed ? gate?.blocking_reason ?? "" : "")}
               onRetry={failed ? () => retry("download") : undefined}
               retryLabel="retry download (same source, same parameters)" />
      <div class="actions" style="justify-content:flex-end">
        {/* the server already told us whether this step may advance;
            disabling up front beats letting the user click into a 409 */}
        {gate?.blocking_reason && (
          <span class="sub stale-note">blocked: {gate.blocking_reason}</span>
        )}
        <button type="button" class="btn-primary" onClick={next}
                disabled={!!gate?.blocking_reason}>
          continue to analyze →
        </button>
      </div>
    </div>
  );
}

function AnalyzeStep({ w, apply }:
                     { w: WizardDetail; apply: (p: Promise<WizardDetail>) => void }) {
  const [profile, setProfile] = useState<ModelDetail | null>(null);
  const [analysis, setAnalysis] = useState<GgufAnalysis | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [blocked, setBlocked] = useState("");
  useEffect(() => {
    fetchWizardAnalyze(w.wizard_id)
      .then((r) => { setProfile(r.profile); setAnalysis(r.analysis); setLoaded(true); })
      .catch(() => setLoaded(true));
  }, [w.wizard_id, w.profile_name]);
  return (
    <div class="widget" style="max-width:760px">
      <h2>analyze{" "}
        <Info label="about analysis">
          Facts read from the GGUF header itself — architecture, trained
          context maximum, layer count. The register step's limits come
          from these numbers, not from anything typed in.
        </Info>
      </h2>
      {!loaded && <p class="sub">reading the GGUF header…</p>}
      {loaded && !profile && (
        <p class="sub">the acquisition job hasn't produced a profile yet —
          it may still be running (see the download step)</p>
      )}
      {profile && (
        <table>
          <tbody>
            <tr><td class="sub">profile</td><td>{profile.name}</td></tr>
            <tr><td class="sub">weights on disk</td>
              <td class="num">{profile.size_bytes != null
                ? `${fmtGiB(profile.size_bytes)} GiB` : "—"}</td></tr>
            {analysis ? (
              <>
                <tr><td class="sub">architecture</td>
                  <td>{analysis.arch}{analysis.is_moe ? " · MoE" : ""}
                    {analysis.expert_count ? ` (${analysis.expert_count} experts)` : ""}</td></tr>
                <tr><td class="sub">trained context max</td>
                  <td class="num">{analysis.model_max_ctx ?? "—"}
                    <span class="sub"> tokens (from the GGUF header)</span></td></tr>
                <tr><td class="sub">layers</td>
                  <td class="num">{analysis.block_count ?? "—"}</td></tr>
                {analysis.kv_bytes_per_token != null && (
                  <tr><td class="sub">KV cache</td>
                    <td class="num">
                      {(analysis.kv_bytes_per_token / 1024).toFixed(1)}
                      <span class="sub"> KiB/token at the profile's cache types</span>
                    </td></tr>
                )}
              </>
            ) : (
              <tr><td class="sub">header</td>
                <td class="sub">unreadable — placement falls back to
                  heuristic estimates</td></tr>
            )}
          </tbody>
        </table>
      )}
      <Blocked reason={blocked} />
      <div class="actions" style="justify-content:flex-end">
        <button type="button" class="btn-primary" onClick={() => {
          setBlocked("");
          apply(wizardAnalyzeNext(w.wizard_id).catch((e) => {
            if (e instanceof ApiError && e.status === 409) setBlocked(e.message);
            throw e;
          }));
        }}>continue to plans →</button>
      </div>
    </div>
  );
}

function PlansStep({ w, apply }:
                   { w: WizardDetail; apply: (p: Promise<WizardDetail>) => void }) {
  const [plans, setPlans] = useState<PlanRow[] | null>(null);
  const [err, setErr] = useState("");
  const [picked, setPicked] = useState(w.selected_plan_id);
  useEffect(() => {
    fetchWizardPlans(w.wizard_id)
      .then((r) => {
        setPlans(r.plans);
        if (!picked && r.plans.length > 0) setPicked(r.plans[0].id);
      })
      .catch((e) => setErr(String(e)));
  }, [w.wizard_id]);
  return (
    <div class="widget" style="max-width:860px">
      <h2>plans{" "}
        <Info label="about plans">
          Each plan is a complete launch configuration with its resource
          math. Numbers are estimated until the test step measures them —
          the tag says which you're looking at.
        </Info>
      </h2>
      {err && <Blocked reason={err} />}
      {!plans && !err && <p class="sub">compiling plans…</p>}
      {plans && (
        <table>
          <tbody>
            {plans.map((p) => (
              <tr key={p.id}>
                <td style="width:30px">
                  <input type="radio" checked={picked === p.id}
                         onChange={() => setPicked(p.id)} />
                </td>
                <td>{p.label}
                  <span style="margin-left:.4em">
                    {p.measured
                      ? <span class="tag measured">measured</span>
                      : <span class="tag estimated">estimated</span>}
                  </span>
                  {p.warnings.length > 0 && (
                    <div class="sub" style="color:var(--warn)">
                      {p.warnings.map((wn) => <div key={wn}>⚠ {wn}</div>)}
                    </div>
                  )}
                </td>
                <td class="num">
                  {p.measured?.generation_tps != null
                    ? `${p.measured.generation_tps.toFixed(1)} tok/s`
                    : ""}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <div class="actions" style="justify-content:flex-end">
        <button type="button" disabled={!plans || !picked}
                onClick={() => apply(wizardPlansSubmit(w.wizard_id, picked, "register"))}>
          skip test, register untested →
        </button>
        <button type="button" class="btn-primary" disabled={!plans || !picked}
                onClick={() => apply(wizardPlansSubmit(w.wizard_id, picked, "test"))}>
          test this plan →
        </button>
      </div>
    </div>
  );
}

function TestStep({ w, apply, retry }:
                  { w: WizardDetail;
                    apply: (p: Promise<WizardDetail>) => void;
                    retry: (step: string) => void }) {
  const [blocked, setBlocked] = useState("");
  const gate = w.step_gates["test"];
  const job = w.jobs["test"] ?? null;
  const failed = gate?.outcome?.status === "failed" || job?.status === "failed";
  return (
    <div class="widget" style="max-width:760px">
      <h2>test{" "}
        <Info label="about the test">
          Launches a real llama-server with the selected plan on a scratch
          port and measures load time and throughput. The measured numbers
          land in the register step and the measurement store — this is
          what turns "estimated" into "measured".
        </Info>
      </h2>
      {w.test_job_id
        ? <JobProgress jobId={w.test_job_id} fallback={job} />
        : <p class="sub">no test has been run yet</p>}
      <Blocked reason={blocked || (failed ? gate?.blocking_reason ?? "" : "")}
               onRetry={failed ? () => retry("test") : undefined}
               retryLabel="retry test (same plan, same parameters)" />
      <div class="actions" style="justify-content:flex-end">
        {!w.test_job_id && (
          <button type="button" class="btn-primary"
                  onClick={() => apply(wizardTestRun(w.wizard_id))}>
            run the test
          </button>
        )}
        {w.test_job_id && (
          <button type="button" class="btn-primary" onClick={() => {
            setBlocked("");
            apply(wizardTestNext(w.wizard_id).catch((e) => {
              if (e instanceof ApiError && e.status === 409) setBlocked(e.message);
              throw e;
            }));
          }}>continue to register →</button>
        )}
      </div>
    </div>
  );
}

interface RegDraft {
  ctx: string;
  flash_attn: string;
  cache_enabled: boolean;
  budgets: Record<string, string>;
}

function RegisterStep({ w, apply, retry }:
                      { w: WizardDetail;
                        apply: (p: Promise<WizardDetail>) => void;
                        retry: (step: string) => void }) {
  const [data, setData] = useState<RegisterData | null>(null);
  const [err, setErr] = useState("");
  const [draft, setDraft] = useState<RegDraft | null>(null);
  const [confirmUntested, setConfirmUntested] = useState(false);
  const [busy, setBusy] = useState(false);
  const debounce = useRef<ReturnType<typeof setTimeout> | null>(null);

  /* first load: profile + analysis + admission for current values */
  useEffect(() => {
    fetchWizardRegister(w.wizard_id)
      .then((d) => {
        setData(d);
        const budgets: Record<string, string> = {};
        for (const g of d.profile.gpus) {
          const b = d.profile.moe_cache.budgets_bytes[g.device];
          budgets[g.device] = b ? (b / 2 ** 30).toFixed(1) : "";
        }
        setDraft({
          ctx: String(d.profile.config.ctx || ""),
          flash_attn: String(d.profile.config.flash_attn || "auto"),
          cache_enabled: d.profile.moe_cache.mode !== "off",
          budgets,
        });
      })
      .catch((e) => setErr(String(e)));
  }, [w.wizard_id]);

  /* draft edits re-ask the server so warnings stay the server's words */
  useEffect(() => {
    if (!draft) return;
    if (debounce.current) clearTimeout(debounce.current);
    debounce.current = setTimeout(() => {
      const ctx = parseInt(draft.ctx, 10);
      const budgets: Record<string, number> = {};
      for (const [dev, gib] of Object.entries(draft.budgets)) {
        const v = parseFloat(gib);
        if (!Number.isNaN(v) && v > 0) budgets[dev] = Math.round(v * 2 ** 30);
      }
      fetchWizardRegister(w.wizard_id, Number.isNaN(ctx) ? null : ctx,
                          draft.cache_enabled ? budgets : null,
                          draft.cache_enabled ? "auto" : "off")
        .then((d) => setData((old) => old ? { ...old, admission: d.admission } : d))
        .catch(() => { /* preview refresh is best-effort */ });
    }, 400);
    return () => {
      if (debounce.current) clearTimeout(debounce.current);
    };
  }, [w.wizard_id, draft?.ctx, draft?.cache_enabled,
      JSON.stringify(draft?.budgets ?? {})]);

  if (err) return <div class="widget"><Blocked reason={err} /></div>;
  if (!data || !draft) {
    return <div class="widget"><span class="sub">loading register form…</span></div>;
  }

  const maxCtx = data.analysis?.model_max_ctx ?? null;
  const kvPerToken = data.analysis?.kv_bytes_per_token ?? null;
  const ctxNum = parseInt(draft.ctx, 10);
  const ctxError = Number.isNaN(ctxNum)
    ? "enter a context length"
    : maxCtx != null && ctxNum > maxCtx
      ? `above the model's trained maximum (${maxCtx.toLocaleString()}, read from the GGUF header at analyze)`
      : ctxNum < 512
        ? "below the 512-token minimum the server accepts"
        : "";
  const kvNote = !ctxError && kvPerToken != null
    ? `KV cache ≈ ${(ctxNum * kvPerToken / 2 ** 30).toFixed(1)} GiB at this setting`
    : "";

  const admission = data.admission?.plan?.admission ?? null;
  const warnings = data.admission?.plan?.warnings ?? [];
  let warningCount = warnings.length;

  const testBlocked = data.test_gate.blocking_reason;
  const errorCount = ctxError ? 1 : 0;

  const submit = async (untested: boolean) => {
    setBusy(true);
    const budgets: Record<string, number> = {};
    for (const [dev, gib] of Object.entries(draft.budgets)) {
      const v = parseFloat(gib);
      if (!Number.isNaN(v) && v > 0) budgets[dev] = Math.round(v * 2 ** 30);
    }
    // The catch attaches BEFORE apply(): apply swallows already-handled
    // rejections, so the handling has to ride the promise itself.
    await apply(wizardRegisterSubmit(w.wizard_id, {
      ctx: Number.isNaN(ctxNum) ? undefined : ctxNum,
      flash_attn: draft.flash_attn,
      moe_mode: draft.cache_enabled ? "auto" : "off",
      budgets_bytes: draft.cache_enabled ? budgets : {},
      register_untested: untested,
    }).catch((e) => {
      if (e instanceof ApiError && e.status === 409
          && e.body.requires_register_untested) {
        setConfirmUntested(true);
      } else {
        toast("err", "✗ register failed", String(e), 9000);
      }
      throw e;
    }));
    setBusy(false);
  };

  return (
    <div class="widget" style="max-width:880px">
      <h2>register profile</h2>
      <p class="sub" style="margin:.2rem 0 .9rem">
        Everything below writes the profile the service launches from. The
        file on disk stays hand-editable from the shell; this form is the
        only editor the console offers.
      </p>
      <form style="display:grid;gap:1rem" onSubmit={(e) => e.preventDefault()}>
        <fieldset>
          <legend>profile</legend>
          <div class="frow">
            <div class="field" style="flex:2 1 220px">
              <div class="lbl"><label for="reg-name">profile name</label>
                <Info label="about the profile name">
                  Derived from the repo and quantization at download and
                  fixed from then on: it is the id in the llama-swap config
                  and the name every page and log line uses. Rename with
                  the CLI if you must.
                </Info>
              </div>
              <input id="reg-name" type="text" value={data.profile.name} disabled />
            </div>
            <div class={ctxError ? "field error" : "field"}>
              <div class="lbl">
                <label for="reg-ctx">context length <span class="sub">tokens</span></label>
                <Info label="about context length">
                  KV cache grows linearly with this; it is the biggest
                  single VRAM lever. The ceiling is the trained maximum
                  read from the GGUF header at the analyze step.
                </Info>
              </div>
              <input id="reg-ctx" type="number" min={512} step={1024} value={draft.ctx}
                     onInput={(e) => setDraft({ ...draft, ctx: (e.target as HTMLInputElement).value })} />
              {ctxError
                ? <div class="msg error">✗ {ctxError}</div>
                : <div class="msg" style="color:var(--ok)">
                    ✓ within the model maximum{kvNote ? ` · ${kvNote}` : ""}
                  </div>}
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>placement <span class="sub">from the selected plan</span></legend>
          <div class="sub">
            {data.profile.config.tensor_split
              ? `split ${data.profile.config.tensor_split} (${data.profile.config.split_mode || "layer"})`
              : (data.profile.config.device || "backend default")}
            {data.measured && Object.keys(data.measured).length > 0 && (
              <>
                {" "}<span class="tag measured">measured</span>{" "}
                {data.measured["generation_tps"] != null
                  ? `${Number(data.measured["generation_tps"]).toFixed(1)} tok/s with this plan`
                  : ""}
              </>
            )}
            {(!data.measured || Object.keys(data.measured).length === 0) && (
              <> <span class="tag estimated">estimated</span> — the plan was
                not tested</>
            )}
          </div>
          {data.profile.config.extra && (
            <div class="sub">{data.profile.config.extra}</div>
          )}
        </fieldset>

        <fieldset>
          <legend>MoE cache</legend>
          <div class="check">
            <input type="checkbox" id="reg-cache" checked={draft.cache_enabled}
                   onChange={(e) => setDraft({
                     ...draft,
                     cache_enabled: (e.target as HTMLInputElement).checked })} />
            <div class="lbl">
              <b><label for="reg-cache">enable expert cache</label></b>
              <Info label="about the expert cache">
                Reserves a VRAM budget per GPU for the experts this model
                actually reuses. Learning starts at load and the hit-ratio
                meter on operate reports learning → engaged. Off means
                experts run wherever the placement put them.
              </Info>
            </div>
          </div>
          {draft.cache_enabled && (
            <div class="frow">
              {data.profile.gpus.map((g) => {
                const dev = admission?.devices?.[g.device];
                const gib = draft.budgets[g.device] ?? "";
                const wanted = parseFloat(gib);
                const headroom = dev
                  ? (dev.usable_bytes - (dev.demand_bytes - dev.cache_bytes)) / 2 ** 30
                  : null;
                const over = headroom != null && !Number.isNaN(wanted)
                  && wanted > Math.max(0, headroom);
                if (over) warningCount += 1;
                return (
                  <div class={over ? "field warning" : "field"} key={g.device}>
                    <label for={`reg-budget-${g.device}`}>{g.device} budget{" "}
                      <span class="sub">GiB</span></label>
                    <input id={`reg-budget-${g.device}`} type="number" min={0}
                           step={0.5} value={gib}
                           onInput={(e) => setDraft({
                             ...draft,
                             budgets: { ...draft.budgets,
                                        [g.device]: (e.target as HTMLInputElement).value } })} />
                    <div class="help">{g.name} · {fmtGiB(g.total_bytes, 0)} GiB
                      {headroom != null
                        ? <> · {Math.max(0, headroom).toFixed(1)} GiB admissible{" "}
                            <span class="tag measured">measured</span></>
                        : null}</div>
                    {/* the fit case was silent; "measurements outrank
                        estimates" has to be visible where the number is
                        being chosen, not only when it is wrong */}
                    {!over && headroom != null && !Number.isNaN(wanted) && (
                      <div class="msg" style="color:var(--ok)">
                        ✓ fits the measured headroom
                        ({Math.max(0, headroom).toFixed(1)} GiB)
                      </div>
                    )}
                    {over && headroom != null && (
                      <div class="msg warning">
                        ⚠ exceeds the admissible headroom
                        ({Math.max(0, headroom).toFixed(1)} GiB) — launch
                        will fall back to a smaller cache
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
          {warnings.map((wn) => (
            <div class="msg warning" key={wn}>⚠ {wn}</div>
          ))}
        </fieldset>

        <fieldset>
          <legend>runtime flags</legend>
          <div class="frow">
            <div class="field" style="max-width:220px">
              <label>flash attention{" "}
                <Info label="about flash attention">
                  Faster prompt processing on SYCL; "auto" lets the probed
                  backend capabilities decide.
                </Info>
              </label>
              <select value={draft.flash_attn}
                      onChange={(e) => setDraft({
                        ...draft,
                        flash_attn: (e.target as HTMLSelectElement).value })}>
                <option value="auto">auto</option>
                <option value="on">on</option>
                <option value="off">off</option>
              </select>
            </div>
          </div>
        </fieldset>

        {testBlocked && !confirmUntested && (
          <Blocked reason={`test gate: ${testBlocked}`}
                   onRetry={() => retry("test")}
                   retryLabel="go back and re-run the test" />
        )}

        {confirmUntested && (
          <fieldset style="border-color:var(--warn)">
            <legend style="color:var(--warn)">register untested?</legend>
            <p class="sub">The test step did not pass
              ({testBlocked || "no successful test"}). Registering anyway
              means the first real launch is the test.</p>
            <div class="actions">
              <button type="button" onClick={() => setConfirmUntested(false)}>
                back</button>
              <button type="button" class="btn-danger" disabled={busy}
                      onClick={() => submit(true)}>
                register untested</button>
            </div>
          </fieldset>
        )}

        <div class="actions" style="justify-content:flex-end">
          <span class="sub" style="margin-right:auto">
            {errorCount > 0 && (
              <span style="color:var(--err);font-weight:600">
                {errorCount} error{errorCount > 1 ? "s" : ""}</span>
            )}
            {errorCount > 0 && warningCount > 0 && " · "}
            {warningCount > 0 && (
              <span style="color:var(--warn);font-weight:600">
                {warningCount} warning{warningCount > 1 ? "s" : ""}</span>
            )}
            {errorCount > 0
              ? " — fix the error to continue"
              : warningCount > 0
                ? " — warnings don't block; the fallback is safe"
                : "all checks pass"}
          </span>
          <button type="button" class="btn-primary"
                  disabled={busy || errorCount > 0}
                  title={errorCount ? "blocked: fix the error above first" : undefined}
                  onClick={() => submit(false)}>
            register &amp; finish
          </button>
        </div>
      </form>
    </div>
  );
}

function DoneStep({ w }: { w: WizardDetail }) {
  const measured = w.measured && Object.keys(w.measured).length > 0
    ? w.measured
    : (w.selected_plan_id ? w.test_observations[w.selected_plan_id] : null);
  return (
    <div class="widget" style="max-width:720px">
      <h2>done — {w.profile_name}</h2>
      <table>
        <tbody>
          <tr><td class="sub">endpoint</td>
            <td class="num">{w.endpoint || "—"}</td></tr>
          <tr><td class="sub">command fingerprint</td>
            <td class="num">{w.command_fingerprint || "—"}</td></tr>
          <tr><td class="sub">measured</td>
            <td>{measured && Object.keys(measured).length > 0
              ? <>
                  <span class="tag measured">measured</span>{" "}
                  {measured["generation_tps"] != null
                    ? `${Number(measured["generation_tps"]).toFixed(1)} tok/s generation` : ""}
                  {measured["load_seconds"] != null
                    ? ` · ${Number(measured["load_seconds"]).toFixed(0)}s load` : ""}
                </>
              : <><span class="tag estimated">not measured</span> — registered
                  without a passing test</>}</td></tr>
        </tbody>
      </table>
      <p class="sub">
        A warm-load verification job was queued — watch it on the{" "}
        <a href="/v2/jobs">jobs page</a>. The model now appears in the{" "}
        <a href={`/v2/models/${encodeURIComponent(w.profile_name)}`}>model hub</a>.
      </p>
    </div>
  );
}

/* ---- page ------------------------------------------------------------ */

export function Wizard({ id }: { id: string }) {
  const { tick } = useStream();
  const [w, setW] = useState<WizardDetail | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    fetchWizard(id).then(setW).catch((e) => setErr(String(e)));
  }, [id]);

  /* When a wizard job finishes in the stream, refresh the wizard so
     step outcomes fold in without a manual reload. The updated_at guard
     keeps an in-flight (stale) GET from overwriting a fresher POST
     response that landed first. */
  const takeNewer = (next: WizardDetail) =>
    setW((old) => (old && old.updated_at > next.updated_at ? old : next));
  const jobStatuses = (tick?.jobs ?? [])
    .filter((j) => j.id === w?.download_job_id || j.id === w?.test_job_id)
    .map((j) => `${j.id}:${j.status}`)
    .join(",");
  useEffect(() => {
    if (!w) return;
    fetchWizard(id).then(takeNewer).catch(() => { /* keep last state */ });
  }, [jobStatuses]);

  const apply = (p: Promise<WizardDetail>) => {
    return p.then((next) => { takeNewer(next); }).catch(() => { /* handled at call site */ });
  };
  const retry = (step: string) => {
    wizardRetry(id, step)
      .then((next) => {
        setW(next);
        toast("ok", `${step} step reset`,
              "its job was re-issued with identical parameters");
      })
      .catch((e) => toast("err", `✗ retry ${step} failed`, String(e), 9000));
  };

  if (err) {
    return (
      <div class="widget">
        <p>couldn't load this wizard: {err}</p>
        <p class="sub"><a href="/v2/add">back to add</a></p>
      </div>
    );
  }
  if (!w) return <div class="widget"><span class="sub">loading wizard…</span></div>;

  const title = w.repo_id || w.local_path || "new model";
  return (
    <>
      <div class="widget" style="background:transparent;border:none;padding:0">
        <div class="sub" style="margin-bottom:.4rem">add model — {title}</div>
        <Stepper w={w} />
      </div>
      {w.step === "source" && <SourceStep w={w} apply={apply} />}
      {w.step === "inspect" && <InspectStep w={w} apply={apply} />}
      {w.step === "download" && <DownloadStep w={w} apply={apply} retry={retry} />}
      {w.step === "analyze" && <AnalyzeStep w={w} apply={apply} />}
      {w.step === "plans" && <PlansStep w={w} apply={apply} />}
      {w.step === "test" && <TestStep w={w} apply={apply} retry={retry} />}
      {w.step === "register" && <RegisterStep w={w} apply={apply} retry={retry} />}
      {w.step === "done" && <DoneStep w={w} />}
    </>
  );
}
