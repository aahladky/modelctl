/* operate: live meters, sparklines, in-place row updates. Everything on
   this page renders from the tick stream; when the stream drops, every
   live region flips to the stale treatment (dashed border, last-known
   value + timestamp + retry countdown) instead of blanking. */
import { useEffect, useRef, useState } from "preact/hooks";
import { useStream } from "../lib/stream";
import { Spark, push } from "../lib/spark";
import { Info } from "../lib/info";
import { ConfirmButton, submitAction } from "../lib/actions";
import { fmtClock, fmtGiB, fmtUp, loadModel, unloadAll, unloadModel }
  from "../lib/api";
import type { ModelRow, Tick } from "../lib/types";

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

function LiveMark({ stale }: { stale: boolean }) {
  return stale
    ? <span class="live stale"><span class="dot"></span>stale</span>
    : <span class="live"><span class="dot"></span>live</span>;
}

function StaleSub({ lastAt, retryIn }: { lastAt: number | null; retryIn: number | null }) {
  return (
    <div class="sub">
      last known {lastAt != null ? fmtClock(lastAt) : "—"} ·{" "}
      <span>{retryIn != null ? `retrying in ${retryIn}s` : "reconnecting…"}</span>
    </div>
  );
}

/* Mutations go through the typed /api/v2 lane and answer with the job id;
   the job itself shows up in the stream, so the SPA toasts and lets the
   jobs badge/page track it. (Phase 3: these used to POST to the old
   console's /models/{name}/{verb} routes, which the cutover removed.
   Phase 4: submitAction moved to lib/actions, shared with the model hub,
   the model page and settings.) */

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

function ModelRowView({ m, spark, stale }:
                      { m: ModelRow; spark: number[]; stale: boolean }) {
  const [busy, setBusy] = useState(false);
  const chipClass = m.state_class ? `chip ${m.state_class}` : "chip";
  const act = (verb: "load" | "unload") => {
    setBusy(true);
    submitAction(() => (verb === "load" ? loadModel : unloadModel)(m.name),
                 `${verb} ${m.name}`).finally(() => setBusy(false));
  };
  return (
    <tr class="rowlink"
        onClick={(e) => {
          if ((e.target as HTMLElement).closest("button")) return;
          location.href = `/v2/models/${encodeURIComponent(m.name)}`;
        }}>
      <td>
        {m.name}
        <div class="sub">
          {[m.size_bytes != null ? `${fmtGiB(m.size_bytes)} GiB` : "",
            m.file,
            m.backend !== "llama-cpp" ? m.backend : ""]
            .filter(Boolean).join(" · ")}
        </div>
      </td>
      <td class="sub">{m.placement}</td>
      <td><span class={chipClass}><span class="dot"></span>{m.state}</span></td>
      <td class={m.tok_s != null ? "num" : "num sub"}>
        {m.tok_s != null ? m.tok_s.toFixed(1) : "—"}
      </td>
      <td>
        {spark.length > 1
          ? <Spark data={spark} height={26} label={`${m.name} tok/s`} />
          : null}
      </td>
      <td class="actions">
        {m.running
          ? <button class={busy ? "busy" : undefined} disabled={busy || stale}
                    onClick={() => act("unload")}>unload</button>
          : m.registered && m.enabled
            ? <button class={busy ? "busy" : undefined} disabled={busy || stale}
                      onClick={() => act("load")}>load</button>
            : null}
      </td>
    </tr>
  );
}

export function Operate() {
  const { tick, stale, lastAt, retryIn } = useStream();
  const series = useSeries(stale ? null : tick);

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
  /* Per-region degradation. The stream can be perfectly healthy while one
     server-side probe throws: `gpus` then arrives empty and `ram` arrives
     zeroed, both indistinguishable from fact. tick.errors names the
     section that failed, so the region marks itself instead of quietly
     lying. */
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

  return (
    <>
      <div class="widget" style="display:flex;gap:1.2rem;align-items:center;flex-wrap:wrap">
        <span class={services.swap.ok ? "chip ok" : "chip err"}>
          <span class="dot"></span>
          llama-swap · {services.swap.ok ? "running" : "down"} · {services.swap.detail}
          {services.swap.ok && services.swap.latency_ms != null ? ` · ${services.swap.latency_ms}ms` : ""}
        </span>
        <span class={services.api.ok ? "chip ok" : "chip err"}>
          <span class="dot"></span>
          API · {services.api.ok ? "ok" : "down"}
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
        <span class="sub num">console up {fmtUp(services.console_started)}</span>
        <span class="grow" style="flex:1"></span>
        <a href="/v2/jobs" class="sub">
          {runningJobs} job{runningJobs === 1 ? "" : "s"} running →
        </a>
      </div>

      <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(200px,1fr))">
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
                <span>{g.device} · {g.name}</span>
                <LiveMark stale={bad("gpus")} />
              </div>
              <div class="meterbar">
                <div class="fill" style={`width:${total ? Math.min(100, (used / total) * 100) : 0}%`}></div>
              </div>
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
          <div class="meterbar">
            <div class="fill" style={`width:${ram.total_bytes ? Math.min(100, (ram.used_bytes / ram.total_bytes) * 100) : 0}%`}></div>
          </div>
          <div class="big">
            {/* a failed meminfo read yields 0/0; print an em dash rather
                than a number that reads as a measurement */}
            {err["ram"] ? "—" : fmtGiB(ram.used_bytes)}
            <span class="sub"> / {err["ram"] ? "—" : fmtGiB(ram.total_bytes, 0)} GiB</span>
          </div>
          <Spark data={series["ram"] ?? []} min={0} max={ram.total_bytes / 2 ** 30}
                 label="RAM used" />
          {sectionNote("ram")}
        </div>

        {cacheModels.map((m) => {
          const c = m.cache!;
          const pct = c.hit_ratio != null ? c.hit_ratio * 100 : null;
          return (
            <div class={cls("models")} key={`cache-${m.name}`}>
              <div class="label">
                <span>
                  {m.name} · MoE cache{" "}
                  <Info label="about the MoE cache">
                    Expert-cache hit ratio for this model, scraped from its
                    runtime metrics. While the cache is still learning which
                    experts recur, the ratio is provisional; "learning done"
                    means placement has settled.
                  </Info>
                </span>
                <LiveMark stale={bad("models")} />
              </div>
              <div class="meterbar">
                <div class="fill" style={`width:${pct ?? 0}%`}></div>
              </div>
              <div class="big">
                {pct != null ? pct.toFixed(1) : "—"}
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
        {cacheModels.length === 0 && (
          <div class="widget">
            <div class="label"><span>MoE cache</span></div>
            <div class="sub" style="margin-top:.4rem">
              no resident model is running with an MoE cache
            </div>
          </div>
        )}
      </div>

      <div class={cls("models")}>
        <div class="label">
          <h2 style="margin:0">resident models</h2>
          {/* the one fleet-wide action, next to the fleet it acts on */}
          <span class="actions">
            <UnloadAll running={models.filter((m) => m.running).length}
                       stale={stale} />
          </span>
        </div>
        {models.length === 0
          ? (err["models"]
              ? <p class="sub stale-note">could not read the profile store —
                  {" "}{err["models"]}. This is not an empty install.</p>
              : <p class="sub">no profiles registered yet — <a href="/v2/add">add a model</a></p>)
          : (
            <table>
              <thead>
                <tr>
                  <th>model</th><th>placement</th><th>state</th>
                  <th class="num">tok/s</th><th style="width:130px">last 60 s</th><th></th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <ModelRowView key={m.name} m={m}
                                spark={series[`tok:${m.name}`] ?? []} stale={stale} />
                ))}
              </tbody>
            </table>
          )}
        <p class="sub" style="margin:.5rem 0 0">
          rows update in place over SSE · click a row for the model hub
          (overview · plans · measurements · logs · configure)
        </p>
        {sectionNote("models")}
      </div>
    </>
  );
}
