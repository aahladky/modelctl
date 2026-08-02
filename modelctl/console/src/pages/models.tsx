/* model hub list: every profile, live state from the tick stream, click
   through to the per-model page. Rendering comes straight from the same
   ModelRow rows operate uses -- no second source of truth.

   Phase 4 puts the measurement triggers here: benchmark, smoke test and
   autotune are all "run something against this model and record what it
   measures", and the hub is the one page that lists every model side by
   side. They submit onto the benchmark lane and are followed on the jobs
   page like every other job. */
import { useState } from "preact/hooks";
import { useStream } from "../lib/stream";
import { autotuneModel, benchModel, fmtGiB, smokeModel } from "../lib/api";
import { submitAction } from "../lib/actions";
import { Info } from "../lib/info";
import type { ModelRow } from "../lib/types";

/* The old bench form carried these two overrides and they are not
   cosmetic: an SSD-mmap model generates well under 1 tok/s, so the
   default 256x3 takes the better part of an hour to measure a steady
   state a smaller budget reaches in minutes. The server clamps to the
   same 1..4096 / 1..10 it always did. */
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

function RowActions({ m }: { m: ModelRow }) {
  const [benching, setBenching] = useState(false);
  const [busy, setBusy] = useState("");
  const run = (key: string, fn: () => Promise<{ job_id: string }>,
               label: string) => {
    setBusy(key);
    submitAction(fn, label).finally(() => setBusy(""));
  };
  return (
    <>
      <span class="actions">
        <button type="button" class={busy === "bench" ? "busy" : undefined}
                onClick={() => setBenching(!benching)}>bench</button>
        <button type="button" class={busy === "smoke" ? "busy" : undefined}
                disabled={!!busy}
                onClick={() => run("smoke", () => smokeModel(m.name),
                                   `smoke test ${m.name}`)}>smoke</button>
        <button type="button" class={busy === "tune" ? "busy" : undefined}
                disabled={!!busy}
                onClick={() => run("tune",
                                   () => autotuneModel(m.name, "balanced"),
                                   `autotune ${m.name}`)}>autotune</button>
      </span>
      {benching && <BenchForm name={m.name} onDone={() => setBenching(false)} />}
    </>
  );
}

function HubRow({ m }: { m: ModelRow }) {
  const chipClass = m.state_class ? `chip ${m.state_class}` : "chip";
  const open = () => {
    location.href = `/v2/models/${encodeURIComponent(m.name)}`;
  };
  return (
    <tr class="rowlink"
        onClick={(e) => {
          /* the row is a link and the cell is full of buttons; a click
             that started on a control must not also navigate */
          const el = e.target as HTMLElement;
          if (el.closest("button") || el.closest("input")) return;
          open();
        }}>
      <td>
        <a href={`/v2/models/${encodeURIComponent(m.name)}`}
           onClick={(e) => e.stopPropagation()}>{m.name}</a>
        <div class="sub">
          {[m.size_bytes != null ? `${fmtGiB(m.size_bytes)} GiB` : "",
            m.file,
            m.backend !== "llama-cpp" ? m.backend : ""]
            .filter(Boolean).join(" · ")}
        </div>
      </td>
      <td class="sub">{m.placement}</td>
      <td><span class={chipClass}><span class="dot"></span>{m.state}</span></td>
      <td class="sub">{m.moe_cache_mode !== "off" ? `cache ${m.moe_cache_mode}` : "—"}</td>
      <td class={m.tok_s != null ? "num" : "num sub"}>
        {m.tok_s != null ? m.tok_s.toFixed(1) : "—"}
      </td>
      <td class="sub">{m.enabled ? "" : "disabled"}</td>
      <td><RowActions m={m} /></td>
    </tr>
  );
}

export function Models() {
  const { tick, stale } = useStream();

  if (!tick) {
    return <div class="widget"><span class="sub">connecting to telemetry…</span></div>;
  }
  const models = tick.models;
  const wcls = stale ? "widget stale" : "widget";
  return (
    <div class={wcls}>
      <h2>
        models{" "}
        <Info label="about the model hub">
          Every registered profile, with its live runtime state. Click a
          row for placement, plans, measurement history, logs, and the
          typed configure form. Measurements outrank estimates everywhere:
          numbers carry a measured or estimated tag.
        </Info>
      </h2>
      {models.length === 0
        ? <p class="sub">no profiles yet — <a href="/v2/add">add a model</a></p>
        : (
          <table>
            <thead>
              <tr>
                <th>model</th><th>placement</th><th>state</th>
                <th>MoE cache</th><th class="num">tok/s</th><th></th>
                <th>measure{" "}
                  <Info label="about the measurement triggers">
                    A benchmark runs the speed harness and records the
                    result; a smoke test proves the model answers at all;
                    autotune launches candidate plans and keeps the one
                    that measures best. All three go on the benchmark
                    lane, which runs one job at a time so two
                    measurements never share the machine.
                  </Info>
                </th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => <HubRow key={m.name} m={m} />)}
            </tbody>
          </table>
        )}
      <p class="sub" style="margin:.5rem 0 0">
        rows update in place over SSE · click a row for the model hub
        (overview · plans · measurements · logs · configure)
      </p>
    </div>
  );
}
