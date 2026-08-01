/* model hub list: every profile, live state from the tick stream, click
   through to the per-model page. Rendering comes straight from the same
   ModelRow rows operate uses -- no second source of truth. */
import { useStream } from "../lib/stream";
import { fmtGiB } from "../lib/api";
import { Info } from "../lib/info";
import type { ModelRow } from "../lib/types";

function HubRow({ m }: { m: ModelRow }) {
  const chipClass = m.state_class ? `chip ${m.state_class}` : "chip";
  return (
    <tr class="rowlink"
        onClick={() => {
          location.href = `/v2/models/${encodeURIComponent(m.name)}`;
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
              </tr>
            </thead>
            <tbody>
              {models.map((m) => <HubRow key={m.name} m={m} />)}
            </tbody>
          </table>
        )}
      <p class="sub" style="margin:.5rem 0 0">
        rows update in place over SSE · click a row for the model page
      </p>
    </div>
  );
}
