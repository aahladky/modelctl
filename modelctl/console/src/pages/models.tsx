/* models: the catalog. Every profile with live state, searchable,
   filterable, sortable; click through to the per-model page for plans,
   measurements, logs, and configuration. The measurement triggers moved
   to the model page's overview tab — rows are clean links now, and the
   click-defense code for buttons-inside-a-link-row went with them. */
import { useState } from "preact/hooks";
import { useStream } from "../lib/stream";
import { fmtGiB } from "../lib/api";
import { EmptyState, PlacementChips } from "../lib/ui";
import { Info } from "../lib/info";
import type { ModelRow } from "../lib/types";

type SortKey = "running" | "name" | "size" | "tok_s";

function HubRow({ m }: { m: ModelRow }) {
  const chipClass = m.state_class ? `chip ${m.state_class}` : "chip";
  const href = `/v2/models/${encodeURIComponent(m.name)}`;
  return (
    <tr class="rowlink" onClick={(e) => {
      if ((e.target as HTMLElement).closest("a")) return;
      location.href = href;
    }}>
      <td>
        <a href={href} onClick={(e) => e.stopPropagation()}>{m.name}</a>
        <div class="sub">
          {[m.file, m.backend !== "llama-cpp" ? m.backend : ""]
            .filter(Boolean).join(" · ")}
        </div>
      </td>
      <td class="opt">
        <PlacementChips summary={m.placement_summary} fallback={m.placement} />
      </td>
      <td>
        <span class={chipClass}><span class="dot"></span>{m.state}</span>
        {!m.enabled && <span class="tag" style="margin-left:.4em">disabled</span>}
      </td>
      <td class="opt sub">
        {m.moe_cache_mode !== "off" ? `cache ${m.moe_cache_mode}` : "—"}
      </td>
      <td class={m.size_bytes != null ? "num" : "num sub"}>
        {m.size_bytes != null ? `${fmtGiB(m.size_bytes)} GiB` : "—"}
      </td>
      <td class={m.tok_s != null ? "num" : "num sub"}>
        {m.tok_s != null ? m.tok_s.toFixed(1) : "—"}
      </td>
    </tr>
  );
}

export function Models() {
  const { tick, stale } = useStream();
  const [q, setQ] = useState("");
  const [state, setState] = useState("all");
  const [sort, setSort] = useState<SortKey>("running");

  if (!tick) {
    return <div class="widget"><span class="sub">connecting to telemetry…</span></div>;
  }
  const models = tick.models;
  const needle = q.trim().toLowerCase();
  const shown = models
    .filter((m) => {
      if (state === "running") return m.running;
      if (state === "stopped") return !m.running && m.enabled;
      if (state === "disabled") return !m.enabled;
      if (state === "unmanaged") return m.backend === "unmanaged";
      return true;
    })
    .filter((m) => !needle
      || m.name.toLowerCase().includes(needle)
      || m.file.toLowerCase().includes(needle))
    .sort((a, b) => {
      if (sort === "name") return a.name.localeCompare(b.name);
      if (sort === "size") return (b.size_bytes ?? -1) - (a.size_bytes ?? -1);
      if (sort === "tok_s") return (b.tok_s ?? -1) - (a.tok_s ?? -1);
      return Number(b.running) - Number(a.running) || a.name.localeCompare(b.name);
    });
  const wcls = stale ? "widget stale" : "widget";
  return (
    <div class={wcls}>
      <h2>
        models{" "}
        <Info label="about this list">
          Every registered profile, with its live runtime state. Click a
          row for placement, plans, measurement history, logs, and the
          typed configure form. Measurements outrank estimates everywhere:
          numbers carry a measured or estimated tag.
        </Info>
      </h2>
      {models.length === 0
        ? <p class="sub">no profiles yet — <a href="/v2/add">add a model</a></p>
        : (
          <>
            <div class="frow" style="align-items:center;margin:.2rem 0 .5rem">
              <input type="search" placeholder="search name or file…"
                     aria-label="search models" value={q}
                     style="flex:1 1 200px;max-width:320px"
                     onInput={(e) => setQ((e.target as HTMLInputElement).value)} />
              <select aria-label="state filter" style="width:auto" value={state}
                      onChange={(e) => setState((e.target as HTMLSelectElement).value)}>
                <option value="all">all states</option>
                <option value="running">running</option>
                <option value="stopped">stopped</option>
                <option value="disabled">disabled</option>
                <option value="unmanaged">unmanaged</option>
              </select>
              <select aria-label="sort" style="width:auto" value={sort}
                      onChange={(e) => setSort((e.target as HTMLSelectElement).value as SortKey)}>
                <option value="running">running first</option>
                <option value="name">by name</option>
                <option value="size">by size</option>
                <option value="tok_s">by tok/s</option>
              </select>
              <span class="sub" style="margin-left:auto">
                {shown.length === models.length
                  ? `${models.length}`
                  : `${shown.length} of ${models.length}`}
              </span>
            </div>
            {shown.length === 0
              ? <EmptyState>nothing matches</EmptyState>
              : (
                <div class="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        <th>model</th><th class="opt">placement</th><th>state</th>
                        <th class="opt">MoE cache</th><th class="num">size</th>
                        <th class="num">tok/s</th>
                      </tr>
                    </thead>
                    <tbody>
                      {shown.map((m) => <HubRow key={m.name} m={m} />)}
                    </tbody>
                  </table>
                </div>
              )}
          </>
        )}
      <p class="sub" style="margin:.5rem 0 0">
        click a row for plans, measurements, logs, and configuration
      </p>
    </div>
  );
}
