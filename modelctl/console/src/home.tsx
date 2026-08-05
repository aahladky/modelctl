/* The stack, not a model.
 *
 * Models are launched by API call and mostly without anyone opening this
 * (Aaron, 2026-08-05): llama-swap loads them on demand from configs
 * already written, and the console is not in the request path at all.
 * So the home screen's job is not to run anything. It is to answer, in
 * one look, "is my stack fine" -- and when it is not, to say so here,
 * rather than leaving a failing API call to be the thing that tells you.
 *
 * Abnormal breaks through first and unconditionally (redesign guide rule
 * 8). An empty trouble band is itself the answer: nothing is wrong.
 *
 * The DETAIL of a failure is deliberately not here. Aaron's call: the
 * record lives on the logs/jobs page, so forensics stay in one place
 * instead of bleeding across every surface. This says what broke and
 * sends you there; it does not try to be there.
 */
import { useEffect, useState } from "preact/hooks";

import {
  fetchFleet, fetchJobs, fetchModels, fmtGiB, stateLabel,
} from "./lib/api";
import { troubles } from "./lib/troubles";
import type { TroubleItem } from "./lib/troubles";
import type { FleetView, JobRow, ModelRow } from "./lib/types";

function Trouble({ items }: { items: TroubleItem[] }) {
  if (!items.length) {
    return (
      <div class="calm">
        <span class="calm-mark">✓</span>
        <span>Nothing is wrong. Every registered model is servable and every
              enabled node is reachable.</span>
      </div>
    );
  }
  return (
    <div class="widget trouble">
      <div class="label">
        <span>{items.length} thing{items.length === 1 ? "" : "s"} to look at</span>
        <a href="/v2/jobs">the record is on jobs &amp; logs →</a>
      </div>
      {items.map((t, i) => (
        <p key={i} class="trouble-row">
          <span class="trouble-what">{t.what}</span>
          <span class="trouble-detail">{t.detail}</span>
        </p>
      ))}
    </div>
  );
}

function Serving({ models }: { models: ModelRow[] }) {
  const live = models.filter((m) => m.running);
  return (
    <div class="widget">
      <div class="label"><span>serving now</span></div>
      {live.length
        ? live.map((m) => (
            <p key={m.name} class="srv">
              <span class="srv-name">{m.name}</span>
              <span class="srv-state">{stateLabel(m.state)}</span>
              <span class="sub">
                {m.tok_s ? `${m.tok_s.toFixed(1)} tok/s` : ""}
                {m.size_bytes ? ` · ${fmtGiB(m.size_bytes)} GiB` : ""}
              </span>
            </p>
          ))
        : <p class="sub">Nothing is loaded. The next API call will load
                         whatever it asks for.</p>}
    </div>
  );
}

function Ready({ models }: { models: ModelRow[] }) {
  const idle = models.filter((m) => m.registered && m.enabled && !m.running);
  return (
    <div class="widget">
      <div class="label">
        <span>{idle.length} more the API can ask for</span>
      </div>
      <p class="ready">
        {idle.map((m) => <span key={m.name} class="ready-chip">{m.name}</span>)}
      </p>
    </div>
  );
}

export function Home() {
  const [models, setModels] = useState<ModelRow[]>([]);
  const [jobs, setJobs] = useState<JobRow[]>([]);
  const [fleet, setFleet] = useState<FleetView | null>(null);
  const [failed, setFailed] = useState<string>("");

  useEffect(() => {
    let live = true;
    /* Each source is awaited on its own: one that is down must not blank
       the two that are up. */
    fetchModels().then((m) => { if (live) setModels(m); })
                 .catch((e) => { if (live) setFailed(String(e)); });
    fetchJobs().then((j) => { if (live) setJobs(j); }).catch(() => {});
    fetchFleet().then((f) => { if (live) setFleet(f); }).catch(() => {});
    return () => { live = false; };
  }, []);

  if (failed) {
    return (
      <div class="widget trouble">
        <div class="label">
          <span>the console cannot read its own state</span>
        </div>
        <p class="trouble-detail">{failed}</p>
      </div>
    );
  }
  return (
    <>
      <Trouble items={troubles(models, jobs, fleet, Date.now() / 1000)} />
      <Serving models={models} />
      <Ready models={models} />
    </>
  );
}
