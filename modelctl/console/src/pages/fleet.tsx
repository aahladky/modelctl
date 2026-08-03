/* fleet: every node the planner can place on, local rig included.

   The layout is deliberately the same widget/meter/chip vocabulary
   operate and settings use — a node card is a card, a budget is a meter,
   a presence state is a chip — because "a node is a node" has to be true
   of the rendering too, not only of the data model. The rig is the first
   card for the same reason it is the first entry in the read model.

   Two things this page refuses to let the eye get wrong:

   * A PIN_MISMATCH node is up. It answers a handshake in milliseconds
     and reports a healthy protocol version, and it is unusable: the two
     ends of an RPC graph must agree on kernels. It is rendered in the
     error treatment, never the ok one, and says outright that it is not
     a planning target.
   * A budget is not a measurement. Every device meter is drawn against
     its ceiling, with the total alongside, so "16 of 30 GiB present" can
     never be read as "16 of 30 GiB allowed" — on the laptop's cpu unit
     those are different numbers and the cgroup enforces the smaller one.
*/
import { useEffect, useState } from "preact/hooks";
import { ConfirmButton, submitAction } from "../lib/actions";
import { Info } from "../lib/info";
import { Meter } from "../lib/meter";
import { toast } from "../lib/toasts";
import { fetchFleet, fmtAgo, fmtGiB, probeFleet, probeNode, setNodeBudget }
  from "../lib/api";
import type { FleetDeviceRow, FleetNodeRow, FleetView, PresenceState,
  StaleProfileRow } from "../lib/types";

const GIB = 2 ** 30;

/* One place decides how a presence state looks. A fourth state added to
   the server without a case here would render as the neutral chip, which
   is the failure mode this map exists to make impossible to reach by
   accident. */
const PRESENCE: Record<PresenceState,
                       { chip: string; label: string; blurb: string }> = {
  PRESENT: {
    chip: "chip ok", label: "present",
    blurb: "reachable, matching build — usable for placement",
  },
  STALE: {
    chip: "chip warn", label: "stale",
    blurb: "not usable — placement falls back to local",
  },
  PIN_MISMATCH: {
    chip: "chip err", label: "version mismatch",
    blurb: "reachable, but built from a different llama.cpp commit — rebuild the "
         + "node to use it; the two ends would disagree on kernels",
  },
};

/* The chip is state and stays visible; what the state MEANS is
   education and hides until asked ("help is summoned, state is visible;
   education hides until asked, state never does"). The blurb used to be
   a sentence of body text under every card, which put five paragraphs of
   teaching copy in the always-visible layer. */
function Presence({ node }: { node: FleetNodeRow }) {
  const p = PRESENCE[node.presence.state] ?? PRESENCE.STALE;
  return (
    <span style="display:inline-flex;align-items:center;gap:.35em">
      <span class={p.chip}><span class="dot"></span>{p.label}</span>
      <Info label={`what "${p.label}" means`}>{p.blurb}</Info>
    </span>
  );
}

function Pin({ node }: { node: FleetNodeRow }) {
  const { node: theirs, expected, agrees } = node.pin;
  if (agrees) {
    return (
      <div class="sub">
        build {theirs.slice(0, 12)} <span style="color:var(--ok)">✓ matches</span>
      </div>
    );
  }
  return (
    <div class="msg error" style="margin:.3rem 0 0">
      build {theirs ? theirs.slice(0, 12) : "(none recorded)"} ≠ this machine's{" "}
      {expected ? expected.slice(0, 12) : "(unknown)"}
    </div>
  );
}

/* The budget editor. Capped at the ceiling in the input itself and
   refused before the request if the typed number still exceeds it: the
   server refuses it too, and both refusals name the same number, because
   an operator who is told "too big" without being told "the most you can
   have is X" has to go read a JSON file to find X. */
function BudgetField({ node, device, staleNames, onDone }: {
  node: FleetNodeRow;
  device: FleetDeviceRow;
  staleNames: string[];
  onDone: () => void;
}) {
  const maxGib = Math.floor((device.ceiling_bytes / GIB) * 100) / 100;
  const [value, setValue] = useState(String(
    Math.round((device.budget_bytes / GIB) * 100) / 100));
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    setValue(String(Math.round((device.budget_bytes / GIB) * 100) / 100));
  }, [device.budget_bytes]);

  const typed = parseFloat(value);
  const bytes = Number.isNaN(typed) ? null : Math.round(typed * GIB);
  const over = bytes != null && bytes > device.ceiling_bytes;
  const changed = bytes != null && bytes !== device.budget_bytes;
  const id = `budget-${node.name}-${device.name}`;

  const submit = () => {
    if (bytes == null) {
      toast("err", "✗ budget not a number", `"${value}" is not a size`, 9000);
      return;
    }
    if (over) {
      /* Client-side refusal, in the server's words and with the server's
         number. The request is not sent — but the endpoint refuses the
         same edit with the same ceiling if it ever is. */
      toast("err", `✗ ${node.name}/${device.name} budget refused`,
            `${(bytes / GIB).toFixed(2)} GiB exceeds the ceiling of `
            + `${fmtGiB(device.ceiling_bytes, 2)} GiB `
            + `(${device.ceiling_basis})`, 9000);
      return;
    }
    setBusy(true);
    submitAction(() => setNodeBudget(node.name, device.name, bytes),
                 `${node.name}/${device.name} budget`)
      .then((ok) => { if (ok) onDone(); })
      .finally(() => setBusy(false));
  };

  return (
    <div class="frow" style="align-items:flex-end;gap:.6rem">
      <div class={over ? "field error" : "field"} style="max-width:180px">
        <div class="lbl">
          <label for={id}>budget <span class="sub">GiB</span></label>
          <Info label="about this budget">
            The most auto-place may use here — an operator
            declaration, not a measurement of the device. The ceiling is
            {" "}{device.ceiling_basis}, minus runtime headroom for the
            RPC server's own memory. A budget above the ceiling is
            admitted by the planner and then killed by the node.
          </Info>
        </div>
        <input id={id} type="number" min={0} max={maxGib} step={0.25}
               value={value} disabled={busy}
               onInput={(e) => setValue((e.target as HTMLInputElement).value)} />
        <div class={over ? "msg error" : "sub"}>
          {over
            ? `over the ${fmtGiB(device.ceiling_bytes, 2)} GiB ceiling`
            : `ceiling ${fmtGiB(device.ceiling_bytes, 2)} GiB`}
        </div>
      </div>
      <ConfirmButton
        label="set budget"
        confirmLabel="yes, set it"
        danger={false}
        disabled={!changed || over}
        busy={busy}
        consequences={<>
          {node.name}/{device.name}: {fmtGiB(device.budget_bytes, 2)} →{" "}
          {bytes != null ? (bytes / GIB).toFixed(2) : "—"} GiB.{" "}
          {staleNames.length > 0
            ? <>Stored machine snapshot for {staleNames.join(", ")} were
                recorded against the old number and go stale; their plans
                keep placing on it until replanned.</>
            : <>Plans compiled from here on use the new number.</>}
        </>}
        onConfirm={submit} />
    </div>
  );
}

function Device({ node, device, staleNames, onDone }: {
  node: FleetNodeRow;
  device: FleetDeviceRow;
  staleNames: string[];
  onDone: () => void;
}) {
  /* The editor is heavy for something touched monthly: collapsed until
     asked for, like the summoned help. */
  const [editing, setEditing] = useState(false);
  return (
    <div style="margin:.8rem 0 0">
      <div class="label">
        <span>
          {device.name} <span class="sub">{device.kind}</span>
          {device.label ? <span class="sub"> · {device.label}</span> : null}
        </span>
        <span class="sub num">{fmtGiB(device.budget_bytes, 2)} GiB budget</span>
      </div>
      <Meter value={device.budget_bytes} max={device.ceiling_bytes}
             label={`${node.name} ${device.name} budget`}
             valuetext={`${fmtGiB(device.budget_bytes, 2)} GiB budgeted of a `
                        + `${fmtGiB(device.ceiling_bytes, 2)} GiB ceiling `
                        + `(${device.ceiling_basis})`} />
      {/* The numbers that differ, never collapsed: what we may spend,
          what the ceiling allows, what the device physically has. When
          the ceiling IS the device total, printing it twice invites the
          reader to look for a distinction that is not there -- so one
          number. Both are kept exactly when they disagree, which is the
          laptop CPU case where the cgroup, not the RAM, is the limit. */}
      <div class="sub">
        {device.ceiling_bytes === device.total_bytes
          ? <>ceiling {fmtGiB(device.ceiling_bytes, 2)} GiB
              {" "}({device.ceiling_basis})</>
          : <>ceiling {fmtGiB(device.ceiling_bytes, 2)} GiB
              {" "}({device.ceiling_basis}){" · "}
              device total {fmtGiB(device.total_bytes, 2)} GiB</>}
        {device.cap_bytes > 0
          ? ` · unit cap ${fmtGiB(device.cap_bytes, 2)} GiB` : ""}
      </div>
      {device.editable
        ? (editing
            ? <BudgetField node={node} device={device} staleNames={staleNames}
                           onDone={() => { setEditing(false); onDone(); }} />
            : <div class="actions" style="margin-top:.35rem">
                <button type="button" onClick={() => setEditing(true)}>
                  edit budget</button>
              </div>)
        : <div class="sub" style="margin-top:.35rem">
            read-only here{" "}
            <Info label="why this budget is read-only">
              {device.edit_note}. <a href="/v2/settings">Open settings</a> to
              change either number.
            </Info>
          </div>}
    </div>
  );
}

function NodeCard({ node, staleNames, onDone }: {
  node: FleetNodeRow;
  staleNames: string[];
  onDone: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const state = node.presence.state;
  /* Anything but PRESENT gets the stale widget treatment: a node that is
     not a planning target must not be able to look like one at a
     glance. */
  const cls = state === "PRESENT" ? "widget" : "widget stale";
  const probe = () => {
    setBusy(true);
    submitAction(() => probeNode(node.name), `check ${node.name}`,
                 undefined,
                 (r) => {
                   const one = r.probed[0];
                   if (!one) return "no answer recorded";
                   return one.reachable
                     ? `reachable · protocol ${one.protocol}`
                     : `unreachable · ${one.detail}`;
                 })
      .then(() => onDone())
      .finally(() => setBusy(false));
  };
  return (
    <div class={cls}>
      <div class="label">
        <span>
          <strong>{node.name}</strong>{" "}
          <span class="sub">
            {node.location === "local" ? "local" : node.endpoint}
            {node.variant ? ` · ${node.variant}` : ""}
          </span>
        </span>
        <Presence node={node} />
      </div>
      {node.location === "remote" && (
        <>
          <Pin node={node} />
          <div class="sub" style="margin-top:.25rem">
            {node.presence.probed_at
              ? `checked ${fmtAgo(node.presence.probed_at)}`
              : "never checked"}
            {node.presence.protocol ? ` · RPC ${node.presence.protocol}` : ""}
            {node.enabled ? "" : " · disabled in the registry"}
            {node.presence.detail ? ` · ${node.presence.detail}` : ""}
          </div>
        </>
      )}
      {/* node.note is no longer rendered. It was operator prose in the
          always-visible layer, and the two notes in the registry
          hardcoded a MemoryMax that the unit had since outgrown -- the
          card printed "MemoryMax=20G" two rows above the correct 26.00
          GiB ceiling it had just computed. The live poll on the operate
          page reports these truthfully now, so the strings were emptied
          rather than re-rendered somewhere quieter. */}
      {node.devices.length === 0 && (
        <div class="sub" style="margin-top:.5rem">no devices recorded</div>
      )}
      {node.devices.map((d) => (
        <Device key={d.name} node={node} device={d} staleNames={staleNames}
                onDone={onDone} />
      ))}
      {node.location === "remote" && (
        <div class="actions" style="margin-top:.8rem">
          <button type="button" class={busy ? "busy" : undefined}
                  disabled={busy} onClick={probe}>check now</button>
        </div>
      )}
    </div>
  );
}

function StaleBanner({ rows }: { rows: StaleProfileRow[] }) {
  if (rows.length === 0) return null;
  return (
    <div class="widget">
      <div class="label">
        <span>
          stored machine snapshot out of step{" "}
          <Info label="what out of step means here">
            These profiles recorded a fleet budget that is no longer the
            one in force. Their stored plans still place against the old
            number — that is the point of recorded inputs — until they
            are replanned.
          </Info>
        </span>
      </div>
      <table>
        <thead><tr><th>profile</th><th>device</th><th class="num">recorded</th>
          <th class="num">now</th></tr></thead>
        <tbody>
          {rows.flatMap((r) =>
            Object.entries(r.changed).map(([key, v]) => (
              <tr key={`${r.name}-${key}`}>
                <td>{r.name}</td>
                <td class="sub">{key}</td>
                <td class="num">{v.recorded == null
                  ? "—" : `${fmtGiB(v.recorded, 2)} GiB`}</td>
                <td class="num">{v.live == null
                  ? "—" : `${fmtGiB(v.live, 2)} GiB`}</td>
              </tr>
            )))}
        </tbody>
      </table>
    </div>
  );
}

function NightLane({ view }: { view: FleetView }) {
  if (view.night_lane.length === 0) return null;
  return (
    <div class="widget">
      <div class="label">
        <span>overnight comparisons that need a node</span>
        <span class="sub" style="display:inline-flex;align-items:center;gap:.35em">
          read-only
          <Info label="why this table is read-only">
            Pre-registered comparisons whose arms require a fleet node.
            Moving a budget under an enabled one changes what it
            measures. Enabling is not offered here: a pre-registration is
            a commitment, and a surface that can flip it in passing is
            one that invalidates it in passing.
          </Info>
        </span>
      </div>
      <table>
        <thead>
          <tr><th>job</th><th>needs</th><th>mode</th><th>state</th></tr>
        </thead>
        <tbody>
          {view.night_lane.map((j) => (
            <tr key={j.id}>
              <td>
                {j.id}
                <div class="sub">{j.title}</div>
              </td>
              <td class="sub">{j.requires_nodes.join(", ")}</td>
              <td class="sub">{j.mode}</td>
              <td>
                <span class={j.enabled ? "chip active" : "chip stopped"}>
                  <span class="dot"></span>{j.enabled ? "enabled" : "not enabled"}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function Fleet() {
  const [view, setView] = useState<FleetView | null>(null);
  const [error, setError] = useState("");
  const [probing, setProbing] = useState(true);

  const load = () =>
    fetchFleet().then(setView).catch((e) => setError(String(e)));

  /* Auto-probe on open, then render what the probe found. The read is
     fired first so the page is never blank while two machines are being
     waited on; the probe's answer replaces it. A refused probe (a
     scratch instance refuses every write) is not fatal — the page still
     renders the last recorded presence, which is what a scratch walk of
     this surface is supposed to show. */
  useEffect(() => {
    let live = true;
    load();
    probeFleet()
      .catch(() => undefined)
      .then(() => { if (live) { setProbing(false); return load(); } });
    return () => { live = false; };
  }, []);

  const reprobeAll = () => {
    setProbing(true);
    submitAction(probeFleet, "check all nodes", undefined,
                 (r) => `${r.probed.length} node(s) checked`)
      .then(() => load())
      .finally(() => setProbing(false));
  };

  if (error) {
    return (
      <div class="widget">
        <p class="msg error">the fleet could not be read: {error}</p>
      </div>
    );
  }
  if (!view) {
    return <div class="widget"><span class="sub">reading the fleet…</span></div>;
  }

  const staleNames = view.stale_profiles.map((p) => p.name);
  const errs = Object.entries(view.errors ?? {});

  return (
    <>
      {errs.map(([section, message]) => (
        <div class="widget" key={section}>
          <p class="msg error">{section}: {message}</p>
        </div>
      ))}

      <StaleBanner rows={view.stale_profiles} />

      <div class="grid" style="grid-template-columns:repeat(auto-fit,minmax(340px,1fr))">
        {view.nodes.map((n) => (
          <NodeCard key={n.name} node={n} staleNames={staleNames}
                    onDone={load} />
        ))}
      </div>

      <div class="widget" style="display:flex;gap:1rem;align-items:center;flex-wrap:wrap">
        <span class="sub">
          this machine's build {view.local_pin
            ? view.local_pin.slice(0, 12) : "(unknown)"}
        </span>
        <span class="sub">
          checks expire after {Math.round(
            (view.nodes[0]?.presence.ttl_seconds ?? 900) / 60)} min
        </span>
        <span class="grow" style="flex:1"></span>
        {probing && <span class="sub">checking…</span>}
        <button type="button" class={probing ? "busy" : undefined}
                disabled={probing} onClick={reprobeAll}>check all nodes</button>
      </div>

      <NightLane view={view} />
    </>
  );
}
