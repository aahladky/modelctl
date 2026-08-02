/* settings: the typed editors for the settings this system actually has.

   Scope is deliberately narrow (phase-3 order): profile defaults and
   hardware policy are writable because settings_service and
   modelctl_hardware already own those writes; access and paths are
   reported read-only with their real source, because changing them means
   changing the systemd unit, not clicking here. No JSON anywhere -- the
   defaults form is generated from the server's own field descriptors, so
   an input's min/max is the bound the write validates against. */
import { useEffect, useState } from "preact/hooks";
import {
  ApiError, calibrateStorage, fetchSettings, fmtGiB, reprobeCapabilities,
  rotateToken, saveDefaults, saveHardware,
} from "../lib/api";
import { Info } from "../lib/info";
import { toast } from "../lib/toasts";
import { effectiveTheme, storedTheme } from "../theme";
import type {
  DefaultField, HardwareSection, PathRow, SettingsOverview,
} from "../lib/types";

const GIB = 2 ** 30;

const LABELS: Record<string, string> = {
  device: "GPU device",
  primary_gpu: "primary GPU",
  split_mode: "split mode",
  tensor_split: "tensor split",
  ctx: "context length",
  ttl: "idle TTL",
  cache_type_k: "KV cache type (K)",
  cache_type_v: "KV cache type (V)",
  flash_attn: "flash attention",
  mtp: "multi-token prediction",
  vram_limit_pct: "VRAM limit",
};

/* Teaching copy: summoned, never inline (spec: "help is summoned, state
   is visible"). Every settable default has an entry -- a field with no
   explanation is a field the operator has to guess at. */
const HELP: Record<string, string> = {
  device:
    "Which SYCL device a new profile targets by default, e.g. SYCL0. " +
    "Per-model placement overrides this; it only decides where a profile " +
    "starts before the planner has looked at it.",
  primary_gpu:
    "The GPU the planner fills first and sorts to the front. \"auto\" lets " +
    "it pick the largest enabled device.",
  split_mode:
    "How a model too big for one GPU is divided. \"layer\" splits whole " +
    "layers across devices (the usual choice); \"row\" splits inside each " +
    "tensor and costs more interconnect traffic; \"none\" keeps everything " +
    "on one device.",
  tensor_split:
    "Proportions for the split, one number per GPU, e.g. \"5,3\". Empty " +
    "means split by device memory.",
  ctx:
    "Default context window for a new profile. KV cache grows linearly " +
    "with it, so this is the largest single VRAM lever a profile inherits.",
  ttl:
    "How long llama-swap keeps a model resident after its last request " +
    "before unloading it. 0 unloads immediately.",
  cache_type_k:
    "Quantization for the K half of the KV cache. q8_0 roughly halves KV " +
    "VRAM against f16 at a quality cost most models tolerate.",
  cache_type_v:
    "Quantization for the V half of the KV cache. Some backends are " +
    "fussier about a quantized V than a quantized K.",
  flash_attn:
    "Faster prompt processing on SYCL. \"auto\" lets the backend decide " +
    "from its own probed capabilities, which is what the capability " +
    "report below reads.",
  mtp:
    "Multi-token prediction, where a model ships a draft head for it. " +
    "Off unless a profile is known to have one.",
  vram_limit_pct:
    "The share of each GPU's memory the planner is allowed to fill. The " +
    "rest is headroom for fragmentation and for whatever else is on the " +
    "card — see the implied budget beside the field.",
};

function Field({ f, value, onInput, note, shadowedBy }: {
  f: DefaultField;
  value: string;
  onInput: (v: string) => void;
  note?: string;
  shadowedBy?: string;
}) {
  const label = LABELS[f.name] ?? f.name;
  return (
    <div class="field">
      <div class="lbl">
        <label for={`d-${f.name}`}>
          {label}{f.unit && <span class="sub"> {f.unit}</span>}
        </label>
        <Info label={`about ${label}`}>{HELP[f.name] ?? f.name}</Info>
      </div>
      {f.kind === "choice" ? (
        <select id={`d-${f.name}`} value={value}
                onChange={(e) => onInput((e.target as HTMLSelectElement).value)}>
          {/* the stored value may predate this vocabulary; keep it
              selectable rather than silently rewriting it on save */}
          {!f.choices.includes(value) && value !== "" && (
            <option value={value}>{value} (stored)</option>
          )}
          {f.choices.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      ) : f.kind === "int" ? (
        <input id={`d-${f.name}`} type="number" value={value}
               min={f.min ?? undefined} max={f.max ?? undefined}
               onInput={(e) => onInput((e.target as HTMLInputElement).value)} />
      ) : (
        <input id={`d-${f.name}`} type="text" value={value}
               onInput={(e) => onInput((e.target as HTMLInputElement).value)} />
      )}
      {f.kind === "int" && f.min != null && (
        <div class="help">accepted range {f.min.toLocaleString()}–
          {(f.max ?? 0).toLocaleString()}</div>
      )}
      {note && <div class="msg" style="color:var(--ok)">✓ {note}</div>}
      {shadowedBy && (
        <div class="msg warning">
          ⚠ {shadowedBy} is set in the service environment and wins over
          the saved file — editing this changes nothing until that variable
          is removed from the unit
        </div>
      )}
    </div>
  );
}

function Defaults({ s, onSaved }: { s: SettingsOverview; onSaved: () => void }) {
  const cur = s.defaults;
  const initial = () => {
    const d: Record<string, string> = {};
    for (const f of cur.fields) d[f.name] = String(cur.values[f.name] ?? "");
    return d;
  };
  const [draft, setDraft] = useState<Record<string, string>>(initial);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  useEffect(() => setDraft(initial()), [JSON.stringify(cur.values)]);

  const dirty = cur.fields.filter(
    (f) => draft[f.name] !== String(cur.values[f.name] ?? ""));

  /* live consequence, inline and always visible: what the VRAM limit
     leaves the planner on the real devices of this machine. */
  const pct = parseInt(draft["vram_limit_pct"] ?? "", 10);
  const limitNote = !Number.isNaN(pct) && s.hardware.devices.length
    ? s.hardware.devices
        .map((g) => `${g.device} ${fmtGiB(g.total_bytes * pct / 100, 1)} GiB`)
        .join(" · ") + " usable to the planner"
    : undefined;

  const save = async () => {
    setBusy(true);
    const updates: Record<string, string | number> = {};
    for (const f of dirty) updates[f.name] = draft[f.name];
    try {
      const res = await saveDefaults(updates);
      setWarnings(res.warnings);
      const n = Object.keys(res.applied as Record<string, unknown>).length;
      if (res.warnings.length && n === 0) {
        toast("err", "✗ nothing saved", res.warnings.join("; "), 9000);
      } else if (res.warnings.length) {
        toast("ok", `saved ${n} default(s), ${res.warnings.length} rejected`,
              "the rejected fields kept their stored value — see the form");
      } else {
        toast("ok", `saved ${n} default(s)`,
              "new profiles inherit these; existing profiles are untouched");
      }
      onSaved();
    } catch (e) {
      if (e instanceof ApiError && e.status === 405) {
        toast("err", "✗ save refused", String(e.body.reason ?? e.message), 9000);
      } else {
        toast("err", "✗ saving defaults failed", String(e), 9000);
      }
    } finally {
      setBusy(false);
    }
  };

  /* A warning names its field; hang it under that field so the rejected
     value and the reason are in the same place. */
  const warnFor = (name: string) =>
    warnings.filter((w) => w.startsWith(name + ":"));
  const unattached = warnings.filter(
    (w) => !cur.fields.some((f) => w.startsWith(f.name + ":")));

  return (
    <div class="widget" style="max-width:900px">
      <h2>profile defaults</h2>
      <p class="sub" style="margin:.2rem 0 .9rem">
        What a newly created profile inherits. Changing them never touches
        an existing profile — edit those on their own page. The same
        validation runs here and in <span class="num">modelctl defaults</span>.
      </p>
      {cur.error && <div class="msg error">✗ {cur.error}</div>}
      <form style="display:grid;gap:1rem" onSubmit={(e) => e.preventDefault()}>
        <fieldset>
          <legend>placement</legend>
          <div class="frow">
            {cur.fields.filter((f) => ["device", "primary_gpu", "split_mode",
                                       "tensor_split"].includes(f.name))
              .map((f) => (
                <div key={f.name} class={warnFor(f.name).length ? "field warning" : ""}>
                  <Field f={f} value={draft[f.name] ?? ""}
                         shadowedBy={cur.shadowed[f.name]}
                         onInput={(v) => setDraft({ ...draft, [f.name]: v })} />
                  {warnFor(f.name).map((w) => (
                    <div class="msg warning" key={w}>⚠ {w}</div>))}
                </div>
              ))}
          </div>
        </fieldset>

        <fieldset>
          <legend>context + KV cache</legend>
          <div class="frow">
            {cur.fields.filter((f) => ["ctx", "ttl", "cache_type_k",
                                       "cache_type_v", "flash_attn", "mtp"]
                                       .includes(f.name))
              .map((f) => (
                <div key={f.name} class={warnFor(f.name).length ? "field warning" : ""}>
                  <Field f={f} value={draft[f.name] ?? ""}
                         shadowedBy={cur.shadowed[f.name]}
                         onInput={(v) => setDraft({ ...draft, [f.name]: v })} />
                  {warnFor(f.name).map((w) => (
                    <div class="msg warning" key={w}>⚠ {w}</div>))}
                </div>
              ))}
          </div>
        </fieldset>

        <fieldset>
          <legend>memory policy</legend>
          <div class="frow">
            {cur.fields.filter((f) => f.name === "vram_limit_pct").map((f) => (
              <div key={f.name} class={warnFor(f.name).length ? "field warning" : ""}>
                <Field f={f} value={draft[f.name] ?? ""} note={limitNote}
                       shadowedBy={cur.shadowed[f.name]}
                       onInput={(v) => setDraft({ ...draft, [f.name]: v })} />
                {warnFor(f.name).map((w) => (
                  <div class="msg warning" key={w}>⚠ {w}</div>))}
              </div>
            ))}
          </div>
        </fieldset>

        {unattached.map((w) => <div class="msg warning" key={w}>⚠ {w}</div>)}

        <div class="actions" style="justify-content:flex-end">
          <span class="sub">
            {dirty.length === 0 ? "no field differs from the stored defaults"
              : `${dirty.length} field(s) changed`}
          </span>
          <button type="button" class="btn-primary"
                  disabled={busy || dirty.length === 0} onClick={save}>
            save defaults
          </button>
        </div>
      </form>
    </div>
  );
}

function Hardware({ hw, onSaved }: { hw: HardwareSection; onSaved: () => void }) {
  type Draft = Record<string, { reserve: string; role: string;
                                enabled: boolean; bw: string }>;
  const initial = (): Draft => {
    const d: Draft = {};
    for (const g of hw.devices) {
      d[g.device] = {
        reserve: (g.reserve_bytes / GIB).toFixed(1),
        role: g.role,
        enabled: g.enabled,
        bw: g.bandwidth_overridden && g.bandwidth_gbs != null
          ? String(g.bandwidth_gbs) : "",
      };
    }
    return d;
  };
  const [draft, setDraft] = useState<Draft>(initial);
  const [ram, setRam] = useState(() => (hw.ram.reserve_bytes / GIB).toFixed(1));
  const [warnings, setWarnings] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    setDraft(initial());
    setRam((hw.ram.reserve_bytes / GIB).toFixed(1));
  }, [hw.fingerprint, hw.devices.length]);

  const set = (dev: string, patch: Partial<Draft[string]>) =>
    setDraft({ ...draft, [dev]: { ...draft[dev], ...patch } });

  const save = async () => {
    setBusy(true);
    const devices: Record<string, Record<string, unknown>> = {};
    for (const g of hw.devices) {
      const d = draft[g.device];
      if (!d) continue;
      const spec: Record<string, unknown> = {};
      const reserve = Math.round(parseFloat(d.reserve) * GIB);
      if (!Number.isNaN(reserve) && reserve !== g.reserve_bytes) {
        spec["reserve_bytes"] = reserve;
      } else if (Number.isNaN(reserve)) {
        spec["reserve_bytes"] = d.reserve; // let the server say why
      }
      if (d.role !== g.role) spec["role"] = d.role;
      if (d.enabled !== g.enabled) spec["enabled"] = d.enabled;
      const wasBw = g.bandwidth_overridden && g.bandwidth_gbs != null
        ? String(g.bandwidth_gbs) : "";
      if (d.bw !== wasBw) spec["memory_bandwidth_gbs_override"] = d.bw;
      if (Object.keys(spec).length) devices[g.device] = spec;
    }
    const body: Parameters<typeof saveHardware>[0] = {};
    if (Object.keys(devices).length) body.devices = devices;
    const ramBytes = Math.round(parseFloat(ram) * GIB);
    if (!Number.isNaN(ramBytes) && ramBytes !== hw.ram.reserve_bytes) {
      body.ram = { reserve_bytes: ramBytes };
    }
    if (!body.devices && !body.ram) {
      toast("ok", "nothing to save", "no hardware setting differs from disk");
      setBusy(false);
      return;
    }
    try {
      const res = await saveHardware(body);
      setWarnings(res.warnings);
      if (res.warnings.length) {
        toast(Number(res.applied) ? "ok" : "err",
              `saved ${res.applied} setting(s), ${res.warnings.length} rejected`,
              res.warnings.join("; "), 9000);
      } else {
        toast("ok", `saved ${res.applied} hardware setting(s)`,
              "the planner uses these on the next plan compile");
      }
      onSaved();
    } catch (e) {
      if (e instanceof ApiError && e.status === 405) {
        toast("err", "✗ save refused", String(e.body.reason ?? e.message), 9000);
      } else {
        toast("err", "✗ saving hardware policy failed", String(e), 9000);
      }
    } finally {
      setBusy(false);
    }
  };

  const calibrate = async () => {
    try {
      const { job_id } = await calibrateStorage();
      toast("ok", "storage calibration queued",
            `job ${job_id} · watch it on the jobs page`);
    } catch (e) {
      if (e instanceof ApiError && e.status === 405) {
        toast("err", "✗ calibration refused",
              String(e.body.reason ?? e.message), 9000);
      } else {
        toast("err", "✗ could not queue calibration", String(e), 9000);
      }
    }
  };

  if (hw.error) {
    return (
      <div class="widget"><h2>hardware policy</h2>
        <div class="msg error">✗ {hw.error}</div>
        <p class="sub">Device policy is unreadable, so it is not editable
          here. Everything else on this page still works.</p>
      </div>
    );
  }

  return (
    <div class="widget" style="max-width:900px">
      <h2>hardware policy</h2>
      <p class="sub" style="margin:.2rem 0 .9rem">
        What the planner is allowed to use. Machine facts are shown beside
        each control; only the controls are yours.
      </p>
      <form style="display:grid;gap:1rem" onSubmit={(e) => e.preventDefault()}>
        {hw.devices.map((g) => {
          const d = draft[g.device];
          if (!d) return null;
          const reserve = parseFloat(d.reserve);
          const usable = Number.isNaN(reserve) ? null
            : Math.max(0, g.total_bytes - reserve * GIB);
          const mine = warnings.filter((w) => w.startsWith(g.device));
          return (
            <fieldset key={g.device}>
              <legend>{g.device}</legend>
              <div class="sub" style="margin-top:-.3rem">
                {g.name} · {fmtGiB(g.total_bytes, 1)} GiB total ·{" "}
                {fmtGiB(g.free_bytes, 1)} GiB free
                {g.pci_address ? ` · ${g.pci_address}` : ""}
                {g.pcie_width ? ` · x${g.pcie_width}` : ""}
              </div>
              <div class="frow">
                <div class="check" style="flex:0 0 auto;align-self:end">
                  <input type="checkbox" id={`hw-en-${g.device}`}
                         checked={d.enabled}
                         onChange={(e) => set(g.device,
                           { enabled: (e.target as HTMLInputElement).checked })} />
                  <div><b><label for={`hw-en-${g.device}`}>enabled</label></b>
                    <div class="help">a disabled device is invisible to the
                      planner</div>
                  </div>
                </div>
                <div class="field">
                  <div class="lbl">
                    <label for={`hw-role-${g.device}`}>role</label>
                    <Info label="about device role">
                      The planner fills the primary GPU first and sorts it to
                      the front. Leaving every device unset lets it choose by
                      size. "primary" is the only role the planner reads.
                    </Info>
                  </div>
                  <select id={`hw-role-${g.device}`} value={d.role}
                          onChange={(e) => set(g.device,
                            { role: (e.target as HTMLSelectElement).value })}>
                    {hw.roles.map((r) => (
                      <option key={r} value={r}>{r || "(unset)"}</option>))}
                  </select>
                </div>
                <div class="field">
                  <div class="lbl">
                    <label for={`hw-res-${g.device}`}>reserve{" "}
                      <span class="sub">GiB</span></label>
                    <Info label="about the device reserve">
                      Memory held back from the planner on this device — for
                      the compositor, another process, or allocator
                      fragmentation. The planner treats total minus reserve
                      as the ceiling it may plan against.
                    </Info>
                  </div>
                  <input id={`hw-res-${g.device}`} type="number" min={0}
                         step={0.5} value={d.reserve}
                         onInput={(e) => set(g.device,
                           { reserve: (e.target as HTMLInputElement).value })} />
                  {usable != null && (
                    <div class="msg" style="color:var(--ok)">
                      ✓ leaves {fmtGiB(usable, 1)} GiB plannable
                    </div>
                  )}
                </div>
                <div class="field">
                  <div class="lbl">
                    <label for={`hw-bw-${g.device}`}>bandwidth override{" "}
                      <span class="sub">GB/s</span></label>
                    <Info label="about the bandwidth override">
                      The planner's memory-bandwidth figure for this device,
                      used to judge whether a tier is bandwidth-bound. Leave
                      empty to use the probed value; set it only when you
                      have measured better.
                    </Info>
                  </div>
                  <input id={`hw-bw-${g.device}`} type="number" min={0}
                         step={10} value={d.bw} placeholder="probed"
                         onInput={(e) => set(g.device,
                           { bw: (e.target as HTMLInputElement).value })} />
                  <div class="help">
                    {g.bandwidth_gbs != null
                      ? <>in effect: {g.bandwidth_gbs} GB/s{" "}
                          <span class={g.bandwidth_overridden
                            ? "tag" : "tag measured"}>
                            {g.bandwidth_overridden ? "override" : "probed"}
                          </span></>
                      : "no bandwidth known for this device"}
                  </div>
                </div>
              </div>
              {mine.map((w) => <div class="msg warning" key={w}>⚠ {w}</div>)}
            </fieldset>
          );
        })}

        <fieldset>
          <legend>system memory</legend>
          <div class="sub" style="margin-top:-.3rem">
            {fmtGiB(hw.ram.total_bytes, 1)} GiB total ·{" "}
            {fmtGiB(hw.ram.available_bytes, 1)} GiB available now
          </div>
          <div class="frow">
            <div class="field" style="max-width:220px">
              <div class="lbl">
                <label for="hw-ram">RAM reserve <span class="sub">GiB</span></label>
                <Info label="about the RAM reserve">
                  Host memory held back from planning — the desktop, the
                  page cache, and whatever else shares this machine. CPU-side
                  expert tiers are planned against total minus this.
                </Info>
              </div>
              <input id="hw-ram" type="number" min={0} step={1} value={ram}
                     onInput={(e) => setRam((e.target as HTMLInputElement).value)} />
              {!Number.isNaN(parseFloat(ram)) && (
                <div class="msg" style="color:var(--ok)">
                  ✓ leaves {fmtGiB(Math.max(0,
                    hw.ram.total_bytes - parseFloat(ram) * GIB), 1)} GiB
                  plannable
                </div>
              )}
            </div>
          </div>
          {warnings.filter((w) => w.startsWith("RAM")).map((w) => (
            <div class="msg warning" key={w}>⚠ {w}</div>))}
        </fieldset>

        <div class="actions" style="justify-content:flex-end">
          <button type="button" class="btn-primary" disabled={busy}
                  onClick={save}>save hardware policy</button>
        </div>
      </form>

      <h3 style="margin-top:1.1rem">storage</h3>
      <p class="sub">
        Where weights are read from. Read speeds are measured, not guessed —
        an unmeasured device makes the planner's miss-tier estimate a guess.
      </p>
      <table>
        <thead>
          <tr><th>path</th><th>kind</th><th>filesystem</th>
            <th class="num">free</th><th class="num">sequential read</th>
            <th>mmap</th></tr>
        </thead>
        <tbody>
          {hw.storage.map((s) => (
            <tr key={s.path}>
              <td class="num">{s.path}</td>
              <td>{s.kind}{s.transport && s.transport !== "unknown"
                ? ` · ${s.transport}` : ""}</td>
              <td>{s.filesystem || "—"}</td>
              <td class="num">{fmtGiB(s.free_bytes, 0)} GiB</td>
              <td class="num">
                {s.measured_sequential_read_bps
                  ? <>{(s.measured_sequential_read_bps / 1e9).toFixed(1)} GB/s{" "}
                      <span class="tag measured">measured</span></>
                  : <span class="tag estimated">unmeasured</span>}
              </td>
              <td>
                <span class={s.allow_mmap ? "chip ok" : "chip"}>
                  <span class="dot"></span>{s.allow_mmap ? "mmap ok" : "no mmap"}
                </span>
              </td>
            </tr>
          ))}
          {hw.storage.length === 0 && (
            <tr><td colSpan={6} class="sub">no storage devices reported</td></tr>
          )}
        </tbody>
      </table>
      <div class="actions" style="margin-top:.6rem">
        <button type="button" onClick={calibrate}>measure read speeds</button>
        <span class="sub">runs on the mutation lane · a few seconds of
          reading, no writes</span>
      </div>
    </div>
  );
}

function Access({ s, onRotated }: { s: SettingsOverview; onRotated: () => void }) {
  const a = s.access;
  const [gate, setGate] = useState<string[] | null>(null);
  const [busy, setBusy] = useState(false);

  const rotate = async (confirm: boolean) => {
    setBusy(true);
    try {
      const res = await rotateToken(confirm);
      setGate(null);
      toast("ok", "token rotated", res.message, 9000);
      onRotated();
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && e.body.gate) {
        setGate((e.body.gate as { changes: string[] }).changes);
      } else if (e instanceof ApiError && e.status === 405) {
        toast("err", "✗ rotation refused",
              String(e.body.reason ?? e.message), 9000);
      } else {
        toast("err", "✗ rotating the token failed", String(e), 9000);
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <div class="widget" style="max-width:900px">
      <h2>access</h2>
      <table>
        <tbody>
          <tr>
            <td class="sub">bind address</td>
            <td class="num">{a.bind}</td>
            <td class="sub">
              from {a.bind_source}{" "}
              <Info label="about the bind address">
                Where the console listens. It is read once at start-up from
                MODELCTL_WEB_BIND in the systemd unit, so it cannot be
                changed from inside the running process — edit the unit and
                restart the service.
              </Info>
            </td>
          </tr>
          <tr>
            <td class="sub">session cookie</td>
            <td class="num">{a.secure_cookie ? "secure" : "not secure-only"}</td>
            <td class="sub">
              from MODELCTL_WEB_SECURE_COOKIE{" "}
              <Info label="about the session cookie">
                The cookie is always HttpOnly and SameSite=strict. The
                secure flag additionally refuses to send it over plain
                HTTP, which only makes sense once the console is behind
                TLS.
              </Info>
            </td>
          </tr>
          <tr>
            <td class="sub">token</td>
            <td class="num">{a.token_path}</td>
            <td class="sub">from {a.token_source}</td>
          </tr>
        </tbody>
      </table>

      {gate && (
        <fieldset style="border-color:var(--warn);margin-top:.9rem">
          <legend style="color:var(--warn)">rotation — confirm required</legend>
          {gate.map((c) => <div class="msg warning" key={c}>⚠ {c}</div>)}
          <div class="actions">
            <button type="button" onClick={() => setGate(null)}>cancel</button>
            <button type="button" class="btn-danger" disabled={busy}
                    onClick={() => rotate(true)}>rotate the token</button>
          </div>
        </fieldset>
      )}

      <div class="actions" style="margin-top:.7rem">
        <button type="button" disabled={busy || !a.token_rotatable || !!gate}
                onClick={() => rotate(false)}>rotate token</button>
        <span class="sub">
          {a.token_rotatable
            ? "this session stays signed in; every other one is signed out"
            : "the token comes from MODELCTL_WEB_TOKEN — rotate it where "
              + "that variable is set"}
        </span>
      </div>
    </div>
  );
}

function Paths({ rows }: { rows: PathRow[] }) {
  return (
    <div class="widget" style="max-width:900px">
      <h2>state paths{" "}
        <Info label="about state paths">
          Where this install keeps its profiles, models and console state.
          They come from the environment the service was started with, so
          they are shown rather than edited — pointing a running console at
          a different store mid-flight is how two stores get half a model
          each.
        </Info>
      </h2>
      <table>
        <thead>
          <tr><th>what</th><th>path</th><th>source</th></tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label}>
              <td class="sub">{r.label}</td>
              <td class="num">{r.value}</td>
              <td class="sub">
                {r.source}{" "}
                {r.overridden && <span class="tag">overridden</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Readiness({ r }: { r: SettingsOverview["readiness"] }) {
  /* The old console's /setup page, folded in at the phase-3 cutover:
     readiness is diagnostics, and the four-domain IA has no setup domain.
     The remediations stay shell/systemd commands because that is where
     these settings actually live. */
  if (r.error) {
    return (
      <div class="widget" style="max-width:900px">
        <h2>readiness</h2>
        <div class="msg warning">⚠ {r.error}</div>
      </div>
    );
  }
  const bad = r.checks.filter((c) => c.severity !== "ok");
  return (
    <div class="widget" style="max-width:900px">
      <h2>readiness{" "}
        <Info label="about readiness">
          What has to be true before the console can do its job: a models
          directory, a state directory, a working backend binary, a
          hardware probe, llama-swap reachable, an oneAPI environment for
          SYCL builds, and how far the console is exposed on the network.
        </Info>
      </h2>
      <div class="actions" style="margin:.3rem 0 .6rem">
        <span class={r.ready ? "chip ok" : "chip err"}>
          <span class="dot"></span>{r.ready ? "ready" : "not ready"}
        </span>
        {r.first_run && (
          <span class="chip warn"><span class="dot"></span>first run</span>
        )}
        {bad.length === 0 && <span class="sub">every check passes</span>}
      </div>
      <table>
        <tbody>
          {r.checks.map((c) => (
            <tr key={c.id}>
              <td style="width:150px">
                <span class={c.severity === "ok" ? "chip ok"
                  : c.severity === "warning" ? "chip warn" : "chip err"}>
                  <span class="dot"></span>{c.severity}
                </span>
              </td>
              <td>
                <b>{c.title}</b>
                <div class="sub">{c.detail}</div>
                {c.severity !== "ok" && c.fix && (
                  <div class="msg warning">⚠ {c.fix}</div>
                )}
                {c.severity !== "ok" && c.fix_command && (
                  <pre class="log" style="max-height:none">{c.fix_command}</pre>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Diagnostics({ s }: { s: SettingsOverview }) {
  const d = s.diagnostics;
  const m = d.manifest;
  const caps = d.capabilities;
  const env = d.environment;
  const probe = (caps.probe ?? {}) as Record<string, unknown>;
  const flags = Object.entries(probe).filter(
    ([, v]) => typeof v === "boolean") as [string, boolean][];

  return (
    <div class="widget" style="max-width:900px">
      <h2>integration + diagnostics</h2>

      <h3 style="margin-top:.7rem">integration manifest{" "}
        <Info label="about the integration manifest">
          The manifest records which modelctl commit and which llama.cpp
          commit were last validated together on this hardware. A mismatch
          means the running system is not the pair that was tested — normal
          mid-development, worth knowing when something breaks.
        </Info>
      </h3>
      {!m.present ? (
        <div class="msg warning">⚠ no manifest{m.error ? ` — ${m.error}` : ""}</div>
      ) : (
        <>
          <div class="actions" style="margin:.3rem 0 .5rem">
            <span class={m.ok ? "chip ok" : "chip warn"}>
              <span class="dot"></span>
              {m.ok ? "matches the validated pair" : "differs from validated"}
            </span>
            {m.working_tree_dirty && (
              <span class="chip warn"><span class="dot"></span>
                working tree dirty</span>
            )}
            {m.newer_than_validated && (
              <span class="chip"><span class="dot"></span>
                newer than validated</span>
            )}
          </div>
          <table>
            <thead>
              <tr><th></th><th>running</th><th>validated</th></tr>
            </thead>
            <tbody>
              <tr>
                <td class="sub">modelctl</td>
                <td class="num">{m.modelctl_commit || "—"}</td>
                <td class="num">{m.validated_modelctl_commit || "—"}</td>
              </tr>
              <tr>
                <td class="sub">llama.cpp</td>
                <td class="num">{m.submodule_checked_out || "—"}</td>
                <td class="num">{m.validated_llama_commit || "—"}</td>
              </tr>
              <tr>
                <td class="sub">submodule pin</td>
                <td class="num">{m.submodule_pinned || "—"}</td>
                <td class="sub">upstream base {m.upstream_base || "—"}</td>
              </tr>
            </tbody>
          </table>
          {(m.mismatches ?? []).map((x) => (
            <div class="msg warning" key={x}>⚠ {x}</div>))}
          {(m.notes ?? []).map((x) => (
            <div class="sub" key={x}>· {x}</div>))}
        </>
      )}

      <h3 style="margin-top:1rem">runtime build</h3>
      {caps.error && <div class="msg warning">⚠ {caps.error}</div>}
      <p class="sub">
        Probed by running the binary and reading what it says about itself,
        then cached. Reprobe after replacing a build in place.
      </p>
      <table>
        <tbody>
          <tr><td class="sub">binary</td>
            <td class="num">{caps.binary || "none found"}</td></tr>
          {caps.capability_fingerprint && (
            <tr><td class="sub">capability fingerprint</td>
              <td class="num">{caps.capability_fingerprint}</td></tr>
          )}
          {(caps.candidates ?? []).length > 1 && (
            <tr><td class="sub">other candidates</td>
              <td class="num">{(caps.candidates ?? []).slice(1).join("  ")}</td></tr>
          )}
        </tbody>
      </table>
      {flags.length > 0 && (
        <div class="actions" style="margin-top:.5rem">
          {flags.map(([k, v]) => (
            <span key={k} class={v ? "chip ok" : "chip"}>
              <span class="dot"></span>{k.replace(/_/g, " ")}: {v ? "yes" : "no"}
            </span>
          ))}
        </div>
      )}
      <div class="actions" style="margin-top:.6rem">
        <button type="button" onClick={() => reprobeCapabilities()
          .then((r) => toast("ok", "capability cache cleared", r.message))
          .catch((e) => toast(
            "err", "✗ reprobe refused",
            e instanceof ApiError ? String(e.body.reason ?? e.message) : String(e),
            9000))}>
          reprobe the backend
        </button>
        <span class="sub">drops every cached verdict; the next read runs
          the binary again</span>
      </div>

      <h3 style="margin-top:1rem">environment</h3>
      <table>
        <tbody>
          <tr><td class="sub">platform</td>
            <td class="num">
              {env.platform?.system} {env.platform?.release} ·{" "}
              {env.platform?.machine} · python {env.platform?.python}
            </td></tr>
          <tr><td class="sub">oneAPI env scripts</td>
            <td class="num">
              {(env.oneapi_env_scripts ?? []).join("  ") || "none found"}
            </td></tr>
          {Object.entries(env.modelctl_env ?? {}).map(([k, v]) => (
            <tr key={k}><td class="sub">{k}</td><td class="num">{v}</td></tr>
          ))}
        </tbody>
      </table>

      {d.errors.map((e) => <div class="msg warning" key={e}>⚠ {e}</div>)}

      <h3 style="margin-top:1rem">support bundle</h3>
      <p class="sub">
        Manifest, capability probe, environment, redacted profiles and
        recent logs in one zip. Tokens and credentials are redacted before
        it is written.
      </p>
      <div class="actions">
        <a class="btn" href="/api/v2/settings/support-bundle">
          download support bundle
        </a>
      </div>
    </div>
  );
}

function Appearance() {
  const stored = storedTheme();
  return (
    <div class="widget" style="max-width:900px">
      <h2>appearance</h2>
      <p class="sub">
        Currently <b>{effectiveTheme()}</b>
        {stored ? " — an explicit choice, stored in this browser"
                : " — following this system's preference"}.
        The toggle lives in the header; it is a per-browser preference, not
        a server setting, so it is not on the machine's disk anywhere.
      </p>
    </div>
  );
}

export function Settings() {
  const [s, setS] = useState<SettingsOverview | null>(null);
  const [err, setErr] = useState("");
  const [n, setN] = useState(0);

  useEffect(() => {
    let live = true;
    fetchSettings()
      .then((d) => { if (live) { setS(d); setErr(""); } })
      .catch((e) => { if (live) setErr(String(e)); });
    return () => { live = false; };
  }, [n]);

  const reload = () => setN((x) => x + 1);

  if (err) {
    return (
      <div class="widget">
        <p class="msg error">✗ couldn't load settings: {err}</p>
        <p class="sub">Every setting on this page is read from disk on
          demand; nothing was changed.</p>
      </div>
    );
  }
  if (!s) return <div class="widget"><span class="sub">loading settings…</span></div>;

  return (
    <>
      <Readiness r={s.readiness} />
      <Defaults s={s} onSaved={reload} />
      <Hardware hw={s.hardware} onSaved={reload} />
      <Access s={s} onRotated={reload} />
      <Paths rows={s.paths} />
      <Diagnostics s={s} />
      <Appearance />
    </>
  );
}
