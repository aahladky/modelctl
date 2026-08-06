/* The placement screen's arithmetic: selection editing, dirtiness, and
 * the row-to-share projection. DOM-free, like device.ts and stack.ts --
 * the screen renders these answers, it does not re-derive them.
 *
 * A selection is sparse on purpose. An absent key means "as the machine
 * offers", which is also what the backend means by it, so the editor
 * must never store an entry that says nothing: {} pruned to absence
 * keeps "automatic" reachable by clearing every choice, and keeps the
 * same intent equal to itself under comparison. An explicit on:true is
 * NOT nothing -- forcing a device is an intent worth recording.
 */
import type {
  DeviceChoice, PlacedDevice, PlacementSelection,
} from "./types";
import type { Share } from "./sharemath";

function pruned(choice: DeviceChoice): DeviceChoice | null {
  const out: DeviceChoice = {};
  if (choice.on != null) out.on = choice.on;
  if (choice.ceiling_bytes != null) out.ceiling_bytes = choice.ceiling_bytes;
  return out.on == null && out.ceiling_bytes == null ? null : out;
}

/* One device's choice changed; everything else stays as it was. patch
   fields set to null/undefined clear rather than store. Never mutates. */
export function updateChoice(selection: PlacementSelection, key: string,
                             patch: DeviceChoice): PlacementSelection {
  const merged = pruned({ ...(selection[key] ?? {}), ...patch });
  const next: PlacementSelection = { ...selection };
  if (merged == null) delete next[key];
  else next[key] = merged;
  return next;
}

/* Same intent, same answer -- regardless of empty entries or undefined
   fields. This is what "unsaved changes" is read off, so a false
   difference would nag and a false equality would swallow an edit. */
export function selectionsEqual(a: PlacementSelection,
                                b: PlacementSelection): boolean {
  const keys = new Set([...Object.keys(a ?? {}), ...Object.keys(b ?? {})]);
  for (const key of keys) {
    const ca = pruned(a?.[key] ?? {});
    const cb = pruned(b?.[key] ?? {});
    if ((ca == null) !== (cb == null)) return false;
    if (ca == null || cb == null) continue;
    if ((ca.on ?? null) !== (cb.on ?? null)) return false;
    if ((ca.ceiling_bytes ?? null) !== (cb.ceiling_bytes ?? null)) return false;
  }
  return true;
}

/* The placement answer's rows, as shares for the model bar. Purely a
   projection: bytes and backing are the planner's own numbers. */
export function sharesOf(devices: Record<string, PlacedDevice>): Share[] {
  return Object.entries(devices ?? {})
    .filter(([, row]) => row.bytes > 0)
    .map(([key, row]) => ({ name: key, bytes: row.bytes,
                            backing: row.backing }));
}

/* Fastest first, matching the share ramp: the rig's own devices, then
   remote compute, then remote CPU memory, then the host's RAM row. */
export function deviceOrder(key: string): number {
  if (key === "RAM") return 3;
  if (key.startsWith("RPC:")) return key.endsWith(":CPU") ? 2 : 1;
  return 0;
}

/* "2026-08-01T17:57:49" -> "1 Aug, 17:57". The recorded_at on a
   placement answer, made glanceable; an unparseable string comes back
   as itself (a wrong-looking date is better than a hidden one), and
   null -- never recorded -- comes back empty for the caller to word. */
export function fmtWhen(iso: string | null): string {
  if (!iso) return "";
  const t = new Date(iso);
  if (Number.isNaN(t.getTime())) return iso;
  const day = t.getDate();
  const month = t.toLocaleString("en", { month: "short" });
  const hh = String(t.getHours()).padStart(2, "0");
  const mm = String(t.getMinutes()).padStart(2, "0");
  return `${day} ${month}, ${hh}:${mm}`;
}
