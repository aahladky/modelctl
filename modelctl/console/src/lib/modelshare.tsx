/* Where this model lives, as one bar.
 *
 * The per-device rows answer "how full is SYCL0". They do not answer the
 * question the screen is actually for -- where does this model run --
 * because five separate tracks make you add them up yourself. This is
 * that answer in one picture: the model's weights, segmented by the
 * device holding them, ordered fastest to slowest so the shape of the
 * bar IS the shape of the placement.
 *
 * The ramp carries the meaning. Hot rust for VRAM, cooling through
 * remote and RAM, grey for anything left on the disk -- so a layout that
 * has gone wrong looks wrong before you read a number. Spill gets the
 * alarm colour, because bytes with nowhere to go but the SSD are the
 * difference between 10 tok/s and 0.4.
 */
import { fmtGiB } from "./api";

export interface Share {
  name: string;
  bytes: number;
  /* VRAM | over RPC | RAM | SSD via mmap -- decides the ramp position. */
  backing: string;
}

/* Fastest first. The order is the point: the bar reads left to right as
   memory getting further from the compute. */
const RANK: Record<string, number> = {
  "VRAM": 0, "over RPC": 1, "RAM": 2, "SSD via mmap": 3,
};

export function rankOf(backing: string): number {
  return RANK[backing] ?? 2;
}

export function orderShares(shares: Share[]): Share[] {
  return [...shares]
    .filter((s) => s.bytes > 0)
    .sort((a, b) => rankOf(a.backing) - rankOf(b.backing)
                 || b.bytes - a.bytes);
}

export function totalBytes(shares: Share[]): number {
  return shares.reduce((sum, s) => sum + Math.max(0, s.bytes), 0);
}

/* spill is tri-state: bytes when the caller knows (the placement answer
   computes it exactly), null when it does not (home draws from live
   holdings, and streamed bytes are precisely the ones nothing holds).
   Null renders NO badge -- "nothing on disk" on no evidence is the kind
   of cheerful lie this console is being rebuilt to stop telling. */
export function ModelShare({ shares, spill }:
                           { shares: Share[]; spill: number | null }) {
  const ordered = orderShares(shares);
  const total = totalBytes(ordered);
  const streaming = ordered.filter((s) => rankOf(s.backing) === 3);
  /* The disk is not a device, and counting it as one turns "43.2 across
     3 devices" into "61.9 across 4" -- a sentence that reads as one more
     card in the machine. This count is of things that HOLD; the SSD
     share is named by the badge and the warning instead. */
  const held = ordered.filter((s) => rankOf(s.backing) !== 3);

  return (
    <div class="share">
      <div class="share-head">
        <span class="share-total">{fmtGiB(total)} GiB</span>
        {held.length
          ? <span class="sub">
              across {held.length} device{held.length === 1 ? "" : "s"}
            </span>
          /* Every byte on the disk and nothing holding any of it.
             "across 0 devices" states that as an absence; the warning
             below states it as the fact it is. */
          : null}
        {spill == null
          ? null
          : spill > 0
            ? <span class="share-spill">
                {fmtGiB(spill)} GiB streaming from disk
              </span>
            : <span class="share-ok">nothing on disk</span>}
      </div>

      <div class="share-bar">
        {ordered.map((s) => (
          <div key={s.name}
               class={`share-seg share-seg-${rankOf(s.backing)}`}
               style={`flex-grow:${Math.max(s.bytes, 1)}`}
               title={`${s.name} — ${fmtGiB(s.bytes)} GiB ${s.backing}`} />
        ))}
      </div>

      <div class="share-legend">
        {ordered.map((s) => (
          <span key={s.name} class="share-key">
            <span class={`share-swatch share-seg-${rankOf(s.backing)}`} />
            <span class="share-key-name">{s.name}</span>
            <span class="sub">{fmtGiB(s.bytes)}</span>
          </span>
        ))}
      </div>

      {streaming.length
        ? <p class="share-warn">
            {streaming.map((s) => s.name).join(", ")} —{" "}
            {streaming.length === 1 ? "this share is" : "these shares are"}
            {" "}addressed off the SSD, not held in memory.
          </p>
        : null}
    </div>
  );
}
