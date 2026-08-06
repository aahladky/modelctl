/* A model's bytes, as parts of one bar: the arithmetic only.
 *
 * Lifted out of modelshare.tsx, which is JSX and therefore unreachable
 * by the test runner (node --test over src/lib/*.test.ts, no transform).
 * That gap is not academic -- it is why the header could disagree with
 * its own legend for as long as it did. Same split the rest of lib/
 * already keeps: device.ts is the math of a CONTROL, stack.ts the math
 * of the home PICTURE, this the math of a SHARE bar, and each of them
 * has tests because none of them is a component.
 */
/* Extension named explicitly: the test runner loads this module raw,
   with no bundler to guess one. */
import { fmtGiB } from "./api.ts";

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

/* The total AS SHOWN: the sum of the parts after each is rounded for
   display, not the exact total rounded once.
 *
 * fmtGiB rounds every number independently from exact bytes, so a
 * legend of 27.8 + 9.9 + 9.6 + 17.6 sat under a header reading 65.0 --
 * live, on the placement screen, for a real layout. Both numbers were
 * individually correct and the screen still did not add up, which is
 * the one thing this console cannot afford to be.
 *
 * The alternative -- keep the exact total and adjust a part to absorb
 * the remainder -- was rejected: the per-device numbers are the ones an
 * operator acts on ("does this fit on that card"), and moving one by a
 * tenth to tidy a summary corrupts the actionable number to flatter the
 * decorative one. This way every part is exactly its own rounding and
 * the header is exactly their sum; the cost is that the header can sit
 * up to 0.05 GiB per part off the true total, which no decision on this
 * screen turns on. */
export function shownTotal(shares: Share[], digits = 1): number {
  return shares.reduce(
    (sum, s) => sum + Number(fmtGiB(Math.max(0, s.bytes), digits)), 0);
}
