/* The two things every phase-4 control needs: a way to submit a job and
   report the server's own words, and an explicit confirm for the writes
   that cannot be undone.

   Both were duplicated per page in phases 1-3 (operate had its own
   submitAction; the configure form grew the only confirm on the surface).
   Phase 4 adds fourteen controls across four pages, which is where one
   implementation stops being a nicety: a refusal reason that renders on
   operate and not on the model page is exactly the failure the scratch
   walk exists to catch. */
import { useState } from "preact/hooks";
import type { ComponentChildren } from "preact";
import { ApiError } from "./api";
import { toast } from "./toasts";

/* A mutating call, with the server's refusal reason surfaced verbatim.
   405 is scratch-safe mode and 409 is a gate: both answer with a body
   that names the reason, and a bare status would throw that away.
   Returns true when the call went through. */
export async function submitAction<T>(
  fn: () => Promise<T>, label: string,
  onOk?: (result: T) => void, okSub?: (result: T) => string,
): Promise<boolean> {
  try {
    const result = await fn();
    const sub = okSub
      ? okSub(result)
      : (result as { job_id?: string })?.job_id
        ? `job ${(result as { job_id: string }).job_id} · watch it on the jobs page`
        : "the server confirmed it";
    toast("ok", `${label} submitted`, sub);
    onOk?.(result);
    return true;
  } catch (e) {
    if (e instanceof ApiError && e.status === 405) {
      toast("err", `✗ ${label} refused`, String(e.body.reason ?? e.message), 9000);
    } else if (e instanceof ApiError) {
      toast("err", `✗ ${label} failed`, String(e.body.error ?? e.message), 9000);
    } else {
      toast("err", `✗ ${label} failed`, String(e), 9000);
    }
    return false;
  }
}

/* Destructive actions confirm in place rather than through window.confirm:
   the confirm has to be able to say WHAT the action does (the browser
   dialog gets one line and no markup), and the structural-change gate on
   the configure form already established the inline pattern. */
export function ConfirmButton({ label, confirmLabel, danger = true, disabled,
                               busy, consequences, onConfirm }: {
  label: string;
  confirmLabel?: string;
  danger?: boolean;
  disabled?: boolean;
  busy?: boolean;
  consequences: ComponentChildren;
  onConfirm: () => void;
}) {
  const [asking, setAsking] = useState(false);
  const cls = [danger ? "btn-danger" : "", busy ? "busy" : ""]
    .filter(Boolean).join(" ") || undefined;
  if (!asking) {
    return (
      <button type="button" class={cls} disabled={disabled || busy}
              onClick={() => setAsking(true)}>{label}</button>
    );
  }
  return (
    <span class="actions">
      <span class="msg warning">{consequences}</span>
      <button type="button" onClick={() => setAsking(false)}>cancel</button>
      <button type="button" class={cls} disabled={busy}
              onClick={() => {
                setAsking(false);
                onConfirm();
              }}>{confirmLabel ?? `yes, ${label}`}</button>
    </span>
  );
}
