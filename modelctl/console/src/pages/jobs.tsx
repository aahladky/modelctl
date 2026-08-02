/* jobs: running / queued / history straight from the real job store.
   Lanes are the store's own (mutation / runtime / download / benchmark).
   Cancel is optimistic: the row reflects "cancelling" instantly, then
   reconciles with the server's answer -- a refusal snaps the row back
   with the flash + toast (the loud un-happen). */
import { useEffect, useRef, useState } from "preact/hooks";
import { useStream } from "../lib/stream";
import { cancelJob, fmtAgo } from "../lib/api";
import { Info } from "../lib/info";
import { toast } from "../lib/toasts";
import type { JobRow } from "../lib/types";

const HISTORY_CHIP: Record<string, { cls: string; label: string }> = {
  done: { cls: "chip ok", label: "done" },
  failed: { cls: "chip err", label: "failed" },
  cancelled: { cls: "chip", label: "cancelled" },
  interrupted: { cls: "chip warn", label: "interrupted" },
};

function LogTail({ text }: { text: string }) {
  const ref = useRef<HTMLPreElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [text]);
  if (!text.trim()) return null;
  return <pre class="log" style="max-width:560px" ref={ref}>{text.trimEnd()}</pre>;
}

function useCancel() {
  const [pending, setPending] = useState<Record<string, boolean>>({});
  const [flash, setFlash] = useState<Record<string, number>>({});

  const request = async (job: JobRow, verb: string) => {
    setPending((p) => ({ ...p, [job.id]: true })); // reflect instantly
    let refusal: string | null = null;
    try {
      const res = await cancelJob(job.id);
      if (!res.cancelled) refusal = res.reason || `job is ${res.status}`;
    } catch (e) {
      refusal = String(e);
    }
    if (refusal) {
      // un-happen: back to truth, loudly
      setPending((p) => {
        const { [job.id]: _, ...rest } = p;
        return rest;
      });
      setFlash((f) => ({ ...f, [job.id]: Date.now() }));
      setTimeout(() => setFlash((f) => {
        const { [job.id]: _, ...rest } = f;
        return rest;
      }), 2600);
      toast("err", `✗ ${verb} failed — ${job.title}`,
            `server refused: ${refusal}. The optimistic ${verb} was rolled back.`, 9000);
    } else {
      toast("ok", `${job.title} ${verb === "dequeue" ? "dequeued" : "cancelled"}`,
            verb === "dequeue"
              ? "removed before it started · nothing to roll back"
              : "server confirmed the cancel");
    }
  };
  return { pending, flash, request };
}

export function Jobs() {
  const { tick, stale } = useStream();
  const { pending, flash, request } = useCancel();

  if (!tick) {
    return <div class="widget"><span class="sub">connecting to job stream…</span></div>;
  }

  const jobs = tick.jobs;
  const running = jobs.filter((j) => j.status === "running");
  const queued = jobs.filter((j) => j.status === "queued");
  const history = jobs.filter((j) => !["running", "queued"].includes(j.status));
  const wcls = stale ? "widget stale" : "widget";

  /* keyed by the flash timestamp: a second refusal on a row that is
     still flashing remounts the <tr>, which restarts the CSS animation.
     Without it the loudest signal in "optimistic actions loudly un-happen"
     goes quiet exactly when the user retries. */
  const rowClass = (id: string) => (flash[id] ? "unhappen" : "");
  const rowKey = (id: string) => (flash[id] ? `${id}:${flash[id]}` : id);

  return (
    <>
      <div class={wcls}>
        <h2>
          running{" "}
          <Info label="about cancel">
            Cancel reflects instantly, then reconciles with the server. If
            the server refuses (the job finished first, or is marked not
            cancellable), the row snaps back and a toast says why.
          </Info>
        </h2>
        {running.length === 0
          ? <p class="sub">nothing running</p>
          : (
            <table>
              <tbody>
                {running.map((j) => (
                  <tr key={rowKey(j.id)} class={rowClass(j.id)}>
                    <td style="width:110px">
                      {pending[j.id]
                        ? <span class="chip warn"><span class="dot"></span>cancelling</span>
                        : <span class="chip active"><span class="dot"></span>running</span>}
                    </td>
                    <td>
                      {j.title} <span class="sub">· lane {j.lane}</span>
                      <div class="meterbar" style="max-width:340px">
                        <div class="fill" style={`width:${Math.round(j.progress * 100)}%`}></div>
                      </div>
                      <div class="sub num">
                        {/* percentage AND detail: the two answer different
                            questions ("how far" vs "doing what"), and the
                            || dropped the number exactly when a job was
                            interesting enough to report detail */}
                        {`${Math.round(j.progress * 100)}%`}
                        {j.detail ? ` · ${j.detail}` : ""}
                        {j.started ? ` · started ${fmtAgo(j.started)}` : ""}
                      </div>
                      <LogTail text={j.result_tail} />
                    </td>
                    <td class="actions" style="width:150px">
                      <button class={pending[j.id] ? "btn-danger busy" : "btn-danger"}
                              disabled={!!pending[j.id] || !j.cancellable}
                              onClick={() => request(j, "cancel")}>
                        cancel
                      </button>
                      {/* why the control is dead is state, not teaching
                          copy: inline and always visible, never a title
                          attribute only a mouse can reach */}
                      {!j.cancellable && (
                        <div class="sub">this lane's jobs cannot be cancelled
                          once started</div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
      </div>

      <div class={wcls}>
        <h2>queued</h2>
        {queued.length === 0
          ? <p class="sub">queue is empty</p>
          : (
            <table>
              <tbody>
                {queued.map((j, i) => (
                  <tr key={rowKey(j.id)} class={rowClass(j.id)}>
                    <td style="width:110px">
                      {pending[j.id]
                        ? <span class="chip warn"><span class="dot"></span>dequeuing</span>
                        : <span class="chip"><span class="dot"></span>queued #{i + 1}</span>}
                    </td>
                    <td>
                      {j.title}{" "}
                      <span class="sub">
                        · lane {j.lane}{j.detail ? ` · ${j.detail}` : ""}
                      </span>
                    </td>
                    <td class="actions" style="width:120px">
                      <button class={pending[j.id] ? "busy" : undefined}
                              disabled={!!pending[j.id]}
                              onClick={() => request(j, "dequeue")}>
                        dequeue
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
      </div>

      <div class={wcls}>
        <h2>history</h2>
        {history.length === 0
          ? <p class="sub">no finished jobs yet</p>
          : (
            <table>
              <tbody>
                {history.map((j) => {
                  const chip = HISTORY_CHIP[j.status] ?? { cls: "chip", label: j.status };
                  const sub = j.status === "failed" && j.error
                    ? j.error.split("\n")[0].slice(0, 120)
                    : j.detail || j.type;
                  return (
                    <tr key={j.id}>
                      <td style="width:110px">
                        <span class={chip.cls}><span class="dot"></span>{chip.label}</span>
                      </td>
                      <td>
                        {j.title}{" "}
                        <span class="sub">
                          · lane {j.lane}{sub ? ` · ${sub}` : ""}
                          {j.finished ? ` · ${fmtAgo(j.finished)}` : ""}
                        </span>
                      </td>
                      <td class="actions" style="width:120px"></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
      </div>
    </>
  );
}
