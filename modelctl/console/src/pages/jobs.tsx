/* jobs: running / queued / history straight from the real job store.
   Lanes are the store's own (mutation / runtime / download / benchmark).
   Cancel is optimistic: the row reflects "cancelling" instantly, then
   reconciles with the server's answer -- a refusal snaps the row back
   with the flash + toast (the loud un-happen). */
import { useEffect, useRef, useState } from "preact/hooks";
import { useStream } from "../lib/stream";
import { cancelJob, fetchJob, fmtAgo, fmtClock } from "../lib/api";
import { Info } from "../lib/info";
import { Meter } from "../lib/meter";
import { toast } from "../lib/toasts";
import type { JobRow } from "../lib/types";

/* Phase 4 gave jobs their URLs back. The link is on the title so the row
   stays clickable-through without stealing the cancel button's clicks. */
const jobHref = (id: string) => `/v2/jobs/${encodeURIComponent(id)}`;

function JobLink({ job }: { job: JobRow }) {
  return <a href={jobHref(job.id)}>{job.title}</a>;
}

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

/* ---- one job, by URL -------------------------------------------------
   The tick stream carries only the newest rows, so a link to an older
   job cannot be answered from it. This reads the stream when the job is
   in it (live progress, no polling) and falls back to the per-job
   endpoint when it is not -- which is also the only thing that makes an
   old /jobs/{id} bookmark work again. */
export function Job({ id }: { id: string }) {
  const { tick, stale } = useStream();
  const { pending, request } = useCancel();
  const [fetched, setFetched] = useState<JobRow | null>(null);
  const [err, setErr] = useState("");

  const live = tick?.jobs.find((j) => j.id === id) ?? null;
  /* A job off the end of the stream is read from the store, and re-read
     on the tick while it can still move. Once it has finished it is
     immutable, so the reads stop -- otherwise a link to a job that ended
     last week polls the store every two seconds for a row that will
     never change again. */
  const settled = fetched != null
    && !["running", "queued"].includes(fetched.status);
  useEffect(() => {
    if (live || settled) return;   // in the stream, or done and immutable
    fetchJob(id).then(setFetched).catch((e) => setErr(String(e)));
  }, [id, live == null, settled, tick?.ts]);

  const job = live ?? fetched;
  if (err && !job) {
    return (
      <div class="widget">
        <p>no job {id}: {err}</p>
        <p class="sub"><a href="/v2/jobs">back to the job list</a></p>
      </div>
    );
  }
  if (!job) {
    return <div class="widget"><span class="sub">loading job {id}…</span></div>;
  }
  const chip = HISTORY_CHIP[job.status]
    ?? { cls: job.status === "running" ? "chip active" : "chip",
         label: job.status };
  const finished = !["running", "queued"].includes(job.status);
  return (
    <>
      <div class={stale && !finished ? "widget stale" : "widget"}>
        <div class="label">
          <span>{job.title}</span>
          {!finished && !live && (
            <span class="live stale"><span class="dot"></span>
              not in the live stream — read once at{" "}
              {fmtClock(Date.now())}
            </span>
          )}
        </div>
        <table>
          <tbody>
            <tr><td class="sub">status</td>
              <td><span class={chip.cls}><span class="dot"></span>{chip.label}</span></td></tr>
            <tr><td class="sub">lane</td><td>{job.lane}</td></tr>
            <tr><td class="sub">type</td><td>{job.type}</td></tr>
            <tr><td class="sub">id</td><td class="num">{job.id}</td></tr>
            <tr><td class="sub">created</td>
              <td class="sub">{job.created ? fmtAgo(job.created) : "—"}</td></tr>
            <tr><td class="sub">started</td>
              <td class="sub">{job.started ? fmtAgo(job.started) : "not yet"}</td></tr>
            <tr><td class="sub">finished</td>
              <td class="sub">{job.finished ? fmtAgo(job.finished) : "—"}</td></tr>
          </tbody>
        </table>
        {!finished && (
          <>
            <Meter value={Math.round(job.progress * 100)} max={100}
                   style="max-width:340px" label={`${job.title} progress`}
                   valuetext={`${Math.round(job.progress * 100)}% complete`
                              + (job.detail ? ` · ${job.detail}` : "")} />
            <div class="sub num">
              {`${Math.round(job.progress * 100)}%`}
              {job.detail ? ` · ${job.detail}` : ""}
            </div>
          </>
        )}
        {job.error && <div class="msg error">{job.error}</div>}
        <div class="actions" style="margin-top:.6rem">
          <a href="/v2/jobs" class="sub">← all jobs</a>
          <span class="grow" style="flex:1"></span>
          {!finished && (
            <button class={pending[job.id] ? "btn-danger busy" : "btn-danger"}
                    disabled={!!pending[job.id] || !job.cancellable}
                    onClick={() => request(job, job.status === "queued"
                      ? "dequeue" : "cancel")}>
              {job.status === "queued" ? "dequeue" : "cancel"}
            </button>
          )}
        </div>
        {!job.cancellable && !finished && (
          <div class="sub">this lane's jobs cannot be cancelled once started</div>
        )}
      </div>
      <div class="widget">
        <div class="label"><span>log</span></div>
        {job.result_tail.trim()
          ? <LogTail text={job.result_tail} />
          : <p class="sub">this job has not written any output yet</p>}
      </div>
    </>
  );
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
            <div class="table-scroll">
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
                      <JobLink job={j} /> <span class="sub">· lane {j.lane}</span>
                      <Meter value={Math.round(j.progress * 100)} max={100}
                             style="max-width:340px"
                             label={`${j.title} progress`}
                             valuetext={`${Math.round(j.progress * 100)}% complete`
                                        + (j.detail ? ` · ${j.detail}` : "")} />

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
                    {/* the cell stays a cell -- `.actions` sets
                        display:flex, which on a <td> drops it out of row
                        height equalization and steps its border-bottom
                        mid-row. Flex on an inner div, width on the td. */}
                    <td style="width:150px">
                      <div class="actions">
                      <button class={pending[j.id] ? "btn-danger busy" : "btn-danger"}
                              disabled={!!pending[j.id] || !j.cancellable}
                              aria-label={`cancel ${j.title}`}
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
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          )}
      </div>

      <div class={wcls}>
        <h2>queued</h2>
        {queued.length === 0
          ? <p class="sub">queue is empty</p>
          : (
            <div class="table-scroll">
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
                      <JobLink job={j} />{" "}
                      <span class="sub">
                        · lane {j.lane}{j.detail ? ` · ${j.detail}` : ""}
                      </span>
                    </td>
                    <td style="width:120px">
                      <div class="actions">
                      <button class={pending[j.id] ? "busy" : undefined}
                              disabled={!!pending[j.id]}
                              aria-label={`dequeue ${j.title}`}
                              onClick={() => request(j, "dequeue")}>
                        dequeue
                      </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          )}
      </div>

      <div class={wcls}>
        <h2>history</h2>
        {history.length === 0
          ? <p class="sub">no finished jobs yet</p>
          : (
            <div class="table-scroll">
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
                        <JobLink job={j} />{" "}
                        <span class="sub">
                          · lane {j.lane}{sub ? ` · ${sub}` : ""}
                          {j.finished ? ` · ${fmtAgo(j.finished)}` : ""}
                        </span>
                      </td>
                      {/* history has no actions: an empty cell, not a
                          flex container pretending to be one */}
                      <td style="width:120px"></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            </div>
          )}
      </div>
    </>
  );
}
