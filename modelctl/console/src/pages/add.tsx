/* /v2/add: active wizards + starting a new one. Search and starting the
   acquisition are one workflow (the old /pull page folded in here). */
import { useEffect, useState } from "preact/hooks";
import {
  createWizard, fetchWizards, hfSearch, wizardDelete, wizardSource,
} from "../lib/api";
import { Info } from "../lib/info";
import { toast } from "../lib/toasts";
import type { SearchResult, WizardSummary } from "../lib/types";

export function Add() {
  const [wizards, setWizards] = useState<WizardSummary[] | null>(null);
  const [q, setQ] = useState("");
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [searchErr, setSearchErr] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = () => fetchWizards().then(setWizards).catch(() => setWizards([]));
  useEffect(() => { refresh(); }, []);

  const start = async () => {
    setBusy(true);
    try {
      const w = await createWizard();
      location.href = `/v2/add/${w.wizard_id}`;
    } catch (e) {
      toast("err", "✗ couldn't start a wizard", String(e), 9000);
      setBusy(false);
    }
  };

  const search = async () => {
    if (!q.trim()) return;
    setResults(null);
    setSearchErr("");
    try {
      const r = await hfSearch(q.trim());
      setResults(r.results);
      setSearchErr(r.error);
    } catch (e) {
      setSearchErr(String(e));
      setResults([]);
    }
  };

  return (
    <>
      <div class="widget">
        <h2>add a model{" "}
          <Info label="about adding models">
            One workflow owns acquisition: source → inspect → download →
            analyze → plans → test → register. Each step gates the next —
            the wizard never advances over a running or failed step, and a
            failed step retries by re-running its own job with the same
            parameters.
          </Info>
        </h2>
        <div class="frow" style="align-items:end">
          <div class="field" style="flex:2 1 300px">
            <label>search Hugging Face</label>
            <input type="text" placeholder="e.g. qwen gguf" value={q}
                   onInput={(e) => setQ((e.target as HTMLInputElement).value)}
                   onKeyDown={(e) => { if (e.key === "Enter") search(); }} />
          </div>
          <div class="actions">
            <button type="button" onClick={search}>search</button>
            <button type="button" class="btn-primary" disabled={busy}
                    onClick={start}>start blank wizard</button>
          </div>
        </div>
        {searchErr && <p class="sub" style="color:var(--err)">{searchErr}</p>}
        {results && results.length === 0 && !searchErr &&
          <p class="sub">no results</p>}
        {results && results.length > 0 && (
          <table>
            <tbody>
              {results.map((r) => (
                <tr key={r.repo_id}>
                  <td>{r.repo_id}
                    <span class="sub">
                      {r.downloads != null ? ` · ${r.downloads} downloads` : ""}
                    </span>
                  </td>
                  <td class="actions" style="width:120px">
                    <button type="button" disabled={busy} onClick={async () => {
                      setBusy(true);
                      try {
                        const w = await createWizard();
                        await wizardSource(w.wizard_id, {
                          source_type: "hf_repo", repo_id: r.repo_id });
                        location.href = `/v2/add/${w.wizard_id}`;
                      } catch (e) {
                        toast("err", "✗ couldn't start wizard", String(e), 9000);
                        setBusy(false);
                      }
                    }}>add →</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div class="widget">
        <h2>wizards in progress</h2>
        {!wizards && <p class="sub">loading…</p>}
        {wizards && wizards.length === 0 && (
          <p class="sub">none active — start one above</p>
        )}
        {wizards && wizards.length > 0 && (
          <table>
            <tbody>
              {wizards.map((w) => (
                <tr key={w.wizard_id} class="rowlink"
                    onClick={(e) => {
                      if ((e.target as HTMLElement).closest("button")) return;
                      location.href = `/v2/add/${w.wizard_id}`;
                    }}>
                  <td>
                    {w.repo_id || w.local_path || "(no source yet)"}
                    <div class="sub">step: {w.step}
                      {w.profile_name ? ` · profile ${w.profile_name}` : ""}</div>
                  </td>
                  <td class="actions" style="width:200px">
                    <button type="button"
                            onClick={() => { location.href = `/v2/add/${w.wizard_id}`; }}>
                      resume</button>
                    <button type="button" class="btn-danger" onClick={async () => {
                      try {
                        await wizardDelete(w.wizard_id);
                        toast("ok", "wizard abandoned", "its jobs keep their history");
                        refresh();
                      } catch (e) {
                        toast("err", "✗ couldn't abandon wizard", String(e), 9000);
                      }
                    }}>abandon</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
