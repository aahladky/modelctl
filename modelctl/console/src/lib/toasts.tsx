/* Toasts, matching the mockup's .toasts/.toast markup. The err variant is
   the loud half of "optimistic actions loudly un-happen on failure". */
import { useEffect, useState } from "preact/hooks";

export interface Toast {
  id: number;
  kind: "ok" | "err";
  title: string;
  sub: string;
}

type Listener = (ts: Toast[]) => void;

let toasts: Toast[] = [];
let seq = 0;
const listeners = new Set<Listener>();

function emit() {
  listeners.forEach((fn) => fn(toasts));
}

export function toast(kind: "ok" | "err", title: string, sub: string, ms = 6000) {
  const t: Toast = { id: ++seq, kind, title, sub };
  toasts = [...toasts, t];
  emit();
  setTimeout(() => {
    toasts = toasts.filter((x) => x.id !== t.id);
    emit();
  }, ms);
}

export function ToastHost() {
  const [list, setList] = useState<Toast[]>(toasts);
  useEffect(() => {
    listeners.add(setList);
    return () => {
      listeners.delete(setList);
    };
  }, []);
  return (
    <div class="toasts">
      {list.map((t) => (
        <div key={t.id} class={t.kind === "err" ? "toast err" : "toast"} role="status">
          <strong>{t.title}</strong>
          <div class="sub">{t.sub}</div>
        </div>
      ))}
    </div>
  );
}
