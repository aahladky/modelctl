/* Summoned help (spec: "help is summoned, state is visible"): teaching
   copy hides behind this small i affordance. Opens on click/tap or
   keyboard activation, closes on Esc or click-out. Never hover-only. */
import { useEffect, useId, useRef, useState } from "preact/hooks";
import type { ComponentChildren } from "preact";

export function Info({ children, label = "what is this?" }:
                     { children: ComponentChildren; label?: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);
  const id = useId();

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onClick);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onClick);
    };
  }, [open]);

  return (
    <span class="info" ref={ref}>
      <button type="button" aria-label={label} aria-expanded={open}
              aria-controls={id} onClick={() => setOpen(!open)}>
        i
      </button>
      {open && <span class="popover" role="note" id={id}>{children}</span>}
    </span>
  );
}
