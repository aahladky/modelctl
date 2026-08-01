/* One SSE connection feeds the whole app (telemetry + jobs + badge).
   Reconnect is owned here, not by EventSource's opaque retry: on drop we
   close the source, count down visibly (the operate page renders it), and
   back off 3s -> 5s -> 10s. Everything rendered from this stream degrades
   to last-known + timestamp while `stale` is true. */
import { useEffect, useState } from "preact/hooks";
import type { Tick } from "./types";

export interface StreamState {
  tick: Tick | null;
  stale: boolean;
  lastAt: number | null; // ms epoch of last received tick
  retryIn: number | null; // seconds until next reconnect attempt, null = connecting
}

type Listener = (s: StreamState) => void;

const BACKOFF = [3, 5, 10];

class TickStream {
  state: StreamState = { tick: null, stale: false, lastAt: null, retryIn: null };
  private listeners = new Set<Listener>();
  private es: EventSource | null = null;
  private attempt = 0;
  private timer: ReturnType<typeof setInterval> | null = null;
  private started = false;

  start() {
    if (this.started) return;
    this.started = true;
    this.connect();
  }

  subscribe(fn: Listener): () => void {
    this.listeners.add(fn);
    fn(this.state);
    return () => this.listeners.delete(fn);
  }

  private emit(patch: Partial<StreamState>) {
    this.state = { ...this.state, ...patch };
    this.listeners.forEach((fn) => fn(this.state));
  }

  private connect() {
    this.clearTimer();
    this.emit({ retryIn: null });
    const es = new EventSource("/api/v2/events");
    this.es = es;
    es.addEventListener("tick", (ev) => {
      this.attempt = 0;
      let tick: Tick;
      try {
        tick = JSON.parse((ev as MessageEvent).data);
      } catch {
        return; // a torn frame is not a reason to blank the page
      }
      this.emit({ tick, stale: false, lastAt: Date.now(), retryIn: null });
    });
    es.onerror = () => {
      es.close();
      if (this.es === es) this.es = null;
      this.scheduleReconnect();
    };
  }

  private scheduleReconnect() {
    const delay = BACKOFF[Math.min(this.attempt, BACKOFF.length - 1)];
    this.attempt += 1;
    let left = delay;
    this.emit({ stale: true, retryIn: left });
    this.clearTimer();
    this.timer = setInterval(() => {
      left -= 1;
      if (left <= 0) {
        this.connect();
      } else {
        this.emit({ retryIn: left });
      }
    }, 1000);
  }

  private clearTimer() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }
}

export const stream = new TickStream();

export function useStream(): StreamState {
  const [state, setState] = useState<StreamState>(stream.state);
  useEffect(() => {
    stream.start();
    return stream.subscribe(setState);
  }, []);
  return state;
}
