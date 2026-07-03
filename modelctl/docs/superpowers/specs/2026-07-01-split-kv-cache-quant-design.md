# Split K/V cache quantization — design

Date: 2026-07-01
Status: approved, not yet implemented

## Context

`modelctl.py` currently has one `kv_quant` config field that sets both
`--cache-type-k` and `--cache-type-v` on `llama-server` to the same value.
Asymmetric K/V cache quantization (e.g. `q8_0` for K, `q4_0` for V) is a
real, useful llama.cpp technique for trading VRAM against quality
differently per cache — this spec adds the ability to set them
independently, in the CLI (`prompt_config`/`prompt_defaults`) and in the
TUI's `ConfigureScreen`.

Every saved profile on disk today has a single `"kv_quant"` field (e.g.
`~/.local/share/modelctl/profiles/Qwythos-9B-Q4.json`), and `defaults.json`
may too. Backward compatibility with these existing files, without a mass
rewrite, is a hard requirement of this design.

## Decisions made during brainstorming

- **Full replace with a compatibility read**, not additive override
  fields. Going forward, profiles and persisted defaults use
  `cache_type_k`/`cache_type_v`. Existing files that only have `kv_quant`
  are NOT rewritten automatically — instead, every place that reads cache
  quant config resolves it through a shared fallback: `cfg.get(new_key) or
  cfg.get("kv_quant")`. A profile only gets migrated to the new field
  names when the user explicitly re-saves it (via `modelctl edit <name>`,
  or by pulling a new profile).
- **Field names: `cache_type_k` / `cache_type_v`** — matches
  `llama-server`'s actual flag names (`--cache-type-k`/`--cache-type-v`)
  exactly, so a raw profile JSON or generated `run.sh` is self-documenting
  without needing to know modelctl's internal naming.
- **Two separate prompts/fields everywhere**, not a single field with a
  comma-shorthand. "K cache quant" and "V cache quant" as distinct CLI
  prompts and distinct TUI `Input` widgets. Pressing Enter twice (CLI) or
  leaving both at their pre-filled defaults (TUI) reproduces today's
  "same value for both" behavior with no new syntax to learn.

## Architecture

One new helper function in `modelctl.py`:

```python
def _resolve_cache_types(cfg: dict) -> tuple[str | None, str | None]:
    """Resolve the effective --cache-type-k/--cache-type-v values for a
    profile's config dict, with backward compatibility for profiles saved
    before this feature existed (which only have a single 'kv_quant' field
    applied to both). New profiles set cache_type_k/cache_type_v directly;
    old ones fall back to kv_quant for whichever of the two isn't set.
    Returns (None, None) if neither the new fields nor the legacy field
    are present, matching today's 'omit the flags entirely' behavior."""
    legacy = cfg.get("kv_quant")
    return cfg.get("cache_type_k") or legacy, cfg.get("cache_type_v") or legacy
```

Both `build_server_args()` and `render_router_preset()` (the two places
that currently duplicate the `if cfg.get("kv_quant"): ... both flags ...`
pattern) call this and emit each flag independently if its resolved value
is truthy -- so a hand-edited profile with only one of the two fields set
still produces a sensible (if asymmetric-by-omission) result, rather than
requiring both or neither.

## Components

- **`DEFAULT_KV_QUANT`** (module constant) becomes
  `DEFAULT_CACHE_TYPE_K` / `DEFAULT_CACHE_TYPE_V`, reading
  `MODELCTL_DEFAULT_CACHE_TYPE_K` / `MODELCTL_DEFAULT_CACHE_TYPE_V`, both
  defaulting to `"q8_0"` -- identical effective default to today.

- **`load_defaults()`** resolves each of `cache_type_k`/`cache_type_v`
  through a fallback chain: new env var -> new persisted key -> legacy
  value (persisted `kv_quant` key, or the old `MODELCTL_DEFAULT_KV_QUANT`
  env var, via the existing `pick("kv_quant", None)` mechanism) ->
  hardcoded default. A user who already has `MODELCTL_DEFAULT_KV_QUANT`
  set in their shell, or an existing `defaults.json` with `kv_quant`,
  keeps getting that value applied to both until they explicitly diverge
  them.

- **`prompt_defaults()`** and **`prompt_config()`** (CLI): each gets two
  prompts, "K cache quant" and "V cache quant", replacing the one "KV
  cache quant" prompt. The returned dict has `cache_type_k`/`cache_type_v`
  keys, not `kv_quant`.

- **`build_server_args()`** and **`render_router_preset()`**: both call
  `_resolve_cache_types(cfg)` and emit `--cache-type-k <k>` / `--cache-type-v
  <v>` (CLI) or `cache-type-k = <k>` / `cache-type-v = <v>` (INI)
  independently, only when each resolved value is truthy.

- **`ConfigureScreen`** (`modelctl_tui.py`): the single `Input(id="config-
  kv-quant")` is replaced by two: `Input(id="config-cache-type-k")` and
  `Input(id="config-cache-type-v")`, each pre-filled from
  `d["cache_type_k"]`/`d["cache_type_v"]` (via `load_defaults()`). The
  submitted config dict carries `cache_type_k`/`cache_type_v`.

## Data flow

Same shape as every other config field in this codebase: CLI/TUI prompts
populate a config dict -> stored on the profile -> read by
`build_server_args()`/`render_router_preset()` at config-generation time.
The only new wrinkle is `_resolve_cache_types()`'s fallback, which exists
purely to bridge old-shape profiles into the new code without requiring
anyone to touch old files.

## Error handling

- Neither new fields nor legacy `kv_quant` present (a profile with no
  cache quant config at all -- possible for very old or hand-crafted
  profiles): `_resolve_cache_types()` returns `(None, None)`, and neither
  flag is emitted -- identical to today's behavior when `kv_quant` is
  falsy/absent.
- Only one of `cache_type_k`/`cache_type_v` set (e.g. a hand-edited
  profile): that one flag is emitted, the other is omitted rather than
  guessed at or defaulted -- an explicit, unsurprising outcome for a
  hand-edited file. (Profiles produced through the normal CLI/TUI flow
  always set both, since both prompts/fields are always shown.)

## Testing

- `TestBuildServerArgs` and `TestRenderRouterPreset` each get three new
  cases: (1) a profile with distinct `cache_type_k`/`cache_type_v` values,
  confirming both flags are emitted with their own values; (2) an
  old-style profile with only `kv_quant`, confirming both flags are
  emitted with that same value (the fallback); (3) a profile with only
  `cache_type_k` set (no `kv_quant`, no `cache_type_v`), confirming only
  `--cache-type-k` is emitted.
- A new `TestResolveCacheTypes` class unit-tests `_resolve_cache_types()`
  directly, covering all four combinations (both new fields set, only
  legacy set, only one new field set, neither set) without needing to go
  through the larger `build_server_args()`/`render_router_preset()`
  functions for each case.
- `TestConfigureScreen` in `test_modelctl_tui.py`, plus any other test
  fixture dict currently using `"kv_quant"` (e.g. `fake_defaults` in
  `TestFullWizardFlow`), gets updated to the `cache_type_k`/`cache_type_v`
  shape, and `ConfigureScreen`'s tests gain assertions for both new input
  IDs.

## Out of scope (deliberately deferred)

- Rewriting/migrating existing profile files on disk. `kv_quant`-only
  profiles keep working forever via the fallback; migration only happens
  when a profile is explicitly re-saved through `edit` or a fresh `pull`.
- Any change to `defaults.json`'s on-disk format beyond what
  `prompt_defaults()` naturally writes going forward (old `kv_quant` key
  in an existing `defaults.json`, if present, is simply left unused
  alongside the new keys once the user re-runs `modelctl defaults` --  not
  actively cleaned up).

## Addendum (2026-07-02): estimator integration + standalone calculator

Written after the VRAM-aware router work landed; approved verbally.

### Estimator integration

`modelctl_vram.kv_cache_bytes` currently takes one `kv_quant` applied to
both caches. It gains an optional fourth parameter:

```python
def kv_cache_bytes(params, ctx, cache_type_k, cache_type_v=None):
```

`cache_type_v=None` means "same as K" (backward compatible; existing
callers/tests unchanged). K and V are computed as separate sums: K uses
k_dim (or k_dim_swa on SWA layers) x bytes_per_element(cache_type_k); V
uses v_dim (or v_dim_swa) x bytes_per_element(cache_type_v). The return
gains no new shape -- still total bytes -- but `estimate_from_parts`
grows the same optional `cache_type_v` param and passes it through, and
`modelctl.estimate_vram_footprint` resolves the profile's effective pair
via `_resolve_cache_types(cfg)` so `place`, the load guard, and pull
hints all see split-quant footprints automatically.

### Standalone calculator

`modelctl_vram.py` stays a pure, single-file, stdlib-only module -- which
makes it copyable anywhere. It gains an argparse `__main__` so it works
as a detached calculator:

```
python3 modelctl_vram.py <model.gguf> [--ctx N] [--cache-type-k T]
                         [--cache-type-v T] [--mmproj PATH]
```

Output: detected architecture params (arch, layers, KV heads, head dims,
SWA window/pattern summary), then the footprint breakdown (weights / KV
with K and V shown separately / overhead / total). Defaults: ctx 32768,
cache types f16. Sharded models: a new `weights_bytes_on_disk(path)`
helper (shard-aware, own copy of the shard regex so the module stays
modelctl-free) sums sibling shards; `modelctl._local_weights_bytes`
delegates to it instead of duplicating the logic. Exit 1 with a message
on an unparseable GGUF (the calculator's whole point is exact math; the
heuristic path would be misleading here).
