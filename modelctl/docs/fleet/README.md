# Fleet node ph16-71

The laptop (`192.168.0.76`, i9-13900HX + RTX 4080 Laptop) serving two
llama.cpp RPC endpoints to the rig over 2.5GbE.

Discovery record and hardware facts: `moe-review/fleet-ph16-71.md`
(re-verified 2026-08-02, unchanged).

## What runs there

| unit | port | device | binary |
|---|---|---|---|
| `rpc-cuda0.service` | 50052 | `CUDA0` (RTX 4080 Laptop, sm_89) | `/home/aaron/rpc/cuda/build/bin/ggml-rpc-server` |
| `rpc-cpu0.service` | 50053 | `CPU` (i9-13900HX) | `/home/aaron/rpc/cpu/build/bin/ggml-rpc-server` |

Both built from llama.cpp `85b7e6556`, the commit this superproject
pins, with gcc-13 (CUDA 12.4's `host_config.h` hard-errors above gcc 13
and the laptop defaults to gcc-15). Each build directory carries a
`build-manifest.json` next to the binary recording the commit,
toolchain, and build time -- `ggml-rpc-server` links only ggml, never
libcommon, so it has no `--version` and no embedded build-info, and the
manifest is how a probe learns which commit a node is running.

Both units are **user** units, not system units: the account has no
passwordless sudo on that box. Linger is enabled and cgroup v2 delegates
`cpu`, `memory` and `pids` to the user manager, so `MemoryMax`,
`CPUQuota=2000%` and `Nice=10` are enforced -- verified by reading them
back off the running units (`CPUQuotaPerSecUSec=20s`, `Nice=10`).

`rpc-cpu0`'s `MemoryMax` was raised 20G -> **26G** on 2026-08-02 with
`systemctl --user set-property`, which applies to the live cgroup
without restarting the unit: `ExecMainPID` stayed 1472071 and
`NRestarts=0` across the change, so the node's presence never lapsed.
The unit file in this directory still reads `MemoryMax=20G` -- the
runtime drop-in under `~/.config/systemd/user.control/` overrides it,
and the committed file is the install-time default, not the live value.
Read the live value with
`systemctl --user show rpc-cpu0 -p MemoryMax` (currently
`27917287424`).

## Security state -- read this before using the node

`ggml-rpc-server` has **no authentication**. Anything that can open a
TCP connection to these ports can make the node execute graphs and read
and write its buffers.

* Both units bind `192.168.0.76` explicitly, so the ProtonVPN tunnel
  (`10.2.0.2`) and loopback are never served. Verified with `ss -ltnp`.
* The ports are **not yet restricted to the rig.** `ph16-71-ufw.sh`
  contains the exact rules and has not been applied -- it needs root,
  and the session that installed the units had no passwordless sudo.
  Until it runs, any host on `192.168.0.0/24` can reach both ports.
* systemd `IPAddressAllow`/`IPAddressDeny` was tried as a root-free
  substitute. The user manager accepts the properties and does not
  enforce them; a `--user` unit with `IPAddressDeny=any` still reached
  an outside address. Do not rely on it.

## Registry

The two endpoints are enrolled as `ph16-71-cuda0` and `ph16-71-cpu0` in
`modelctl_fleet`'s registry (`$MODELCTL_HOME/fleet.json`), each with a
declared budget that is an operator ceiling rather than a measurement of
what the node has:

* `ph16-71-cuda0` / `CUDA0`: 10 GiB declared of 11.6 GiB present.
* `ph16-71-cpu0` / `CPU`: 16 GiB declared, under the unit's own
  `MemoryMax` (26G since 2026-08-02) -- a budget above the cgroup
  ceiling would be admitted by the planner and then OOM-killed by
  systemd.

### Changing a budget

`modelctl fleet set-budget <node> <device> <bytes>` is the one writer of
that number (the console's fleet page submits the same primitive). It
takes the state lock, refuses anything above the device's ceiling, and
reports which profiles' recorded planning inputs the change stales:

    modelctl fleet set-budget ph16-71-cpu0 CPU 19327352832   # 18 GiB

The **ceiling** is not the device total. For a cpu device it is the
`MemoryMax` recorded for its unit, for a gpu device the reported device
total, in both cases minus runtime headroom (`max(1 GiB, 5%)`) for the
RPC server's own resident set and staging buffers. cpu0 today: 26 GiB
cap -> **24.70 GiB ceiling**; cuda0: 11.6 GiB total -> **10.60 GiB**.

The cap is operator-recorded, never probed -- reading it means asking
another machine's service manager, and no planning path may shell out
over SSH. After changing `MemoryMax` on the node, record it here or the
ceiling will still be derived from the old number:

    ssh aaron@192.168.0.76 'systemctl --user set-property rpc-cpu0 MemoryMax=26G'
    ssh aaron@192.168.0.76 'systemctl --user show rpc-cpu0 -p MemoryMax'
    modelctl fleet set-cap ph16-71-cpu0 CPU 27917287424     # what it now says

`set-property` applies live and does **not** restart the unit, which
matters: a restart drops the node's presence and any graph running
across it. A device with no recorded cap falls back to the reported
total and says so in the ceiling basis -- a guess would silently
authorize a budget the cgroup refuses at load time.

A budget is a planning input: it is spent by the same admission path a
local card's VRAM is, so it is recorded with the rest of a profile's
inputs and a change stales every stored-input plan built against the old
number. Stale means "reported, not rewritten" -- the stored plan keeps
placing against the number it was built for until it is replanned, and
the launch preview says so.

A registered node is inert until a presence probe records it reachable
*and* built at this checkout's pin, and that record expires after 15
minutes. Planning never opens a socket; it reads the stored record.

    python3 -c "import modelctl_fleet as f; [print(p.to_dict()) for p in f.refresh_presence()]"

## Rebuilding after the pin moves

The units point at fixed paths, so a rebuild in place is enough:

    ssh aaron@192.168.0.76 'cd /home/aaron/rpc/cuda && git fetch && git checkout <pin> && bash /home/aaron/rpc/build_cuda.sh'
    ssh aaron@192.168.0.76 'bash /home/aaron/rpc/manifest.sh'
    ssh aaron@192.168.0.76 'systemctl --user restart rpc-cuda0 rpc-cpu0'

Then re-enroll so the registry's recorded pin matches, or every probe
will correctly report `pin_agrees: false` and the planner will stop
offering fleet plans.
