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
`cpu`, `memory` and `pids` to the user manager, so `MemoryMax=20G`,
`CPUQuota=2000%` and `Nice=10` are enforced -- verified by reading them
back off the running units (`MemoryMax=21474836480`,
`CPUQuotaPerSecUSec=20s`, `Nice=10`).

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
  `MemoryMax=20G` -- a budget above the cgroup ceiling would be admitted
  by the planner and then OOM-killed by systemd.

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
