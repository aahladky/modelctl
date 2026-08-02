#!/usr/bin/env python3
"""Full capability probe of the fleet nodes.

modelctl_fleet.probe_node() gives reachability, protocol and pin
agreement. The order also wants RTT and what each rpc-server reports for
backend/device/free VRAM, so this drives the same wire protocol and adds
DEVICE_COUNT + GET_DEVICE_MEMORY on the same connection.

Read-only: HELLO, DEVICE_COUNT and GET_DEVICE_MEMORY allocate nothing on
the node.
"""
import json
import socket
import struct
import sys
import time

sys.path.insert(0, "/home/aaron/workspace/moe-serving/modelctl")
import modelctl_fleet as fleet  # noqa: E402

RPC_CMD_GET_DEVICE_MEMORY = 11
RPC_CMD_HELLO = 14
RPC_CMD_DEVICE_COUNT = 15
CONN_CAPS = fleet.RPC_CONN_CAPS_SIZE


def recv_exactly(sock, n):
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"closed after {len(buf)}/{n}")
        buf += chunk
    return buf


def call(sock, cmd, payload=b""):
    sock.sendall(bytes([cmd]) + struct.pack("<Q", len(payload)) + payload)
    (size,) = struct.unpack("<Q", recv_exactly(sock, 8))
    return recv_exactly(sock, size) if size else b""


def probe(node):
    out = {"node": node.name, "endpoint": node.endpoint, "pin": node.pin,
           "variant": node.variant}
    t0 = time.perf_counter()
    sock = socket.create_connection((node.host, node.port), timeout=5.0)
    out["connect_ms"] = round((time.perf_counter() - t0) * 1000, 3)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        # HELLO, timed over several round trips for a stable RTT.
        t0 = time.perf_counter()
        body = call(sock, RPC_CMD_HELLO, bytes(CONN_CAPS))
        out["hello_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        out["protocol"] = f"{body[0]}.{body[1]}.{body[2]}"

        rtts = []
        for _ in range(20):
            t0 = time.perf_counter()
            call(sock, RPC_CMD_DEVICE_COUNT)
            rtts.append((time.perf_counter() - t0) * 1000)
        rtts.sort()
        out["rtt_ms_min"] = round(rtts[0], 3)
        out["rtt_ms_median"] = round(rtts[len(rtts) // 2], 3)
        out["rtt_ms_max"] = round(rtts[-1], 3)

        (count,) = struct.unpack("<I", call(sock, RPC_CMD_DEVICE_COUNT))
        out["device_count"] = count

        devices = []
        for dev in range(count):
            rsp = call(sock, RPC_CMD_GET_DEVICE_MEMORY, struct.pack("<I", dev))
            free_mem, total_mem = struct.unpack("<QQ", rsp)
            devices.append({
                "index": dev,
                "free_bytes": free_mem,
                "total_bytes": total_mem,
                "free_gib": round(free_mem / 2**30, 3),
                "total_gib": round(total_mem / 2**30, 3),
            })
        out["devices"] = devices
    finally:
        sock.close()
    return out


def main():
    expected = fleet.local_pin()
    print(f"rig checkout pin: {expected}")
    results = []
    for node in fleet.load_fleet():
        # The modelctl probe path, for pin agreement + reachability.
        std = fleet.probe_node(node, expected_pin=expected)
        rec = {"modelctl_probe": {
            "reachable": std.reachable, "protocol": std.protocol,
            "pin_agrees": std.pin_agrees, "detail": std.detail}}
        try:
            rec.update(probe(node))
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["node"] = node.name
            rec["endpoint"] = node.endpoint
        results.append(rec)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
