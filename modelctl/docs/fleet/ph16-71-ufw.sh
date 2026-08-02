#!/bin/bash
# ufw rules for ph16-71's RPC ports. NOT APPLIED -- run this on the
# laptop as a user with sudo.
#
# Why it is not applied: the unattended session that built and installed
# the RPC units had no passwordless sudo on ph16-71 (`sudo -n true` ->
# "interactive authentication is required"), and ufw needs root.
#
# Why it matters: ggml-rpc-server has NO AUTHENTICATION of any kind. It
# accepts a connection and then executes whatever graph it is handed and
# reads or writes whatever tensor buffers it is told to. The units bind
# 192.168.0.76 explicitly, so the ProtonVPN tunnel (10.2.0.2) and
# loopback are not served -- but every other host on 192.168.0.0/24
# currently can reach both ports. Measured, not assumed: a listener on
# 192.168.0.76:50052 was reached from the rig with ufw active, so the
# current policy does not restrict these ports.
#
# systemd's per-unit IPAddressAllow/IPAddressDeny would express the same
# restriction without root, and it was tried -- the user manager accepts
# the properties silently but does not enforce them (a --user unit with
# IPAddressDeny=any still reached 1.1.1.1:53). So ufw is the only
# mechanism available here.

set -euo pipefail

RIG=192.168.0.184

# Order matters: ufw evaluates rules top-down and stops at the first
# match, so the allow must precede the deny.
sudo ufw allow from "$RIG" to 192.168.0.76 port 50052 proto tcp \
  comment 'rpc-cuda0 from rig'
sudo ufw allow from "$RIG" to 192.168.0.76 port 50053 proto tcp \
  comment 'rpc-cpu0 from rig'

# Everything else on those ports, denied. `deny` rather than `reject` so
# a scanner gets a timeout instead of a courteous closed-port reply.
sudo ufw deny to any port 50052 proto tcp comment 'rpc-cuda0 default deny'
sudo ufw deny to any port 50053 proto tcp comment 'rpc-cpu0 default deny'

sudo ufw status numbered

# Verify from the rig afterwards -- the rule set is not the guarantee,
# the observed behaviour is:
#
#   rig$    python3 -c "import socket;socket.create_connection(('192.168.0.76',50052),timeout=5)"   # expect success
#   other$  python3 -c "import socket;socket.create_connection(('192.168.0.76',50052),timeout=5)"   # expect timeout
