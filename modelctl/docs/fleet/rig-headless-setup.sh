#!/bin/bash
# One-time root setup for `modelctl headless` and `modelctl
# remote-hands`. RUN THIS YOURSELF, AS ROOT.
#
#   sudo modelctl/docs/fleet/rig-headless-setup.sh
#
# It does exactly three things and prints all three before doing any:
#
#   1. loginctl enable-linger <user>
#   2. installs a sudoers drop-in permitting NOPASSWD for precisely two
#      commands: `systemctl isolate multi-user.target` and
#      `systemctl isolate graphical.target`.
#   3. tailscale set --operator=<user>, so `tailscale funnel` can be
#      configured without root by `modelctl remote-hands on/off`.
#
# Step 3 exposes nothing by itself. It grants the right to *configure*
# serve/funnel; `modelctl remote-hands` still ships with its unit
# disabled, its listener on loopback and no funnel, and Aaron turns it
# on per use. It lives here rather than in a second script because the
# alternative was `modelctl remote-hands` shelling out to sudo on the
# exposure path, and a toggle that can ask for root is a toggle that can
# be talked into asking for root.
#
# It does NOT change the default target. `modelctl headless` never does
# either. A reboot always comes up graphical, and that is the escape
# hatch if a headless rig ever wedges -- do not take it away.
#
# Why linger, and why it matters beyond headless mode: llama-swap and the
# web console are user services under user@1000. With Linger=no, ending
# the last session (logging out, or dropping the graphical target)
# terminates the user manager and takes the live serving stack with it.
# Enabling linger hardens the stack against a plain logout whether or not
# headless mode is ever used.
#
# Why the sudoers drop-in is this narrow: `systemctl isolate` with an
# unconstrained argument is a general-purpose way to reach any target,
# including rescue.target and emergency.target -- a root shell without a
# password. Naming both targets in full removes that. The command paths
# are absolute because sudoers matches the command line literally; a
# rule for `systemctl` would not constrain a different binary of that
# name earlier in PATH.
#
# Idempotent: re-running it changes nothing that is already correct.

set -euo pipefail

DROPIN=/etc/sudoers.d/modelctl-headless
SYSTEMCTL=/usr/bin/systemctl
TAILSCALE=/usr/bin/tailscale
TARGET_USER="${SUDO_USER:-${1:-}}"

if [ "$(id -u)" -ne 0 ]; then
    echo "This script needs root: sudo $0" >&2
    exit 1
fi

if [ -z "$TARGET_USER" ]; then
    echo "Cannot tell which user to set this up for." >&2
    echo "Run it with sudo (so SUDO_USER is set), or pass the username:" >&2
    echo "  sudo $0 aaron" >&2
    exit 1
fi

if ! id "$TARGET_USER" >/dev/null 2>&1; then
    echo "No such user: $TARGET_USER" >&2
    exit 1
fi

# The drop-in authorizes these exact command lines. modelctl_display.py
# builds the same two from the same absolute path; if this check fails,
# the two have drifted and the drop-in would authorize commands modelctl
# never runs (and not the ones it does).
if [ ! -x "$SYSTEMCTL" ]; then
    echo "$SYSTEMCTL is not executable on this box." >&2
    echo "modelctl_display.SYSTEMCTL must be updated to match before this" >&2
    echo "drop-in will authorize anything modelctl actually runs." >&2
    exit 1
fi

RULE="$TARGET_USER ALL=(root) NOPASSWD: $SYSTEMCTL isolate multi-user.target, $SYSTEMCTL isolate graphical.target"

cat <<EOF

About to make exactly three changes, for user '$TARGET_USER':

  1. loginctl enable-linger $TARGET_USER
     (so user@$(id -u "$TARGET_USER") -- llama-swap, the web console --
     survives the last session ending)

  2. write $DROPIN containing exactly:

$RULE

  3. $TAILSCALE set --operator=$TARGET_USER
     (so \`modelctl remote-hands on/off\` can configure the funnel
     without root -- it publishes nothing by itself)

Not changed: the default systemd target. A reboot still comes up
graphical. Not changed: any funnel or serve configuration.

EOF

# --- 1. linger ------------------------------------------------------------
if loginctl show-user "$TARGET_USER" --property=Linger 2>/dev/null \
        | grep -qx 'Linger=yes'; then
    echo "[1/3] linger already enabled for $TARGET_USER -- nothing to do"
else
    loginctl enable-linger "$TARGET_USER"
    echo "[1/3] linger enabled for $TARGET_USER"
fi

# --- 2. sudoers drop-in ---------------------------------------------------
if [ -f "$DROPIN" ] && [ "$(cat "$DROPIN")" = "$RULE" ]; then
    echo "[2/3] $DROPIN already correct -- nothing to do"
else
    # Validate before installing, never after: a syntactically broken file
    # under /etc/sudoers.d breaks sudo for every user on the box, and the
    # way back from that needs the sudo it just broke.
    TMP="$(mktemp /tmp/modelctl-headless-sudoers.XXXXXX)"
    trap 'rm -f "$TMP"' EXIT
    printf '%s\n' "$RULE" > "$TMP"
    if ! visudo -c -f "$TMP" >/dev/null; then
        echo "Refusing to install: visudo rejected the rule." >&2
        visudo -c -f "$TMP" >&2 || true
        exit 1
    fi
    install -m 0440 -o root -g root "$TMP" "$DROPIN"
    echo "[2/3] installed $DROPIN"
fi

# --- 3. tailscale operator -------------------------------------------------
# `tailscale funnel` writes tailscaled's serve config, which is root-owned
# state; without an operator the only ways to reach it are running the
# whole toggle as root or teaching `modelctl remote-hands` to call sudo.
# The operator grant is the narrower of the three: it is per-user, it is
# revocable with `tailscale set --operator=`, and it authorizes
# configuring serve/funnel and nothing else on this box.
#
# Not fatal if tailscale is absent -- `modelctl headless` does not need
# it, and the rest of this script has already run.
# Run unconditionally rather than after a "is it already set?" probe:
# setting the operator to the value it already has is a no-op, and the
# probe would have to guess at a pref field name that tailscaled omits
# entirely while it is unset -- a guess that fails open into running the
# set anyway. So run the set, then report what tailscaled says.
if [ ! -x "$TAILSCALE" ]; then
    echo "[3/3] $TAILSCALE not found -- skipped."
    echo "      \`modelctl remote-hands\` needs it; \`modelctl headless\` does not."
elif "$TAILSCALE" set --operator="$TARGET_USER"; then
    echo "[3/3] tailscale operator set to $TARGET_USER"
else
    echo "[3/3] FAILED to set the tailscale operator." >&2
    echo "      \`modelctl remote-hands on\` will refuse; \`modelctl headless\` is unaffected." >&2
fi

echo
echo "Verifying as $TARGET_USER (this is the same probe modelctl uses):"
for target in multi-user.target graphical.target; do
    if runuser -u "$TARGET_USER" -- \
            sudo -n -l "$SYSTEMCTL" isolate "$target" >/dev/null 2>&1; then
        echo "  OK    sudo -n $SYSTEMCTL isolate $target"
    else
        echo "  FAIL  sudo -n $SYSTEMCTL isolate $target"
    fi
done

if [ -x "$TAILSCALE" ]; then
    if runuser -u "$TARGET_USER" -- "$TAILSCALE" funnel status >/dev/null 2>&1
    then
        echo "  OK    $TARGET_USER can read the funnel config"
    else
        echo "  FAIL  $TARGET_USER cannot read the funnel config"
    fi
    OPERATOR=$($TAILSCALE debug prefs 2>/dev/null | grep -i operator || true)
    echo "  tailscaled prefs:${OPERATOR:- (no operator field reported)}"
fi

cat <<'EOF'

Done. From here:

  modelctl headless status    # should show linger yes, sudoers ready
  modelctl headless verify    # detached round-trip, files its own evidence
  modelctl remote-hands status  # should show exposure: hidden

Nothing is exposed to the internet by this script. `modelctl
remote-hands on` is the only thing that publishes anything, and it is
per-use and torn down by `off`.

To undo: rm /etc/sudoers.d/modelctl-headless; tailscale set --operator=
(empty, to revoke); and loginctl disable-linger <user> (note that
disabling linger also removes the protection the live stack now has
against a plain logout).
EOF
