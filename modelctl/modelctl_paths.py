"""One definition of where modelctl state lives.

Historically modelctl.py read MODELCTL_HOME (the documented knob) while
four other modules read an undocumented MODELCTL_STATE_DIR, so setting
the documented variable split state across two trees: profiles in one,
the capability cache / hardware settings / runtime.db reservations /
web job DB in the other -- two control planes silently diverging.

Resolution order, applied identically by every module:
  1. MODELCTL_HOME   (documented; wins when both are set)
  2. MODELCTL_STATE_DIR  (legacy alias, kept so existing setups keep
     working)
  3. ~/.local/share/modelctl

This module must stay a leaf: no modelctl imports.
"""
import os
from pathlib import Path

STATE_DIR = Path(
    os.environ.get("MODELCTL_HOME")
    or os.environ.get("MODELCTL_STATE_DIR")
    or (Path.home() / ".local" / "share" / "modelctl"))
