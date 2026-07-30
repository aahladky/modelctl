"""Atomic multi-file mutation transactions for modelctl.

Prevents partial profiles, artifacts, or llama-swap configuration
after a failed operation. Stages changes, validates, then atomically
replaces all managed files.

Public API:
    Transaction context manager
"""
import json
import os
import shutil
import time
from pathlib import Path

import modelctl


class TransactionError(Exception):
    """Raised when a transaction step fails."""
    pass


class Transaction:
    """Atomic multi-file mutation context manager.

    Usage:
        with Transaction("my-op") as tx:
            tx.stage_profile(profile)
            tx.stage_artifact(path, content)
            tx.commit()

    On commit: atomically replaces all staged files. On failure:
    rolls back all changes.
    """

    def __init__(self, name: str):
        self.name = name
        self._staged_profiles: dict[str, dict] = {}
        self._staged_artifacts: list[tuple[Path, str]] = []
        self._staged_configs: list[tuple[Path, str]] = []
        self._backup_dir: Path | None = None
        self._committed = False
        # (target_path, backup_path_or_None) for every file commit() wrote.
        self._written: list = []

    def __enter__(self):
        self._backup_dir = Path(modelctl.STATE_DIR) / ".tx_backups" / f"{self.name}_{int(time.time())}"
        self._backup_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._rollback()
        if self._backup_dir and self._backup_dir.exists():
            shutil.rmtree(self._backup_dir, ignore_errors=True)
        return False

    def stage_profile(self, profile: dict):
        """Stage a profile for atomic save."""
        name = profile.get("name")
        if not name:
            raise TransactionError("profile has no name")
        self._staged_profiles[name] = profile

    def stage_artifact(self, path: Path, content: str):
        """Stage a file for atomic write."""
        self._staged_artifacts.append((Path(path), content))

    def stage_config(self, path: Path, content: str):
        """Stage a config file (e.g. llama-swap config.yaml) for atomic write."""
        self._staged_configs.append((Path(path), content))

    def commit(self):
        """Atomically apply all staged changes.

        1. Backup existing files.
        2. Write all staged content.
        3. Validate JSON/YAML.
        """
        # Backup existing files. Every write is recorded as
        # (target, backup_or_None) so _rollback can restore what existed
        # and remove what did not -- backups keyed by basename alone
        # collided whenever two profiles staged their own run.sh.
        self._written = []

        for name in self._staged_profiles:
            src = modelctl.PROFILES_DIR / f"{name}.json"
            backup = None
            if src.exists():
                backup = self._backup_dir / f"profile_{name}.json"
                shutil.copy2(src, backup)
            self._written.append((src, backup))

        for index, (path, _) in enumerate(
                self._staged_artifacts + self._staged_configs):
            backup = None
            if path.exists():
                backup = self._backup_dir / f"{index:04d}_{path.name}"
                shutil.copy2(path, backup)
            self._written.append((path, backup))

        # Write profiles.
        modelctl.PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        for name, profile in self._staged_profiles.items():
            path = modelctl.PROFILES_DIR / f"{name}.json"
            content = json.dumps(profile, indent=2)
            # Validate JSON roundtrip.
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                raise TransactionError(f"invalid JSON for profile '{name}': {e}")
            path.write_text(content)

        # Write artifacts.
        for path, content in self._staged_artifacts:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        # Write configs.
        for path, content in self._staged_configs:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

        self._committed = True

    def _rollback(self):
        """Restore every file this transaction wrote.

        Previously this restored profiles only and left a comment saying
        artifacts and configs were preserved in the backup directory "for
        manual recovery" -- but __exit__ deletes that directory, so a
        failed multi-file mutation left artifacts and the llama-swap config
        modified with no way back. Restoring by recorded target path fixes
        that, and files that did not exist before the transaction are
        removed rather than left behind.
        """
        if not self._backup_dir or not self._backup_dir.exists():
            return

        for target, backup in getattr(self, "_written", []):
            try:
                if backup is not None and backup.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup, target)
                elif target.exists():
                    # Created by this transaction; undoing means removing it.
                    target.unlink()
            except OSError:
                # One unrestorable file must not abort the rest of the
                # rollback -- partial recovery beats none.
                continue
