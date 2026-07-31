"""Multi-file mutation transactions for modelctl.

Prevents partial profiles, artifacts, or llama-swap configuration after
a failed operation. Stages changes, then writes each file atomically
(tmp + rename via modelctl_fsutil) under the cross-process state lock.

What this does and does not guarantee: each individual file is always
either its old bytes or its new bytes (never truncated), and an
in-process failure rolls every file back. A hard kill mid-commit can
still leave a MIX of old and new files -- the backup directory survives
for manual recovery in that case (see orphaned_backups()).

Public API:
    Transaction context manager
    orphaned_backups()
"""
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import modelctl
import modelctl_fsutil


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
        # pid + random suffix: two same-named transactions in the same
        # second (CLI racing a web job) must never share a backup dir.
        unique = f"{self.name}_{int(time.time())}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        self._backup_dir = Path(modelctl.STATE_DIR) / ".tx_backups" / unique
        self._backup_dir.mkdir(parents=True)
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
        """Apply all staged changes under the state lock.

        1. Backup existing files (kept until the with-block exits, so a
           hard kill mid-commit leaves them for manual recovery).
        2. Write every staged file atomically (old bytes or new bytes,
           never truncated).
        Profile JSON is round-trip validated before writing; artifact
        and config content is written as staged.
        """
        with modelctl_fsutil.state_lock():
            self._commit_locked()

    def _commit_locked(self):
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
            modelctl_fsutil.atomic_write_text(path, content)

        # Write artifacts.
        for path, content in self._staged_artifacts:
            path.parent.mkdir(parents=True, exist_ok=True)
            modelctl_fsutil.atomic_write_text(path, content)

        # Write configs.
        for path, content in self._staged_configs:
            path.parent.mkdir(parents=True, exist_ok=True)
            modelctl_fsutil.atomic_write_text(path, content)

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


def orphaned_backups():
    """Backup dirs left by transactions that never exited their with-block.

    A hard kill mid-commit is the only way these survive; each one means
    a mutation may have landed partially. Surfaced by diagnostics so the
    operator can inspect and delete them -- never auto-restored, because
    the files on disk may be newer than the backups.
    """
    root = Path(modelctl.STATE_DIR) / ".tx_backups"
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())
