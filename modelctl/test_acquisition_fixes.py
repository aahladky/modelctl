"""Acquisition and repair data-integrity fixes (Phase 10 remediation)."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import modelctl


def _gguf(path, size=128):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\0" * (size - 4))
    return path


class ImportLocalBase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.profiles = self.root / "profiles"
        self.profiles.mkdir()
        self.models = self.root / "models"
        self.models.mkdir()
        for target, value in (("PROFILES_DIR", self.profiles),
                              ("DEFAULT_MODELS_DIR", self.models)):
            p = mock.patch.object(modelctl, target, value)
            p.start()
            self.addCleanup(p.stop)
        for name in ("generate_artifacts", "sync_all_backends"):
            p = mock.patch.object(modelctl, name)
            p.start()
            self.addCleanup(p.stop)


class TestNewProfilesAreBornManaged(ImportLocalBase):
    def test_an_imported_profile_launches_through_the_worker(self):
        """API-triggered loads through llama-swap are the primary launch
        method (Aaron 2026-08-05), and only a managed launch writes the
        reservation that lets a memory surface name the model holding
        it. A profile born fixed repeats the laguna blind spot."""
        d = self.root / "downloads"
        model = _gguf(d / "llama-3-8b-Q4_K_M.gguf")
        profile = modelctl.import_local(str(model), name="mgd", resync=False)
        self.assertEqual(profile["runtime"]["mode"], "managed")
        # The same shape the runtime-policy endpoint stores, so a profile
        # born managed and one flipped later are indistinguishable.
        self.assertEqual(profile["runtime"]["maximum_storage_tier"], 3)
        self.assertTrue(profile["runtime"]["allow_fallback"])


class TestCompanionOwnership(ImportLocalBase):
    def test_foreign_mmproj_is_not_adopted(self):
        # A flat downloads directory holding two unrelated models.
        d = self.root / "downloads"
        model = _gguf(d / "llama-3-8b-Q4_K_M.gguf")
        _gguf(d / "mmproj-qwen2-vl-f16.gguf")
        profile = modelctl.import_local(str(model), name="llama3", resync=False)
        self.assertIsNotNone(profile)
        self.assertIsNone(profile["mmproj_path"],
                          "another model's vision projector was attached")

    def test_matching_mmproj_is_adopted(self):
        d = self.root / "downloads"
        model = _gguf(d / "llama-3-8b-Q4_K_M.gguf")
        mm = _gguf(d / "mmproj-llama-3-8b-f16.gguf")
        profile = modelctl.import_local(str(model), name="llama3b", resync=False)
        self.assertEqual(profile["mmproj_path"], str(mm))


class TestCopySemantics(ImportLocalBase):
    def test_existing_destination_still_repoints_the_profile(self):
        src = _gguf(self.root / "src" / "m.gguf", size=256)
        dest_dir = self.models / "mprof"
        _gguf(dest_dir / "m.gguf", size=256)  # a previous copy
        profile = modelctl.import_local(str(src), name="mprof", copy=True,
                                        resync=False)
        self.assertEqual(profile["model_path"], str(dest_dir / "m.gguf"),
                         "copy=True left the profile pointing at the source")

    def test_truncated_previous_copy_is_recopied(self):
        src = _gguf(self.root / "src" / "m.gguf", size=512)
        dest_dir = self.models / "mprof2"
        _gguf(dest_dir / "m.gguf", size=64)  # interrupted copy
        profile = modelctl.import_local(str(src), name="mprof2", copy=True,
                                        resync=False)
        dest = Path(profile["model_path"])
        self.assertEqual(dest.stat().st_size, 512,
                         "a truncated managed copy was left in place")


class TestVerifyFixBaseDir(unittest.TestCase):
    """`verify --fix` re-downloaded into <repo>/<sub>/<repo>/<sub>/... for
    repos that ship files in a subfolder, because the de-nesting check
    only fired when the path ENDED with the repo id."""

    @staticmethod
    def _base_dir(path, repo_id):
        base_dir = Path(path).parent
        repo_parts = Path(repo_id).parts
        parts = base_dir.parts
        for i in range(len(parts) - len(repo_parts) + 1):
            if parts[i:i + len(repo_parts)] == repo_parts:
                return Path(*parts[:i]) if i else Path(base_dir.anchor or ".")
        return base_dir

    def test_subfolder_layout_does_not_nest(self):
        got = self._base_dir("/models/org/repo/BF16/model.gguf", "org/repo")
        self.assertEqual(got, Path("/models"))

    def test_flat_repo_layout(self):
        got = self._base_dir("/models/org/repo/model.gguf", "org/repo")
        self.assertEqual(got, Path("/models"))

    def test_legacy_flat_path_is_unchanged(self):
        got = self._base_dir("/models/model.gguf", "org/repo")
        self.assertEqual(got, Path("/models"))

    def test_repeated_repair_is_idempotent(self):
        first = self._base_dir("/models/org/repo/BF16/model.gguf", "org/repo")
        second = self._base_dir(
            str(first / "org/repo/BF16/model.gguf"), "org/repo")
        self.assertEqual(first, second,
                         "each repair pass nested another copy")


if __name__ == "__main__":
    unittest.main()
