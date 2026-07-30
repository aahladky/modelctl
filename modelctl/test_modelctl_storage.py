import unittest
from unittest import mock

import modelctl_storage


class TestProbeStorage(unittest.TestCase):
    def test_probe_root(self):
        """Probing / should always return valid data on Linux."""
        info = modelctl_storage.probe_storage("/")
        self.assertTrue(info.mount_point)
        self.assertTrue(info.total_bytes > 0)
        # rotational may be None on virtual/container filesystems
        self.assertIn(info.transport, ("nvme", "sata", "ssd", "hdd", "unknown", "usb"))

    def test_probe_tmp(self):
        """Probing /tmp returns valid storage info."""
        info = modelctl_storage.probe_storage("/tmp")
        self.assertTrue(info.mount_point)
        self.assertTrue(info.total_bytes > 0)

    def test_fields_populated(self):
        info = modelctl_storage.probe_storage("/")
        self.assertTrue(info.path)
        self.assertTrue(info.mount_point)
        self.assertTrue(info.filesystem)
        self.assertIsInstance(info.block_devices, tuple)

    def test_allow_mmap_on_real_fs(self):
        info = modelctl_storage.probe_storage("/")
        # Root filesystem should allow mmap.
        self.assertTrue(info.allow_mmap)

    def test_probe_nonexistent_uses_parent(self):
        """Probing a nonexistent path should resolve to parent."""
        info = modelctl_storage.probe_storage("/nonexistent/path/here")
        self.assertTrue(info.mount_point)


class TestProbeAllMounts(unittest.TestCase):
    def test_returns_list(self):
        mounts = modelctl_storage.probe_all_mounts()
        self.assertIsInstance(mounts, list)
        # Should have at least one real mount.
        self.assertTrue(len(mounts) > 0)

    def test_all_have_capacity(self):
        mounts = modelctl_storage.probe_all_mounts()
        for m in mounts:
            self.assertTrue(m.total_bytes > 0, f"{m.mount_point} has no capacity")
