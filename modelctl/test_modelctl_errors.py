import unittest
from unittest import mock

import modelctl_errors


class TestValidationMessage(unittest.TestCase):
    def test_basic_creation(self):
        msg = modelctl_errors.ValidationMessage(
            code="binary_missing",
            severity="error",
            summary="Binary not found",
            detail="/usr/bin/llama-server",
            recovery="Install llama.cpp",
            field="binary",
        )
        self.assertEqual(msg.code, "binary_missing")
        self.assertEqual(msg.severity, "error")
        self.assertEqual(msg.summary, "Binary not found")
        self.assertEqual(msg.field, "binary")

    def test_as_tuple(self):
        msg = modelctl_errors.ValidationMessage(
            code="test", severity="warning", summary="msg"
        )
        self.assertEqual(msg.as_tuple(), ("warning", "msg"))

    def test_frozen(self):
        msg = modelctl_errors.ValidationMessage(
            code="test", severity="info", summary="msg"
        )
        with self.assertRaises(AttributeError):
            msg.code = "changed"


class TestHelpers(unittest.TestCase):
    def test_errors_only(self):
        msgs = [
            modelctl_errors.ValidationMessage(code="a", severity="error", summary="e"),
            modelctl_errors.ValidationMessage(code="b", severity="warning", summary="w"),
            modelctl_errors.ValidationMessage(code="c", severity="error", summary="e2"),
        ]
        self.assertEqual(len(modelctl_errors.errors_only(msgs)), 2)

    def test_warnings_only(self):
        msgs = [
            modelctl_errors.ValidationMessage(code="a", severity="error", summary="e"),
            modelctl_errors.ValidationMessage(code="b", severity="warning", summary="w"),
        ]
        self.assertEqual(len(modelctl_errors.warnings_only(msgs)), 1)

    def test_has_errors(self):
        self.assertFalse(modelctl_errors.has_errors([]))
        self.assertFalse(modelctl_errors.has_errors([
            modelctl_errors.ValidationMessage(code="a", severity="warning", summary="w"),
        ]))
        self.assertTrue(modelctl_errors.has_errors([
            modelctl_errors.ValidationMessage(code="a", severity="error", summary="e"),
        ]))

    def test_max_severity(self):
        self.assertEqual(modelctl_errors.max_severity([]), "info")
        msgs = [
            modelctl_errors.ValidationMessage(code="a", severity="info", summary="i"),
            modelctl_errors.ValidationMessage(code="b", severity="warning", summary="w"),
        ]
        self.assertEqual(modelctl_errors.max_severity(msgs), "warning")

    def test_from_preflight_tuples(self):
        tuples = [
            ("error", "moe_cache mode=manual requested but backend does not support moe_weight_transfer_cache"),
            ("warning", "prefetch requested but unsupported"),
            ("ok", "binary supports MoE weight transfer cache"),
        ]
        result = modelctl_errors.from_preflight_tuples(tuples)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].code, "cache_feature_unsupported")
        self.assertEqual(result[0].severity, "error")
        self.assertEqual(result[1].code, "cache_prefetch_unsupported")
        self.assertEqual(result[2].severity, "ok")
