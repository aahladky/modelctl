import unittest
from unittest import mock

from textual.widgets import Input, ListView, Static

from modelctl_tui import PullWizardApp, StepIndicator, WizardState, next_screen_after


class TestNextScreenAfter(unittest.TestCase):
    def test_search_always_goes_to_quant(self):
        state = WizardState()
        self.assertEqual(next_screen_after("search", state), "quant")

    def test_quant_goes_to_vision_mtp_when_repo_has_mmproj(self):
        state = WizardState(repo_contents={"mmproj_files": [{"name": "mmproj-F16.gguf"}], "mtp_files": []})
        self.assertEqual(next_screen_after("quant", state), "vision_mtp")

    def test_quant_goes_to_vision_mtp_when_repo_has_mtp(self):
        state = WizardState(repo_contents={"mmproj_files": [], "mtp_files": [{"name": "model-mtp.gguf"}]})
        self.assertEqual(next_screen_after("quant", state), "vision_mtp")

    def test_quant_skips_vision_mtp_when_repo_has_neither(self):
        state = WizardState(repo_contents={"mmproj_files": [], "mtp_files": []})
        self.assertEqual(next_screen_after("quant", state), "configure")

    def test_quant_skips_vision_mtp_when_repo_contents_missing(self):
        # Defensive: repo_contents should always be set by the time this
        # runs, but don't crash if it isn't.
        state = WizardState(repo_contents=None)
        self.assertEqual(next_screen_after("quant", state), "configure")

    def test_vision_mtp_goes_to_configure(self):
        state = WizardState()
        self.assertEqual(next_screen_after("vision_mtp", state), "configure")

    def test_full_chain_after_configure(self):
        state = WizardState()
        self.assertEqual(next_screen_after("configure", state), "name")
        self.assertEqual(next_screen_after("name", state), "download")
        self.assertEqual(next_screen_after("download", state), "summary")


class TestStepIndicator(unittest.TestCase):
    def test_renders_all_steps_with_current_marked(self):
        indicator = StepIndicator(current="quant")
        rendered = str(indicator.render())
        self.assertIn("[quant]", rendered)
        self.assertNotIn("[search]", rendered)
        self.assertNotIn("[configure]", rendered)


class TestPullWizardAppBoots(unittest.IsolatedAsyncioTestCase):
    async def test_app_starts_on_search_screen(self):
        app = PullWizardApp()
        async with app.run_test() as pilot:
            self.assertEqual(app.screen.STEP, "search")


class TestSearchScreen(unittest.IsolatedAsyncioTestCase):
    async def test_typing_query_and_pressing_enter_shows_results(self):
        fake_results = [
            {"repo_id": "unsloth/Qwen3.5-35B-A3B-GGUF", "downloads": 138881, "likes": 854,
             "is_gguf": True, "has_mtp": False, "contents": {"quant_groups": [], "mmproj_files": [], "mtp_files": []}},
        ]
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.search_models", return_value=fake_results) as mock_search:
            async with app.run_test() as pilot:
                await pilot.click("#search-input")
                await pilot.press(*"qwen3.5", "enter")
                await app.workers.wait_for_complete()
                await pilot.pause()
                mock_search.assert_called_once()
                results_view = app.screen.query_one("#search-results", ListView)
                self.assertEqual(len(results_view.children), 1)

    async def test_selecting_a_result_stores_repo_id_and_contents(self):
        fake_results = [
            {"repo_id": "unsloth/Qwen3.5-35B-A3B-GGUF", "downloads": 1, "likes": 1,
             "is_gguf": True, "has_mtp": False, "contents": {"quant_groups": [{"label": "Q4_K_M", "files": ["a.gguf"], "sharded": False, "total_size": 100}], "mmproj_files": [], "mtp_files": []}},
        ]
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.search_models", return_value=fake_results):
            async with app.run_test() as pilot:
                await pilot.click("#search-input")
                await pilot.press(*"qwen3.5", "enter")
                await app.workers.wait_for_complete()
                await pilot.pause()
                await pilot.click("ListItem")
                await pilot.pause()
                self.assertEqual(app.state.repo_id, "unsloth/Qwen3.5-35B-A3B-GGUF")
                self.assertEqual(app.state.repo_contents, fake_results[0]["contents"])
                self.assertEqual(app.screen.STEP, "quant")

    async def test_no_results_shows_status_message(self):
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.search_models", return_value=[]):
            async with app.run_test() as pilot:
                await pilot.click("#search-input")
                await pilot.press(*"nonexistentmodel12345", "enter")
                await app.workers.wait_for_complete()
                await pilot.pause()
                status = app.screen.query_one("#search-status", Static)
                self.assertIn("No results", str(status.render()))

    async def test_search_exception_does_not_crash_app(self):
        app = PullWizardApp()
        with mock.patch(
            "modelctl_tui.modelctl.search_models",
            side_effect=Exception("network error"),
        ):
            async with app.run_test() as pilot:
                await pilot.click("#search-input")
                await pilot.press(*"qwen3.5", "enter")
                await app.workers.wait_for_complete()
                await pilot.pause()
                # The app must still be alive and responsive, not torn down
                # by an uncaught exception from the search worker.
                self.assertTrue(app.is_running)
                status = app.screen.query_one("#search-status", Static)
                self.assertIn("failed", str(status.render()).lower())
                # Confirm we can still interact with the screen afterwards.
                self.assertEqual(app.screen.STEP, "search")


if __name__ == "__main__":
    unittest.main()
