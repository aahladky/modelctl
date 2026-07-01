import unittest

from textual.widgets import Static

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


if __name__ == "__main__":
    unittest.main()
