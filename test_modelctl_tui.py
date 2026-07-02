import unittest
from unittest import mock

from textual.widgets import Button, Input, ListView, Static

import modelctl
from modelctl_tui import (
    ConfigureScreen,
    DownloadScreen,
    NameScreen,
    PullWizardApp,
    QuantPickScreen,
    StepIndicator,
    SummaryScreen,
    VisionMtpScreen,
    WizardState,
    next_screen_after,
)


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


class TestQuantPickScreen(unittest.IsolatedAsyncioTestCase):
    def _state_with_contents(self, mmproj=None, mtp=None):
        return WizardState(
            repo_id="unsloth/Qwen3.5-35B-A3B-GGUF",
            repo_contents={
                "quant_groups": [
                    {"label": "Q4_K_M", "files": ["a.gguf"], "sharded": False, "total_size": 100},
                    {"label": "Q5_K_M", "files": ["b.gguf"], "sharded": False, "total_size": 120},
                ],
                "mmproj_files": mmproj or [],
                "mtp_files": mtp or [],
            },
        )

    async def test_lists_quant_groups_from_state(self):
        app = PullWizardApp()
        async with app.run_test() as pilot:
            app.state = self._state_with_contents()
            await app.push_screen(QuantPickScreen())
            await pilot.pause()
            options = app.screen.query_one("#quant-options", ListView)
            self.assertEqual(len(options.children), 2)

    async def test_picking_quant_with_no_extras_skips_to_configure(self):
        app = PullWizardApp()
        async with app.run_test() as pilot:
            app.state = self._state_with_contents()  # no mmproj/mtp
            await app.push_screen(QuantPickScreen())
            await pilot.pause()
            await pilot.click("ListItem")
            await pilot.pause()
            self.assertEqual(app.state.quant_group["label"], "Q4_K_M")
            self.assertEqual(app.screen.STEP, "configure")

    async def test_picking_quant_with_mmproj_goes_to_vision_mtp(self):
        app = PullWizardApp()
        async with app.run_test() as pilot:
            app.state = self._state_with_contents(mmproj=[{"name": "mmproj-F16.gguf", "size": 500}])
            await app.push_screen(QuantPickScreen())
            await pilot.pause()
            await pilot.click("ListItem")
            await pilot.pause()
            self.assertEqual(app.screen.STEP, "vision_mtp")


class TestVisionMtpScreen(unittest.IsolatedAsyncioTestCase):
    async def test_lists_mmproj_and_mtp_options_with_skip(self):
        app = PullWizardApp()
        async with app.run_test() as pilot:
            app.state = WizardState(repo_contents={
                "mmproj_files": [{"name": "mmproj-F16.gguf", "size": 500}],
                "mtp_files": [{"name": "model-mtp.gguf", "size": 300}],
            })
            await app.push_screen(VisionMtpScreen())
            await pilot.pause()
            mmproj_options = app.screen.query_one("#mmproj-options", ListView)
            mtp_options = app.screen.query_one("#mtp-options", ListView)
            # +1 each for the "skip" entry
            self.assertEqual(len(mmproj_options.children), 2)
            self.assertEqual(len(mtp_options.children), 2)

    async def test_picking_mmproj_and_skipping_mtp_advances_to_configure(self):
        app = PullWizardApp()
        async with app.run_test() as pilot:
            app.state = WizardState(repo_contents={
                "mmproj_files": [{"name": "mmproj-F16.gguf", "size": 500}],
                "mtp_files": [],
            })
            await app.push_screen(VisionMtpScreen())
            await pilot.pause()
            await pilot.click("#mmproj-options ListItem")
            await pilot.click("#continue-button")
            await pilot.pause()
            self.assertEqual(app.state.mmproj_choice["name"], "mmproj-F16.gguf")
            self.assertIsNone(app.state.mtp_choice)
            self.assertEqual(app.screen.STEP, "configure")

    async def test_clicking_trailing_skip_row_on_nonempty_list_yields_none(self):
        # Boundary case for the skip-last reordering: a list with a REAL
        # file plus a trailing "(skip)" row, where the skip row itself
        # (the LAST ListItem) is the one chosen. pilot.click("ListItem")
        # always hits the first DOM match (the real file, given the
        # reordering), so it can't reach the skip row here -- instead we
        # drive the ListView's own keyboard cursor (arrow-key highlight,
        # same mechanism on_button_pressed reads via `.index`) down past
        # the real file to land on the trailing skip row.
        app = PullWizardApp()
        async with app.run_test() as pilot:
            app.state = WizardState(repo_contents={
                "mmproj_files": [{"name": "mmproj-F16.gguf", "size": 500}],
                "mtp_files": [],
            })
            await app.push_screen(VisionMtpScreen())
            await pilot.pause()
            mmproj_options = app.screen.query_one("#mmproj-options", ListView)
            # Sanity: 2 rows total (1 real file + 1 trailing skip), so
            # index 1 (the last one) is the skip row, not the real file.
            self.assertEqual(len(mmproj_options.children), 2)
            app.set_focus(mmproj_options)
            await pilot.pause()
            # ListView.index starts at None; each "down" press moves the
            # highlight one row, landing on the highest index (the skip
            # row) after `len(files) + 1` presses.
            for _ in range(len(app.screen._mmproj_files) + 1):
                await pilot.press("down")
            await pilot.pause()
            # Confirm we actually reached the trailing skip row, not the
            # real file, before trusting the Continue click below.
            self.assertEqual(mmproj_options.index, 1)
            await pilot.click("#continue-button")
            await pilot.pause()
            self.assertIsNone(app.state.mmproj_choice)


class TestConfigureScreen(unittest.IsolatedAsyncioTestCase):
    async def test_prefills_from_defaults(self):
        fake_defaults = {
            "device": "", "split_mode": "layer", "tensor_split": "3,1",
            "ctx": 32768, "kv_quant": "q8_0", "flash_attn": "auto",
            "ttl": 3600, "mtp": "off",
        }
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.load_defaults", return_value=fake_defaults):
            async with app.run_test() as pilot:
                app.state = WizardState()
                await app.push_screen(ConfigureScreen())
                await pilot.pause()
                ctx_input = app.screen.query_one("#config-ctx", Input)
                self.assertEqual(ctx_input.value, "32768")

    async def test_submit_stores_config_and_advances(self):
        fake_defaults = {
            "device": "", "split_mode": "layer", "tensor_split": "3,1",
            "ctx": 32768, "kv_quant": "q8_0", "flash_attn": "auto",
            "ttl": 3600, "mtp": "off",
        }
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.load_defaults", return_value=fake_defaults), \
             mock.patch("modelctl_tui.modelctl.preflight", return_value=(True, "llama-server", {}, [])):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x", quant_group={"label": "Q4_K_M", "files": ["a.gguf"]},
                )
                await app.push_screen(ConfigureScreen())
                await pilot.pause()
                await pilot.click("#submit-config")
                await pilot.pause()
                self.assertEqual(app.state.config["ctx"], "32768")
                self.assertEqual(app.screen.STEP, "name")

    async def test_preflight_warning_does_not_block_submit(self):
        # Not a file-not-found message, so it must survive the filter and
        # be stored/displayed -- proves the filter is selective, not a
        # blanket "drop everything preflight says" hammer.
        fake_defaults = {
            "device": "", "split_mode": "layer", "tensor_split": "3,1",
            "ctx": 32768, "kv_quant": "q8_0", "flash_attn": "auto",
            "ttl": 3600, "mtp": "off",
        }
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.load_defaults", return_value=fake_defaults), \
             mock.patch("modelctl_tui.modelctl.preflight",
                         return_value=(False, None, {}, ["ERROR: llama-server not found"])):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x", quant_group={"label": "Q4_K_M", "files": ["a.gguf"]},
                )
                await app.push_screen(ConfigureScreen())
                await pilot.pause()
                await pilot.click("#submit-config")
                await pilot.pause()
                self.assertIn("ERROR: llama-server not found", app.state.warnings)
                self.assertEqual(app.screen.STEP, "name")

    async def test_file_not_found_preflight_messages_are_filtered_out(self):
        # probe_profile's model_path is a bare repo filename at this stage
        # (download hasn't happened yet -- that's Task 9's DownloadScreen),
        # so preflight()'s file-existence errors are guaranteed false
        # positives here and must not reach state.warnings or the on-screen
        # warning area at all.
        fake_defaults = {
            "device": "", "split_mode": "layer", "tensor_split": "3,1",
            "ctx": 32768, "kv_quant": "q8_0", "flash_attn": "auto",
            "ttl": 3600, "mtp": "off",
        }
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.load_defaults", return_value=fake_defaults), \
             mock.patch("modelctl_tui.modelctl.preflight",
                         return_value=(False, None, {},
                                       ["ERROR: model file not found on disk: some/path.gguf"])):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x", quant_group={"label": "Q4_K_M", "files": ["a.gguf"]},
                )
                configure_screen = ConfigureScreen()
                await app.push_screen(configure_screen)
                await pilot.pause()
                await pilot.click("#submit-config")
                await pilot.pause()
                self.assertEqual(app.state.warnings, [])
                self.assertEqual(
                    configure_screen.query_one("#preflight-warning", Static).content, ""
                )
                self.assertEqual(app.screen.STEP, "name")

    async def test_ctx_and_ttl_inputs_reject_non_numeric_characters(self):
        # Regression guard for Issue 1: typo'd values like "32,768" must
        # never be typeable into these fields, since they flow unmodified
        # into build_server_args() at actual llama-server launch time.
        fake_defaults = {
            "device": "", "split_mode": "layer", "tensor_split": "3,1",
            "ctx": 32768, "kv_quant": "q8_0", "flash_attn": "auto",
            "ttl": 3600, "mtp": "off",
        }
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.load_defaults", return_value=fake_defaults):
            async with app.run_test() as pilot:
                app.state = WizardState()
                await app.push_screen(ConfigureScreen())
                await pilot.pause()

                ctx_input = app.screen.query_one("#config-ctx", Input)
                self.assertEqual(ctx_input.type, "integer")
                ctx_input.focus()
                # select_on_focus selects the whole value asynchronously;
                # give it a pause to land before overriding cursor position,
                # otherwise the next keypress replaces the selected text
                # instead of appending at the end.
                await pilot.pause()
                ctx_input.cursor_position = len(ctx_input.value)
                await pilot.pause()
                await pilot.press(",")
                await pilot.press("x")
                await pilot.pause()
                self.assertEqual(ctx_input.value, "32768")
                await pilot.press("5")
                await pilot.pause()
                self.assertEqual(ctx_input.value, "327685")

                ttl_input = app.screen.query_one("#config-ttl", Input)
                self.assertEqual(ttl_input.type, "integer")
                ttl_input.focus()
                await pilot.pause()
                ttl_input.cursor_position = len(ttl_input.value)
                await pilot.pause()
                await pilot.press(".")
                await pilot.press("a")
                await pilot.pause()
                self.assertEqual(ttl_input.value, "3600")


class TestNameScreen(unittest.IsolatedAsyncioTestCase):
    async def test_prefills_name_from_quant_label_and_default_dest_dir(self):
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.next_unique_profile_name", side_effect=lambda s: s):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"]},
                )
                await app.push_screen(NameScreen())
                await pilot.pause()
                name_input = app.screen.query_one("#profile-name", Input)
                self.assertEqual(name_input.value, "model")
                dest_input = app.screen.query_one("#dest-dir", Input)
                self.assertEqual(dest_input.value, str(modelctl.DEFAULT_MODELS_DIR))

    async def test_submit_stores_name_and_dest_dir_and_advances(self):
        # DownloadScreen.on_mount is stubbed out even though this test is
        # about NameScreen, not DownloadScreen: submitting advances INTO
        # DownloadScreen, whose real on_mount (added in Task 9) immediately
        # fires a background download_if_needed() worker. A same-value mock
        # of download_if_needed would resolve near-instantly and race with
        # the assertion below (the worker can finish and advance to
        # "summary" within the same pilot.pause()), so instead the screen's
        # own on_mount is replaced with a no-op -- this test only cares that
        # NameScreen pushed DownloadScreen, not what DownloadScreen does
        # once mounted (that's TestDownloadScreen's job).
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.next_unique_profile_name", side_effect=lambda s: s), \
             mock.patch("modelctl_tui.DownloadScreen.on_mount", lambda self: None):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"]},
                )
                await app.push_screen(NameScreen())
                await pilot.pause()
                await pilot.click("#submit-name")
                await pilot.pause()
                self.assertEqual(app.state.profile_name, "model")
                self.assertEqual(app.state.dest_dir, str(modelctl.DEFAULT_MODELS_DIR))
                self.assertEqual(app.screen.STEP, "download")

    async def test_submit_strips_leading_and_trailing_whitespace(self):
        # See test_submit_stores_name_and_dest_dir_and_advances for why
        # DownloadScreen.on_mount is stubbed out here too.
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.next_unique_profile_name", side_effect=lambda s: s), \
             mock.patch("modelctl_tui.DownloadScreen.on_mount", lambda self: None):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"]},
                )
                await app.push_screen(NameScreen())
                await pilot.pause()
                name_input = app.screen.query_one("#profile-name", Input)
                name_input.value = "  padded-name  "
                dest_input = app.screen.query_one("#dest-dir", Input)
                dest_input.value = "  /tmp/padded-dest  "
                await pilot.click("#submit-name")
                await pilot.pause()
                self.assertEqual(app.state.profile_name, "padded-name")
                self.assertEqual(app.state.dest_dir, "/tmp/padded-dest")

    async def test_blank_name_and_dest_dir_fall_back_to_computed_defaults(self):
        # Mirrors cmd_pull's `input(...).strip() or name_default` fallback:
        # a user who clears the prefilled fields and clicks Continue must
        # get the computed defaults back, not empty strings. An empty
        # profile_name would make save_profile() write to
        # PROFILES_DIR / ".json", and an empty dest_dir would silently
        # resolve to the current working directory via Path("").
        # DownloadScreen.on_mount is stubbed out for the same reason as the
        # two tests above -- submitting advances into DownloadScreen's real
        # worker, which isn't this test's concern.
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.next_unique_profile_name", side_effect=lambda s: s), \
             mock.patch("modelctl_tui.DownloadScreen.on_mount", lambda self: None):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"]},
                    dest_dir="/some/dest",
                )
                await app.push_screen(NameScreen())
                await pilot.pause()
                name_input = app.screen.query_one("#profile-name", Input)
                name_input.value = "   "
                dest_input = app.screen.query_one("#dest-dir", Input)
                dest_input.value = "   "
                await pilot.click("#submit-name")
                await pilot.pause()
                self.assertEqual(app.state.profile_name, "model")
                self.assertEqual(app.state.dest_dir, "/some/dest")


class TestDownloadScreen(unittest.IsolatedAsyncioTestCase):
    async def test_downloads_model_file_and_advances_on_success(self):
        # SummaryScreen.on_mount is stubbed out even though this test is
        # about DownloadScreen, not SummaryScreen: succeeding advances INTO
        # SummaryScreen, whose real on_mount (Task 10) immediately fires a
        # background save/generate/sync worker against real modelctl.py
        # functions. This test only cares that DownloadScreen pushed
        # SummaryScreen, not what SummaryScreen does once mounted (that's
        # TestSummaryScreen's job) -- mirrors how NameScreen's tests stub
        # DownloadScreen.on_mount for the same reason.
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.download_if_needed", return_value="/models/model-Q4_K_M.gguf") as mock_dl, \
             mock.patch("modelctl_tui.SummaryScreen.on_mount", lambda self: None):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"], "sharded": False},
                    dest_dir="/models",
                )
                await app.push_screen(DownloadScreen())
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                mock_dl.assert_called_once_with("repo/x", "model-Q4_K_M.gguf", modelctl.Path("/models"))
                self.assertEqual(app.state.__dict__.get("model_path"), "/models/model-Q4_K_M.gguf")
                self.assertEqual(app.screen.STEP, "summary")

    async def test_download_failure_shows_retry_and_does_not_advance(self):
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.download_if_needed", side_effect=OSError("network drop")):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"], "sharded": False},
                    dest_dir="/models",
                )
                await app.push_screen(DownloadScreen())
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertEqual(app.screen.STEP, "download")
                retry_button = app.screen.query_one("#retry-download", Button)
                self.assertFalse(retry_button.disabled)

    async def test_sharded_quant_group_downloads_each_shard_in_order(self):
        # SummaryScreen.on_mount stubbed out -- see comment on
        # test_downloads_model_file_and_advances_on_success above.
        app = PullWizardApp()
        shard1 = "/models/model-Q4_K_M-00001-of-00002.gguf"
        shard2 = "/models/model-Q4_K_M-00002-of-00002.gguf"
        with mock.patch(
            "modelctl_tui.modelctl.download_if_needed",
            side_effect=[shard1, shard2],
        ) as mock_dl, mock.patch("modelctl_tui.SummaryScreen.on_mount", lambda self: None):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={
                        "label": "model-Q4_K_M",
                        "files": [
                            "model-Q4_K_M-00001-of-00002.gguf",
                            "model-Q4_K_M-00002-of-00002.gguf",
                        ],
                        "sharded": True,
                    },
                    dest_dir="/models",
                )
                await app.push_screen(DownloadScreen())
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertEqual(
                    mock_dl.call_args_list,
                    [
                        mock.call("repo/x", "model-Q4_K_M-00001-of-00002.gguf", modelctl.Path("/models")),
                        mock.call("repo/x", "model-Q4_K_M-00002-of-00002.gguf", modelctl.Path("/models")),
                    ],
                )
                # model_path must be the FIRST shard's path, not the last.
                self.assertEqual(app.state.__dict__.get("model_path"), shard1)
                self.assertEqual(app.screen.STEP, "summary")

    async def test_mmproj_and_mtp_choices_get_local_path_set_on_success(self):
        # SummaryScreen.on_mount stubbed out -- see comment on
        # test_downloads_model_file_and_advances_on_success above.
        app = PullWizardApp()
        model_path = "/models/model-Q4_K_M.gguf"
        mmproj_path = "/models/mmproj-F16.gguf"
        mtp_path = "/models/model-mtp.gguf"
        mmproj_choice = {"name": "mmproj-F16.gguf"}
        mtp_choice = {"name": "model-mtp.gguf"}
        with mock.patch(
            "modelctl_tui.modelctl.download_if_needed",
            side_effect=[model_path, mmproj_path, mtp_path],
        ) as mock_dl, mock.patch("modelctl_tui.SummaryScreen.on_mount", lambda self: None):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"], "sharded": False},
                    mmproj_choice=mmproj_choice,
                    mtp_choice=mtp_choice,
                    dest_dir="/models",
                )
                await app.push_screen(DownloadScreen())
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertEqual(
                    mock_dl.call_args_list,
                    [
                        mock.call("repo/x", "model-Q4_K_M.gguf", modelctl.Path("/models")),
                        mock.call("repo/x", "mmproj-F16.gguf", modelctl.Path("/models")),
                        mock.call("repo/x", "model-mtp.gguf", modelctl.Path("/models")),
                    ],
                )
                self.assertEqual(app.state.model_path, model_path)
                self.assertEqual(mmproj_choice.get("local_path"), mmproj_path)
                self.assertEqual(mtp_choice.get("local_path"), mtp_path)
                self.assertEqual(app.screen.STEP, "summary")

    async def test_rapid_double_press_of_retry_does_not_spawn_two_workers(self):
        # Simulates two Button.Pressed messages landing back-to-back, which
        # is what happens if two physical clicks are queued before the first
        # handler's `disabled = True` assignment takes effect (Button only
        # consults `disabled` at click time, not at dispatch time). We call
        # on_button_pressed() twice with zero `await` between the calls --
        # that's the crux of the race: run_download's @work(exclusive=True)
        # cancels the first worker's asyncio Task before the event loop ever
        # gets a chance to hand its thread body to the executor, so the
        # first download attempt never actually starts. If a rapid double
        # click landed further apart in time (e.g. the first worker had
        # already reached the executor thread), exclusive=True would not be
        # able to stop the in-flight thread -- but that's not the race this
        # fix targets: this test asserts the specific "both clicks land
        # before the first thread body starts" case is safe.
        # SummaryScreen.on_mount stubbed out too -- see comment on
        # test_downloads_model_file_and_advances_on_success above.
        app = PullWizardApp()
        calls = []
        with mock.patch(
            "modelctl_tui.modelctl.download_if_needed",
            side_effect=lambda repo_id, name, dest: (calls.append(name), f"/models/{name}")[1],
        ) as mock_dl, mock.patch("modelctl_tui.DownloadScreen.on_mount", lambda self: None), \
             mock.patch("modelctl_tui.SummaryScreen.on_mount", lambda self: None):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "x", "files": ["model.gguf"], "sharded": False},
                    dest_dir="/models",
                )
                screen = DownloadScreen()
                await app.push_screen(screen)
                retry_button = screen.query_one("#retry-download", Button)
                press_event = Button.Pressed(retry_button)
                screen.on_button_pressed(press_event)
                screen.on_button_pressed(press_event)
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                mock_dl.assert_called_once_with("repo/x", "model.gguf", modelctl.Path("/models"))
                self.assertEqual(calls, ["model.gguf"])
                summary_screens = [s for s in app.screen_stack if type(s).__name__ == "SummaryScreen"]
                self.assertEqual(len(summary_screens), 1)


class TestSummaryScreen(unittest.IsolatedAsyncioTestCase):
    async def test_saves_generates_and_syncs_on_mount(self):
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.save_profile") as mock_save, \
             mock.patch("modelctl_tui.modelctl.generate_artifacts", return_value=True) as mock_gen, \
             mock.patch("modelctl_tui.modelctl.sync_all_backends") as mock_sync, \
             mock.patch("modelctl_tui.modelctl.sync_hermes_custom_providers") as mock_hermes, \
             mock.patch("modelctl_tui.modelctl.capture_env_passthrough", return_value=[]):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"]},
                    profile_name="model",
                    model_path="/models/model-Q4_K_M.gguf",
                    config={"ctx": "32768"},
                )
                await app.push_screen(SummaryScreen())
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                mock_save.assert_called_once()
                mock_gen.assert_called_once()
                mock_sync.assert_called_once()
                mock_hermes.assert_called_once()
                status = app.screen.query_one("#summary-status", Static)
                self.assertIn("model", str(status.render()))

    async def test_shows_warnings_if_any_were_collected(self):
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.save_profile"), \
             mock.patch("modelctl_tui.modelctl.generate_artifacts", return_value=True), \
             mock.patch("modelctl_tui.modelctl.sync_all_backends"), \
             mock.patch("modelctl_tui.modelctl.sync_hermes_custom_providers"), \
             mock.patch("modelctl_tui.modelctl.capture_env_passthrough", return_value=[]):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"]},
                    profile_name="model",
                    model_path="/models/model-Q4_K_M.gguf",
                    config={"ctx": "32768"},
                    warnings=["ERROR: llama-server not found"],
                )
                await app.push_screen(SummaryScreen())
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                status = app.screen.query_one("#summary-status", Static)
                self.assertIn("ERROR: llama-server not found", str(status.render()))

    async def test_profile_dict_matches_cmd_pull_shape(self):
        # Regression guard for the profile assembly itself: same field set
        # cmd_pull builds (name, repo_id, file, model_path, mmproj_path,
        # mtp_path, config, env), with mmproj/mtp local_path pulled from
        # DownloadScreen's mmproj_choice/mtp_choice (Task 9), not the bare
        # repo filename.
        app = PullWizardApp()
        captured = {}

        def capture_save(profile):
            captured.update(profile)

        with mock.patch("modelctl_tui.modelctl.save_profile", side_effect=capture_save), \
             mock.patch("modelctl_tui.modelctl.generate_artifacts", return_value=True), \
             mock.patch("modelctl_tui.modelctl.sync_all_backends"), \
             mock.patch("modelctl_tui.modelctl.sync_hermes_custom_providers"), \
             mock.patch("modelctl_tui.modelctl.capture_env_passthrough", return_value=["FOO=bar"]):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"]},
                    mmproj_choice={"name": "mmproj-F16.gguf", "local_path": "/models/mmproj-F16.gguf"},
                    mtp_choice={"name": "model-mtp.gguf", "local_path": "/models/model-mtp.gguf"},
                    profile_name="model",
                    model_path="/models/model-Q4_K_M.gguf",
                    config={"ctx": "32768"},
                )
                await app.push_screen(SummaryScreen())
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertEqual(
                    captured,
                    {
                        "name": "model",
                        "repo_id": "repo/x",
                        "file": "model-Q4_K_M",
                        "model_path": "/models/model-Q4_K_M.gguf",
                        "mmproj_path": "/models/mmproj-F16.gguf",
                        "mtp_path": "/models/model-mtp.gguf",
                        "config": {"ctx": "32768"},
                        "env": ["FOO=bar"],
                    },
                )

    async def test_mmproj_and_mtp_paths_are_none_when_choices_skipped(self):
        # mmproj_choice/mtp_choice default to None on WizardState when the
        # user skipped VisionMtpScreen entirely -- (state.mmproj_choice or
        # {}).get("local_path") must yield None, not raise, in that case.
        app = PullWizardApp()
        captured = {}

        def capture_save(profile):
            captured.update(profile)

        with mock.patch("modelctl_tui.modelctl.save_profile", side_effect=capture_save), \
             mock.patch("modelctl_tui.modelctl.generate_artifacts", return_value=True), \
             mock.patch("modelctl_tui.modelctl.sync_all_backends"), \
             mock.patch("modelctl_tui.modelctl.sync_hermes_custom_providers"), \
             mock.patch("modelctl_tui.modelctl.capture_env_passthrough", return_value=[]):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"]},
                    profile_name="model",
                    model_path="/models/model-Q4_K_M.gguf",
                    config={"ctx": "32768"},
                )
                await app.push_screen(SummaryScreen())
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertIsNone(captured["mmproj_path"])
                self.assertIsNone(captured["mtp_path"])

    async def test_done_button_exits_app(self):
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.save_profile"), \
             mock.patch("modelctl_tui.modelctl.generate_artifacts", return_value=True), \
             mock.patch("modelctl_tui.modelctl.sync_all_backends"), \
             mock.patch("modelctl_tui.modelctl.sync_hermes_custom_providers"), \
             mock.patch("modelctl_tui.modelctl.capture_env_passthrough", return_value=[]):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"]},
                    profile_name="model",
                    model_path="/models/model-Q4_K_M.gguf",
                    config={"ctx": "32768"},
                )
                await app.push_screen(SummaryScreen())
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                await pilot.click("#done-button")
                await pilot.pause()
                self.assertFalse(app.is_running)

    async def test_exception_during_sync_shows_failure_and_does_not_crash_app(self):
        # Critical issue 1: run_sync() previously had zero error handling,
        # so any exception (disk full, permissions, network hiccup during
        # Hermes sync) propagated unhandled out of the worker thread. With
        # Textual's default exit_on_error=True, that crashes the whole app
        # (app._handle_exception() exits, tears down the alt-screen, dumps
        # a raw traceback) instead of showing the user an honest message.
        # This test simulates save_profile itself failing (OSError("disk
        # full")) -- i.e. nothing was actually saved -- and asserts the app
        # survives with a visible failure message on #summary-status.
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.save_profile", side_effect=OSError("disk full")), \
             mock.patch("modelctl_tui.modelctl.generate_artifacts", return_value=True), \
             mock.patch("modelctl_tui.modelctl.sync_all_backends"), \
             mock.patch("modelctl_tui.modelctl.sync_hermes_custom_providers"), \
             mock.patch("modelctl_tui.modelctl.capture_env_passthrough", return_value=[]):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"]},
                    profile_name="model",
                    model_path="/models/model-Q4_K_M.gguf",
                    config={"ctx": "32768"},
                )
                await app.push_screen(SummaryScreen())
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertTrue(app.is_running)
                status = app.screen.query_one("#summary-status", Static)
                rendered = str(status.render())
                self.assertIn("disk full", rendered)
                # Done must still work after a failure -- re-running the
                # wizard is the recovery path, but the user must be able to
                # exit cleanly rather than getting stuck on this screen.
                await pilot.click("#done-button")
                await pilot.pause()
                self.assertFalse(app.is_running)

    async def test_stderr_warnings_during_sync_reach_summary_status(self):
        # Critical issue 2: restart_router_service() (called transitively
        # via sync_all_backends -> sync_router_preset) reports failures by
        # printing to stderr and returning False -- it never raises. While
        # the Textual app is running, stdout/stderr are globally redirected
        # by the framework, so nothing previously captured that output and
        # the user saw a clean "Saved profile" message with zero indication
        # the router preset was written but never actually reloaded. This
        # simulates that by making sync_all_backends print a warning to
        # stderr as a side effect, and asserts it shows up in
        # #summary-status instead of silently vanishing.
        import sys as _sys

        def fake_sync_all_backends(*args, **kwargs):
            print("WARNING: couldn't restart llama-router: unit not found", file=_sys.stderr)

        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.save_profile"), \
             mock.patch("modelctl_tui.modelctl.generate_artifacts", return_value=True), \
             mock.patch("modelctl_tui.modelctl.sync_all_backends", side_effect=fake_sync_all_backends), \
             mock.patch("modelctl_tui.modelctl.sync_hermes_custom_providers"), \
             mock.patch("modelctl_tui.modelctl.capture_env_passthrough", return_value=[]):
            async with app.run_test() as pilot:
                app.state = WizardState(
                    repo_id="repo/x",
                    quant_group={"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"]},
                    profile_name="model",
                    model_path="/models/model-Q4_K_M.gguf",
                    config={"ctx": "32768"},
                )
                await app.push_screen(SummaryScreen())
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                status = app.screen.query_one("#summary-status", Static)
                rendered = str(status.render())
                self.assertIn("couldn't restart llama-router", rendered)


class TestFullWizardFlow(unittest.IsolatedAsyncioTestCase):
    """End-to-end Pilot-driven test walking the complete wizard chain:
    search -> quant -> configure (vision_mtp skipped, repo has neither
    mmproj nor mtp files) -> name -> download -> summary. Every
    modelctl.py function that would otherwise do real network/filesystem/
    subprocess work is mocked; the test proves the seven screens actually
    hand off WizardState to each other correctly, which the individual
    per-screen tests elsewhere in this file can't catch on their own."""

    async def test_search_to_summary_no_extras(self):
        fake_results = [{
            "repo_id": "repo/x", "downloads": 1, "likes": 1, "is_gguf": True, "has_mtp": False,
            "contents": {
                "quant_groups": [{"label": "model-Q4_K_M", "files": ["model-Q4_K_M.gguf"], "sharded": False, "total_size": 100}],
                "mmproj_files": [], "mtp_files": [],
            },
        }]
        fake_defaults = {
            "device": "", "split_mode": "layer", "tensor_split": "3,1",
            "ctx": 32768, "kv_quant": "q8_0", "flash_attn": "auto", "ttl": 3600, "mtp": "off",
        }
        app = PullWizardApp()
        with mock.patch("modelctl_tui.modelctl.search_models", return_value=fake_results), \
             mock.patch("modelctl_tui.modelctl.load_defaults", return_value=fake_defaults), \
             mock.patch("modelctl_tui.modelctl.preflight", return_value=(True, "llama-server", {}, [])), \
             mock.patch("modelctl_tui.modelctl.next_unique_profile_name", side_effect=lambda s: s), \
             mock.patch("modelctl_tui.modelctl.download_if_needed", return_value="/models/model-Q4_K_M.gguf"), \
             mock.patch("modelctl_tui.modelctl.capture_env_passthrough", return_value=[]), \
             mock.patch("modelctl_tui.modelctl.save_profile") as mock_save, \
             mock.patch("modelctl_tui.modelctl.generate_artifacts", return_value=True), \
             mock.patch("modelctl_tui.modelctl.sync_all_backends"), \
             mock.patch("modelctl_tui.modelctl.sync_hermes_custom_providers"):
            async with app.run_test() as pilot:
                # search -> quant. SearchScreen runs search_models() in a
                # background worker (Task 4), so wait_for_complete() is
                # required after submitting the query -- pilot.pause() alone
                # is not enough to let the worker thread finish and the
                # call_from_thread callback populate #search-results.
                await pilot.click("#search-input")
                await pilot.press(*"x", "enter")
                await app.workers.wait_for_complete()
                await pilot.pause()
                await pilot.click("ListItem")
                await pilot.pause()
                self.assertEqual(app.screen.STEP, "quant")

                # quant -> configure (skips vision_mtp: repo has neither
                # mmproj nor mtp files). QuantPickScreen does no I/O, so a
                # plain pilot.pause() is enough here.
                await pilot.click("ListItem")
                await pilot.pause()
                self.assertEqual(app.screen.STEP, "configure")

                # configure -> name. ConfigureScreen's preflight() call is
                # synchronous (Task 7), so no worker wait is needed.
                await pilot.click("#submit-config")
                await pilot.pause()
                self.assertEqual(app.screen.STEP, "name")

                # name -> download -> summary. Clicking #submit-name pushes
                # DownloadScreen, whose on_mount immediately fires a
                # background run_download() worker (Task 9). Once that
                # worker finishes, its success callback pushes SummaryScreen,
                # whose own on_mount immediately fires a second background
                # run_sync() worker (Task 10) that does the actual
                # save_profile/generate_artifacts/sync_all_backends/
                # sync_hermes_custom_providers sequence -- so two rounds of
                # wait_for_complete()+pause() are needed, one per worker,
                # before it's safe to assert against the mocked save_profile.
                await pilot.click("#submit-name")
                await pilot.pause()
                await app.workers.wait_for_complete()
                await pilot.pause()
                self.assertEqual(app.screen.STEP, "summary")
                await app.workers.wait_for_complete()
                await pilot.pause()

                mock_save.assert_called_once()
                saved_profile = mock_save.call_args[0][0]
                self.assertEqual(saved_profile["repo_id"], "repo/x")
                self.assertEqual(saved_profile["model_path"], "/models/model-Q4_K_M.gguf")


if __name__ == "__main__":
    unittest.main()
