import unittest

from hypermindlabs.run_mode_handlers import normalize_run_mode, run_modes_manifest


class RunModeHandlersTests(unittest.TestCase):
    def test_normalize_run_mode_defaults_to_chat(self):
        self.assertEqual(normalize_run_mode(None), "chat")
        self.assertEqual(normalize_run_mode("unknown_mode"), "chat")

    def test_manifest_includes_expected_modes(self):
        modes = run_modes_manifest()
        ids = [item.get("id") for item in modes]
        self.assertEqual(ids, ["chat", "workflow", "batch", "compare", "replay"])


if __name__ == "__main__":
    unittest.main()
