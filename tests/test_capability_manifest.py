import unittest
from unittest import mock

from hypermindlabs.capability_manifest import build_capability_manifest, find_capability


class CapabilityManifestTests(unittest.TestCase):
    def test_build_manifest_contains_expected_capabilities(self):
        fake_config = mock.Mock()
        fake_config.inference = {
            "chat": {"model": "llama3.2:latest"},
            "tool": {"model": "llama3.2:latest"},
            "generate": {"model": "llama3.2:latest"},
            "embedding": {"model": "nomic-embed-text:latest"},
            "multimodal": {"model": "llama3.2-vision:latest"},
        }
        fake_config.runtimeValue = mock.Mock(return_value="http://127.0.0.1:11434")

        with mock.patch("hypermindlabs.capability_manifest.ConfigManager", return_value=fake_config):
            manifest = build_capability_manifest(member_roles=["user"])

        self.assertEqual(manifest["manifest_version"], "1.0.0")
        capability_ids = {item["id"] for item in manifest.get("capabilities", [])}
        self.assertIn("runs.lifecycle", capability_ids)
        self.assertIn("models.list", capability_ids)
        self.assertIn("tools.list", capability_ids)

    def test_find_capability_returns_none_for_missing_id(self):
        fake_manifest = {"capabilities": [{"id": "a"}, {"id": "b"}]}
        self.assertIsNone(find_capability(fake_manifest, "c"))


if __name__ == "__main__":
    unittest.main()
