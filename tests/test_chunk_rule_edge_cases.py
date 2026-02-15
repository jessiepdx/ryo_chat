import unittest
from typing import Any

from hypermindlabs.document_chunker import build_document_chunks


class _Config:
    def __init__(self, values: dict[str, Any] | None = None):
        self._values = dict(values or {})

    def runtimeValue(self, path: str, default: Any) -> Any:  # noqa: N802
        return self._values.get(path, default)


class ChunkRuleEdgeCaseTests(unittest.TestCase):
    def test_empty_tree_generates_no_chunks(self):
        payload = build_document_chunks(
            tree_payload={"nodes": []},
            canonical_output={"sections": []},
            document_version_id=1,
            config_manager=_Config(),
        )
        self.assertEqual(payload["stats"]["chunk_count"], 0)
        self.assertFalse(bool(payload["stats"]["truncated"]))

    def test_max_total_chunks_cap_sets_truncated(self):
        long_text = " ".join(f"word{idx}" for idx in range(1, 500))
        tree_payload = {
            "nodes": [
                {
                    "node_key": "root",
                    "node_type": "document",
                    "parent_node_key": None,
                    "node_title": "Doc",
                    "path": "/document",
                },
                {
                    "node_key": "p1",
                    "node_type": "paragraph",
                    "parent_node_key": "root",
                    "node_title": "",
                    "path": "/document/paragraph:1",
                    "char_start": 0,
                    "char_end": len(long_text),
                    "node_metadata": {
                        "section_id": "s1",
                    },
                },
            ]
        }
        canonical_output = {
            "sections": [
                {
                    "section_id": "s1",
                    "text": long_text,
                }
            ]
        }
        payload = build_document_chunks(
            tree_payload=tree_payload,
            canonical_output=canonical_output,
            document_version_id=2,
            config_manager=_Config(
                {
                    "documents.chunk_target_tokens": 24,
                    "documents.chunk_overlap_tokens": 8,
                    "documents.chunk_max_tokens": 28,
                    "documents.chunk_max_total_chunks": 2,
                }
            ),
        )
        self.assertEqual(payload["stats"]["chunk_count"], 2)
        self.assertTrue(bool(payload["stats"]["truncated"]))

    def test_figure_and_footnote_use_single_chunk_mode(self):
        figure_text = "Figure 3 shows the pipeline graph."
        footnote_text = "[1] Additional details are in appendix."
        tree_payload = {
            "nodes": [
                {"node_key": "root", "node_type": "document", "parent_node_key": None, "path": "/document"},
                {
                    "node_key": "f1",
                    "node_type": "figure",
                    "parent_node_key": "root",
                    "path": "/document/figure:1",
                    "node_metadata": {"section_id": "s1"},
                },
                {
                    "node_key": "n1",
                    "node_type": "footnote",
                    "parent_node_key": "root",
                    "path": "/document/footnote:1",
                    "node_metadata": {"section_id": "s2"},
                },
            ]
        }
        canonical_output = {
            "sections": [
                {"section_id": "s1", "text": figure_text},
                {"section_id": "s2", "text": footnote_text},
            ]
        }

        payload = build_document_chunks(
            tree_payload=tree_payload,
            canonical_output=canonical_output,
            document_version_id=3,
            config_manager=_Config({"documents.chunk_target_tokens": 12}),
        )
        chunks = list(payload.get("chunks") or [])
        figure_chunks = [
            chunk for chunk in chunks
            if str((chunk.get("chunk_metadata") or {}).get("node_type")) == "figure"
        ]
        footnote_chunks = [
            chunk for chunk in chunks
            if str((chunk.get("chunk_metadata") or {}).get("node_type")) == "footnote"
        ]
        self.assertEqual(len(figure_chunks), 1)
        self.assertEqual(len(footnote_chunks), 1)


if __name__ == "__main__":
    unittest.main()
