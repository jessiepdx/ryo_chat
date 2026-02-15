import unittest
from typing import Any

from hypermindlabs.document_chunker import build_document_chunks
from hypermindlabs.document_tree_builder import build_canonical_document_tree


class _Config:
    def __init__(self, values: dict[str, Any] | None = None):
        self._values = dict(values or {})

    def runtimeValue(self, path: str, default: Any) -> Any:  # noqa: N802
        return self._values.get(path, default)


def _canonical_payload() -> dict[str, Any]:
    paragraph_text = " ".join(f"token{i}" for i in range(1, 180))
    list_text = "\n".join(f"- list item {idx}" for idx in range(1, 19))
    table_rows = ["col_a | col_b"] + [f"r{idx}a | r{idx}b" for idx in range(1, 26)]
    table_text = "\n".join(table_rows)
    code_text = "\n".join(f"line_{idx} = {idx}" for idx in range(1, 22))

    return {
        "schema_version": 1,
        "canonical_schema": "document.parse.canonical.v1",
        "status": "parsed",
        "content_text": "\n\n".join(
            [
                "Overview",
                paragraph_text,
                list_text,
                "Data Table",
                table_text,
                "Code Block",
                code_text,
            ]
        ),
        "sections": [
            {
                "section_id": "s1",
                "title": "Overview",
                "level": 1,
                "text": "Overview",
                "start_char": 0,
                "end_char": 8,
                "page_start": 1,
                "page_end": 1,
                "metadata": {"element_type": "heading"},
            },
            {
                "section_id": "s2",
                "title": "",
                "level": 2,
                "text": paragraph_text,
                "start_char": 10,
                "end_char": 10 + len(paragraph_text),
                "page_start": 1,
                "page_end": 1,
                "metadata": {"element_type": "paragraph"},
            },
            {
                "section_id": "s3",
                "title": "",
                "level": 2,
                "text": list_text,
                "start_char": 12 + len(paragraph_text),
                "end_char": 12 + len(paragraph_text) + len(list_text),
                "page_start": 1,
                "page_end": 1,
                "metadata": {"element_type": "list"},
            },
            {
                "section_id": "s4",
                "title": "Data Table",
                "level": 2,
                "text": "Data Table",
                "start_char": 14 + len(paragraph_text) + len(list_text),
                "end_char": 24 + len(paragraph_text) + len(list_text),
                "page_start": 2,
                "page_end": 2,
                "metadata": {"element_type": "heading"},
            },
            {
                "section_id": "s5",
                "title": "",
                "level": 2,
                "text": table_text,
                "start_char": 26 + len(paragraph_text) + len(list_text),
                "end_char": 26 + len(paragraph_text) + len(list_text) + len(table_text),
                "page_start": 2,
                "page_end": 2,
                "metadata": {"element_type": "table"},
            },
            {
                "section_id": "s6",
                "title": "Code Block",
                "level": 2,
                "text": "Code Block",
                "start_char": 28 + len(paragraph_text) + len(list_text) + len(table_text),
                "end_char": 38 + len(paragraph_text) + len(list_text) + len(table_text),
                "page_start": 3,
                "page_end": 3,
                "metadata": {"element_type": "heading"},
            },
            {
                "section_id": "s7",
                "title": "",
                "level": 2,
                "text": code_text,
                "start_char": 40 + len(paragraph_text) + len(list_text) + len(table_text),
                "end_char": 40 + len(paragraph_text) + len(list_text) + len(table_text) + len(code_text),
                "page_start": 3,
                "page_end": 3,
                "metadata": {"element_type": "code"},
            },
        ],
        "metadata": {},
        "warnings": [],
        "errors": [],
        "provenance": {
            "selected_adapter": "unit",
            "selected_adapter_version": "1.0.0",
            "adapter_chain": ["unit"],
            "confidence": 0.9,
            "cost": 0.1,
            "duration_ms": 8,
            "route_debug": {},
            "profile_summary": {},
        },
    }


class DocumentChunkerTests(unittest.TestCase):
    def test_chunker_applies_hierarchical_rules_and_overlap(self):
        canonical = _canonical_payload()
        tree = build_canonical_document_tree(canonical, document_version_id=501)
        config = _Config(
            {
                "documents.chunk_target_tokens": 36,
                "documents.chunk_overlap_tokens": 8,
                "documents.chunk_max_tokens": 48,
                "documents.chunk_max_chunks_per_node": 10,
                "documents.chunk_list_items_per_chunk": 5,
                "documents.chunk_table_rows_per_chunk": 6,
                "documents.chunk_code_lines_per_chunk": 6,
            }
        )

        chunk_payload = build_document_chunks(
            tree_payload=tree,
            canonical_output=canonical,
            document_version_id=501,
            config_manager=config,
        )

        chunks = list(chunk_payload.get("chunks") or [])
        self.assertTrue(chunks)
        self.assertGreater(int(chunk_payload.get("stats", {}).get("chunk_count", 0) or 0), 5)

        paragraph_chunks = [
            chunk for chunk in chunks
            if str((chunk.get("chunk_metadata") or {}).get("node_type")) == "paragraph"
        ]
        self.assertGreaterEqual(len(paragraph_chunks), 2)
        first_tail = set(paragraph_chunks[0]["chunk_text"].split()[-8:])
        second_head = set(paragraph_chunks[1]["chunk_text"].split()[:10])
        self.assertTrue(bool(first_tail.intersection(second_head)))

        table_chunks = [
            chunk for chunk in chunks
            if str((chunk.get("chunk_metadata") or {}).get("node_type")) == "table"
        ]
        self.assertGreaterEqual(len(table_chunks), 2)
        self.assertTrue(table_chunks[1]["chunk_text"].splitlines()[0].startswith("col_a | col_b"))
        self.assertTrue(bool((table_chunks[1].get("chunk_metadata") or {}).get("header_repeated")))

        for chunk in chunks:
            metadata = chunk.get("chunk_metadata") if isinstance(chunk.get("chunk_metadata"), dict) else {}
            self.assertIn("node_path", metadata)
            self.assertIn("heading_trail", metadata)
            self.assertIn("token_count", metadata)

    def test_chunk_keys_are_deterministic_for_same_tree(self):
        canonical = _canonical_payload()
        tree = build_canonical_document_tree(canonical, document_version_id=777)
        config = _Config({"documents.chunk_target_tokens": 30, "documents.chunk_overlap_tokens": 6})

        first = build_document_chunks(
            tree_payload=tree,
            canonical_output=canonical,
            document_version_id=777,
            config_manager=config,
        )
        second = build_document_chunks(
            tree_payload=tree,
            canonical_output=canonical,
            document_version_id=777,
            config_manager=config,
        )

        first_keys = [str(chunk.get("chunk_key")) for chunk in list(first.get("chunks") or [])]
        second_keys = [str(chunk.get("chunk_key")) for chunk in list(second.get("chunks") or [])]
        self.assertEqual(first_keys, second_keys)


if __name__ == "__main__":
    unittest.main()
