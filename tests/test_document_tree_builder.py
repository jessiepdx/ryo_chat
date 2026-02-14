import unittest

from hypermindlabs.document_tree_builder import build_canonical_document_tree


def _canonical_payload() -> dict:
    return {
        "schema_version": 1,
        "canonical_schema": "document.parse.canonical.v1",
        "status": "parsed",
        "content_text": "Introduction\n\nParagraph body\n\n- item one\n\nDetails\n\nR1 | C1\n\nprint('x')\n\n[1] Footnote",
        "sections": [
            {
                "section_id": "s1",
                "title": "Introduction",
                "level": 1,
                "text": "Introduction",
                "start_char": 0,
                "end_char": 12,
                "page_start": 1,
                "page_end": 1,
                "metadata": {"element_type": "heading"},
            },
            {
                "section_id": "s2",
                "title": "",
                "level": 2,
                "text": "Paragraph body",
                "start_char": 14,
                "end_char": 28,
                "page_start": 1,
                "page_end": 1,
                "metadata": {"element_type": "paragraph"},
            },
            {
                "section_id": "s3",
                "title": "",
                "level": 2,
                "text": "- item one",
                "start_char": 30,
                "end_char": 40,
                "page_start": 1,
                "page_end": 1,
                "metadata": {"element_type": "list"},
            },
            {
                "section_id": "s4",
                "title": "Details",
                "level": 2,
                "text": "Details",
                "start_char": 42,
                "end_char": 49,
                "page_start": 1,
                "page_end": 1,
                "metadata": {"element_type": "heading"},
            },
            {
                "section_id": "s5",
                "title": "",
                "level": 2,
                "text": "R1 | C1",
                "start_char": 51,
                "end_char": 58,
                "page_start": 1,
                "page_end": 1,
                "metadata": {"element_type": "table"},
            },
            {
                "section_id": "s6",
                "title": "",
                "level": 2,
                "text": "print('x')",
                "start_char": 60,
                "end_char": 70,
                "page_start": 1,
                "page_end": 1,
                "metadata": {"element_type": "code"},
            },
            {
                "section_id": "s7",
                "title": "",
                "level": 2,
                "text": "[1] Footnote",
                "start_char": 72,
                "end_char": 84,
                "page_start": 1,
                "page_end": 1,
                "metadata": {"element_type": "footnote"},
            },
        ],
        "metadata": {"fixture": "tree-builder"},
        "warnings": [],
        "errors": [],
        "provenance": {
            "selected_adapter": "unit",
            "selected_adapter_version": "1.0.0",
            "adapter_chain": ["unit"],
            "confidence": 0.9,
            "cost": 0.1,
            "duration_ms": 5,
            "route_debug": {},
            "profile_summary": {},
        },
    }


class DocumentTreeBuilderTests(unittest.TestCase):
    def test_tree_builder_generates_expected_taxonomy_and_parentage(self):
        tree = build_canonical_document_tree(_canonical_payload(), document_version_id=101)
        nodes = list(tree.get("nodes") or [])
        edges = list(tree.get("edges") or [])
        diagnostics = dict(tree.get("integrity") or {})

        self.assertGreaterEqual(len(nodes), 8)
        self.assertGreaterEqual(len(edges), len(nodes) - 1)
        self.assertFalse(bool(diagnostics.get("was_repaired")))

        node_types = {str(node.get("node_type")) for node in nodes}
        self.assertIn("document", node_types)
        self.assertIn("section", node_types)
        self.assertIn("subsection", node_types)
        self.assertIn("paragraph", node_types)
        self.assertIn("list", node_types)
        self.assertIn("table", node_types)
        self.assertIn("code", node_types)
        self.assertIn("footnote", node_types)

        by_section_id: dict[str, dict] = {}
        for node in nodes:
            metadata = node.get("node_metadata")
            if not isinstance(metadata, dict):
                continue
            section_id = str(metadata.get("section_id") or "").strip()
            if section_id:
                by_section_id[section_id] = node

        intro_node = by_section_id["s1"]
        subsection_node = by_section_id["s4"]
        paragraph_node = by_section_id["s2"]
        list_node = by_section_id["s3"]
        table_node = by_section_id["s5"]
        code_node = by_section_id["s6"]
        footnote_node = by_section_id["s7"]

        self.assertEqual(paragraph_node.get("parent_node_key"), intro_node.get("node_key"))
        self.assertEqual(list_node.get("parent_node_key"), intro_node.get("node_key"))
        self.assertEqual(subsection_node.get("parent_node_key"), intro_node.get("node_key"))
        self.assertEqual(table_node.get("parent_node_key"), subsection_node.get("node_key"))
        self.assertEqual(code_node.get("parent_node_key"), subsection_node.get("node_key"))
        self.assertEqual(footnote_node.get("parent_node_key"), subsection_node.get("node_key"))

    def test_tree_builder_is_deterministic_for_same_canonical_input(self):
        first = build_canonical_document_tree(_canonical_payload(), document_version_id=202)
        second = build_canonical_document_tree(_canonical_payload(), document_version_id=202)

        first_nodes = [str(node.get("node_key")) for node in list(first.get("nodes") or [])]
        second_nodes = [str(node.get("node_key")) for node in list(second.get("nodes") or [])]
        self.assertEqual(first_nodes, second_nodes)

        first_edges = [
            (
                str(edge.get("source_node_key")),
                str(edge.get("target_node_key")),
                str(edge.get("edge_type")),
                int(edge.get("ordinal", 0) or 0),
            )
            for edge in list(first.get("edges") or [])
        ]
        second_edges = [
            (
                str(edge.get("source_node_key")),
                str(edge.get("target_node_key")),
                str(edge.get("edge_type")),
                int(edge.get("ordinal", 0) or 0),
            )
            for edge in list(second.get("edges") or [])
        ]
        self.assertEqual(first_edges, second_edges)


if __name__ == "__main__":
    unittest.main()
