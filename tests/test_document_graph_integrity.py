import unittest

from hypermindlabs.document_graph import repair_document_graph


class DocumentGraphIntegrityTests(unittest.TestCase):
    def test_repair_handles_orphans_cycles_and_sibling_ordering(self):
        malformed_nodes = [
            {"node_key": "root", "node_type": "document", "ordinal": 0, "path": "/document"},
            {"node_key": "a", "node_type": "section", "parent_node_key": "root", "ordinal": 4},
            {"node_key": "b", "node_type": "paragraph", "parent_node_key": "missing", "ordinal": 0},
            {"node_key": "c", "node_type": "paragraph", "parent_node_key": "d", "ordinal": 0},
            {"node_key": "d", "node_type": "paragraph", "parent_node_key": "c", "ordinal": 0},
        ]

        repaired = repair_document_graph(malformed_nodes, root_node_key="root")
        diagnostics = dict(repaired.get("diagnostics") or {})
        nodes = list(repaired.get("nodes") or [])
        edges = list(repaired.get("edges") or [])

        self.assertTrue(bool(diagnostics.get("was_repaired")))
        self.assertIn("b", diagnostics.get("orphan_node_keys") or [])
        self.assertTrue(bool(diagnostics.get("cycle_node_keys")))

        by_key = {str(node.get("node_key")): node for node in nodes}
        self.assertEqual(by_key["b"].get("parent_node_key"), "root")
        self.assertTrue(
            by_key["c"].get("parent_node_key") == "root"
            or by_key["d"].get("parent_node_key") == "root"
        )

        ordinals_by_parent: dict[str, list[int]] = {}
        for node in nodes:
            node_key = str(node.get("node_key") or "")
            parent_node_key = str(node.get("parent_node_key") or "")
            if not node_key or not parent_node_key:
                continue
            ordinals_by_parent.setdefault(parent_node_key, []).append(int(node.get("ordinal", 0) or 0))
        for ordinals in ordinals_by_parent.values():
            self.assertEqual(sorted(ordinals), list(range(len(ordinals))))

        parent_edges = [edge for edge in edges if str(edge.get("edge_type")) == "parent_child"]
        self.assertEqual(len(parent_edges), len(nodes) - 1)

    def test_repair_is_deterministic_after_first_repair(self):
        nodes = [
            {"node_key": "root", "node_type": "document", "ordinal": 0, "path": "/document"},
            {"node_key": "a", "node_type": "section", "parent_node_key": "root", "ordinal": 2},
            {"node_key": "b", "node_type": "paragraph", "parent_node_key": "root", "ordinal": 0},
        ]

        first = repair_document_graph(nodes, root_node_key="root")
        second = repair_document_graph(first.get("nodes"), root_node_key="root")

        self.assertEqual(first.get("nodes"), second.get("nodes"))
        self.assertEqual(first.get("edges"), second.get("edges"))


if __name__ == "__main__":
    unittest.main()
