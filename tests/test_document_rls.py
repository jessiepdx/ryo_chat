from pathlib import Path
import unittest


MIGRATION_PATH = Path(__file__).resolve().parent.parent / "db" / "migrations" / "095_document_rls_policies.sql"


class DocumentRlsMigrationTests(unittest.TestCase):
    def test_migration_enables_rls_on_all_document_tables(self):
        sql = MIGRATION_PATH.read_text(encoding="utf-8")
        for table in (
            "document_sources",
            "document_versions",
            "document_nodes",
            "document_chunks",
            "document_retrieval_events",
        ):
            self.assertIn(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;", sql)

    def test_migration_defines_scope_policy_keys(self):
        sql = MIGRATION_PATH.read_text(encoding="utf-8")
        expected_settings = (
            "app.scope_bypass",
            "app.owner_member_id",
            "app.chat_host_id",
            "app.chat_type",
            "app.community_id",
            "app.topic_id",
            "app.platform",
        )
        for setting_key in expected_settings:
            self.assertIn(setting_key, sql)

    def test_migration_has_per_table_policy(self):
        sql = MIGRATION_PATH.read_text(encoding="utf-8")
        self.assertIn("CREATE POLICY document_sources_scope_policy ON document_sources", sql)
        self.assertIn("CREATE POLICY document_versions_scope_policy ON document_versions", sql)
        self.assertIn("CREATE POLICY document_nodes_scope_policy ON document_nodes", sql)
        self.assertIn("CREATE POLICY document_chunks_scope_policy ON document_chunks", sql)
        self.assertIn(
            "CREATE POLICY document_retrieval_events_scope_policy ON document_retrieval_events",
            sql,
        )


if __name__ == "__main__":
    unittest.main()
