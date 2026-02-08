import unittest

from hypermindlabs import database_router as dr


class _FakeConnection:
    def close(self):
        return None


class TestDatabaseRouter(unittest.TestCase):
    def setUp(self):
        self._original_connect = dr.psycopg.connect

    def tearDown(self):
        dr.psycopg.connect = self._original_connect

    def test_primary_selected_when_primary_available(self):
        calls = []

        def fake_connect(*, conninfo, connect_timeout):
            calls.append(conninfo)
            return _FakeConnection()

        dr.psycopg.connect = fake_connect

        router = dr.DatabaseRouter(
            primary_database={
                "db_name": "main",
                "user": "u",
                "password": "p",
                "host": "host1",
                "port": "5432",
            },
            fallback_database={
                "enabled": True,
                "db_name": "fallback",
                "user": "u",
                "password": "p",
                "host": "host2",
                "port": "5433",
            },
        )
        route = router.resolve()

        self.assertEqual(route.status, "primary")
        self.assertEqual(route.active_target, "primary")
        self.assertTrue(route.primary_available)
        self.assertEqual(len(calls), 1)

    def test_fallback_selected_when_primary_unavailable(self):
        def fake_connect(*, conninfo, connect_timeout):
            if "host=host1" in conninfo:
                raise RuntimeError("primary down")
            return _FakeConnection()

        dr.psycopg.connect = fake_connect

        router = dr.DatabaseRouter(
            primary_database={
                "db_name": "main",
                "user": "u",
                "password": "p",
                "host": "host1",
                "port": "5432",
            },
            fallback_database={
                "enabled": True,
                "db_name": "fallback",
                "user": "u",
                "password": "p",
                "host": "host2",
                "port": "5433",
            },
            fallback_enabled=True,
        )
        route = router.resolve()

        self.assertEqual(route.status, "fallback")
        self.assertEqual(route.active_target, "fallback")
        self.assertFalse(route.primary_available)
        self.assertTrue(route.fallback_available)
        self.assertIn("primary:", route.errors[0])

    def test_failed_all_when_no_targets_available(self):
        def fake_connect(*, conninfo, connect_timeout):
            raise RuntimeError("all down")

        dr.psycopg.connect = fake_connect

        router = dr.DatabaseRouter(
            primary_database={
                "db_name": "main",
                "user": "u",
                "password": "p",
                "host": "host1",
                "port": "5432",
            },
            fallback_database={
                "enabled": True,
                "db_name": "fallback",
                "user": "u",
                "password": "p",
                "host": "host2",
                "port": "5433",
            },
            fallback_enabled=True,
        )
        route = router.resolve()

        self.assertEqual(route.status, "failed_all")
        self.assertEqual(route.active_target, "primary")
        self.assertFalse(route.primary_available)
        self.assertFalse(route.fallback_available)
        self.assertTrue(len(route.errors) >= 2)

    def test_fallback_not_used_when_disabled(self):
        def fake_connect(*, conninfo, connect_timeout):
            if "host=host1" in conninfo:
                raise RuntimeError("primary down")
            return _FakeConnection()

        dr.psycopg.connect = fake_connect

        router = dr.DatabaseRouter(
            primary_database={
                "db_name": "main",
                "user": "u",
                "password": "p",
                "host": "host1",
                "port": "5432",
            },
            fallback_database={
                "enabled": False,
                "db_name": "fallback",
                "user": "u",
                "password": "p",
                "host": "host2",
                "port": "5433",
            },
            fallback_enabled=False,
        )
        route = router.resolve()

        self.assertEqual(route.status, "failed_all")
        self.assertEqual(route.active_target, "primary")
        self.assertFalse(route.fallback_available)


if __name__ == "__main__":
    unittest.main()
