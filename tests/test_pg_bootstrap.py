import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


_BOOTSTRAP_PATH = Path(__file__).resolve().parents[1] / "scripts" / "bootstrap_postgres.py"
_BOOTSTRAP_SPEC = importlib.util.spec_from_file_location("bootstrap_postgres", _BOOTSTRAP_PATH)
bootstrap_postgres = importlib.util.module_from_spec(_BOOTSTRAP_SPEC)
assert _BOOTSTRAP_SPEC.loader is not None
sys.modules[_BOOTSTRAP_SPEC.name] = bootstrap_postgres
_BOOTSTRAP_SPEC.loader.exec_module(bootstrap_postgres)


class TestPgBootstrap(unittest.TestCase):
    def test_load_targets_from_config_primary_and_fallback(self):
        payload = {
            "database": {
                "db_name": "main_db",
                "user": "main_user",
                "password": "main_pass",
                "host": "127.0.0.1",
                "port": "5432",
            },
            "database_fallback": {
                "enabled": True,
                "db_name": "fallback_db",
                "user": "fallback_user",
                "password": "fallback_pass",
                "host": "127.0.0.1",
                "port": "5433",
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            targets = bootstrap_postgres.load_targets_from_config(config_path, target_mode="both", maintenance_db="postgres")

        self.assertEqual(len(targets), 2)
        self.assertEqual(targets[0].label, "primary")
        self.assertEqual(targets[1].label, "fallback")
        self.assertEqual(targets[1].port, "5433")

    def test_load_targets_from_config_skips_disabled_fallback_for_both(self):
        payload = {
            "database": {
                "db_name": "main_db",
                "user": "main_user",
                "password": "main_pass",
                "host": "127.0.0.1",
                "port": "5432",
            },
            "database_fallback": {
                "enabled": False,
                "db_name": "fallback_db",
                "user": "fallback_user",
                "password": "fallback_pass",
                "host": "127.0.0.1",
                "port": "5433",
            },
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")
            targets = bootstrap_postgres.load_targets_from_config(config_path, target_mode="both", maintenance_db="postgres")

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].label, "primary")

    def test_redacted_conninfo_masks_password(self):
        target = bootstrap_postgres.BootstrapTarget(
            label="manual",
            db_name="ryo_chat",
            user="postgres_user",
            password="secret_password",
            host="127.0.0.1",
            port="5432",
        )
        value = bootstrap_postgres._redacted_conninfo(target)
        self.assertIn("password=***", value)
        self.assertNotIn("secret_password", value)

    def test_manual_target_requires_required_fields(self):
        args = argparse.Namespace(
            user=None,
            password="pass",
            db_name="db",
            host="127.0.0.1",
            port="5432",
            maintenance_db="postgres",
        )
        with self.assertRaises(ValueError):
            bootstrap_postgres._manual_target_from_args(args)

    def test_bootstrap_target_returns_false_when_connection_fails(self):
        target = bootstrap_postgres.BootstrapTarget(
            label="manual",
            db_name="db",
            user="u",
            password="p",
            host="127.0.0.1",
            port="5432",
        )
        with mock.patch.object(
            bootstrap_postgres,
            "_database_exists",
            side_effect=RuntimeError("connection failed"),
        ):
            ok = bootstrap_postgres.bootstrap_target(
                target=target,
                skip_db_create=False,
                verify_only=False,
                sql_check_path=None,
                retries=1,
                retry_delay=0.0,
            )
        self.assertFalse(ok)

    def test_main_returns_2_for_missing_sql_check_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "database": {
                            "db_name": "main_db",
                            "user": "main_user",
                            "password": "main_pass",
                            "host": "127.0.0.1",
                            "port": "5432",
                        }
                    }
                ),
                encoding="utf-8",
            )

            parser = bootstrap_postgres.build_parser()
            parsed = parser.parse_args(
                [
                    "--config",
                    str(config_path),
                    "--sql-check",
                    str(Path(tmp_dir) / "missing.sql"),
                ]
            )
            with mock.patch("argparse.ArgumentParser.parse_args", return_value=parsed):
                exit_code = bootstrap_postgres.main()

        self.assertEqual(exit_code, 2)

    def test_resolve_docker_host_port_increments_when_initial_port_occupied(self):
        target = bootstrap_postgres.BootstrapTarget(
            label="primary",
            db_name="db",
            user="u",
            password="p",
            host="127.0.0.1",
            port="5432",
        )
        with (
            mock.patch.object(bootstrap_postgres, "_can_bind_host_port", side_effect=[False, True]),
            mock.patch.object(
                bootstrap_postgres,
                "_verify_instance_identity",
                return_value=(False, "occupied by non-matching service"),
            ),
        ):
            selected_port, reused_existing, reason = bootstrap_postgres._resolve_docker_host_port(
                target,
                scan_limit=3,
                retries=0,
                retry_delay=0.0,
            )

        self.assertEqual(selected_port, "5433")
        self.assertFalse(reused_existing)
        self.assertIn("selected free local port", reason)

    def test_resolve_docker_host_port_reuses_matching_existing_service(self):
        target = bootstrap_postgres.BootstrapTarget(
            label="primary",
            db_name="db",
            user="u",
            password="p",
            host="127.0.0.1",
            port="5432",
        )
        with (
            mock.patch.object(bootstrap_postgres, "_can_bind_host_port", return_value=False),
            mock.patch.object(
                bootstrap_postgres,
                "_verify_instance_identity",
                return_value=(True, "auth and db checks passed"),
            ),
            mock.patch.object(
                bootstrap_postgres,
                "_vector_extension_available_on_server",
                return_value=(True, "pgvector extension is available on server"),
            ),
        ):
            selected_port, reused_existing, reason = bootstrap_postgres._resolve_docker_host_port(
                target,
                scan_limit=0,
                retries=0,
                retry_delay=0.0,
            )

        self.assertEqual(selected_port, "5432")
        self.assertTrue(reused_existing)
        self.assertIn("instance identity/auth checks passed", reason)

    def test_resolve_docker_host_port_skips_matching_service_without_pgvector(self):
        target = bootstrap_postgres.BootstrapTarget(
            label="primary",
            db_name="db",
            user="u",
            password="p",
            host="127.0.0.1",
            port="5432",
        )
        with (
            mock.patch.object(bootstrap_postgres, "_can_bind_host_port", side_effect=[False, True]),
            mock.patch.object(
                bootstrap_postgres,
                "_verify_instance_identity",
                return_value=(True, "auth and db checks passed"),
            ),
            mock.patch.object(
                bootstrap_postgres,
                "_vector_extension_available_on_server",
                return_value=(False, "pgvector extension is unavailable on server"),
            ),
        ):
            selected_port, reused_existing, reason = bootstrap_postgres._resolve_docker_host_port(
                target,
                scan_limit=1,
                retries=0,
                retry_delay=0.0,
            )

        self.assertEqual(selected_port, "5433")
        self.assertFalse(reused_existing)
        self.assertIn("selected free local port", reason)

    def test_ensure_docker_container_uses_mapped_port_for_existing_container(self):
        target = bootstrap_postgres.BootstrapTarget(
            label="primary",
            db_name="db",
            user="u",
            password="p",
            host="127.0.0.1",
            port="5432",
        )
        with (
            mock.patch.object(bootstrap_postgres.shutil, "which", return_value="/usr/bin/docker"),
            mock.patch.object(
                bootstrap_postgres,
                "_resolve_docker_host_port",
                return_value=("5432", False, "selected free local port"),
            ),
            mock.patch.object(bootstrap_postgres, "_docker_container_exists", return_value=True),
            mock.patch.object(bootstrap_postgres, "_docker_container_running", return_value=True),
            mock.patch.object(bootstrap_postgres, "_docker_container_host_port", return_value="5544"),
        ):
            managed = bootstrap_postgres._ensure_docker_container(
                target=target,
                image="pgvector/pgvector:pg16",
                container_name="ryo-pg-primary",
                recreate=False,
                volume=None,
                scan_limit=10,
                retries=0,
                retry_delay=0.0,
            )

        self.assertTrue(managed)
        self.assertEqual(target.port, "5544")

    def test_persist_target_port_in_config_updates_section(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "database": {
                            "host": "127.0.0.1",
                            "port": "5432",
                        },
                        "database_fallback": {
                            "host": "127.0.0.1",
                            "port": "5433",
                        },
                    }
                ),
                encoding="utf-8",
            )

            bootstrap_postgres._persist_target_port_in_config(
                config_path,
                label="fallback",
                host="127.0.0.1",
                port="5543",
            )

            persisted = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["database_fallback"]["port"], "5543")


if __name__ == "__main__":
    unittest.main()
