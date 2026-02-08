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


if __name__ == "__main__":
    unittest.main()
