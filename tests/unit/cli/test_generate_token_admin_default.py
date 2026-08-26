"""Tests for generate-token's admin-grant defaults and --admin/--no-admin flag.

Covers the fix for viewer users silently ending up non-admin: a brand-new
user created while the deployment is still single-user (0-1 existing users)
is admin by default; once a 3rd distinct user exists, new users default to
non-admin unless --admin is passed explicitly. An existing user's admin
status is only touched by an explicit --admin/--no-admin flag.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from typer.testing import CliRunner

from videoannotator.cli import app
from videoannotator.database.crud import UserCRUD

runner = CliRunner()


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    import videoannotator.database.database as db_module

    db_path = tmp_path / "test_generate_token.db"
    engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(
        db_module,
        "SessionLocal",
        sessionmaker(autocommit=False, autoflush=False, bind=engine),
    )
    db_module.Base.metadata.create_all(bind=engine)
    yield db_module.SessionLocal


def _generate(email: str, username: str, *extra_args: str):
    return runner.invoke(
        app,
        [
            "generate-token",
            "--user",
            email,
            "--username",
            username,
            "--key-name",
            f"{username} key",
            *extra_args,
        ],
    )


class TestAutoAdminDefault:
    def test_first_new_user_is_auto_admin(self, temp_db):
        result = _generate("first@example.com", "first")
        assert result.exit_code == 0, result.output
        assert "Admin:      Yes" in result.output

        db = temp_db()
        try:
            user = UserCRUD.get_by_email(db, "first@example.com")
            assert user.is_admin is True
        finally:
            db.close()

    def test_second_new_user_is_still_auto_admin(self, temp_db):
        _generate("first@example.com", "first")
        result = _generate("second@example.com", "second")
        assert "Admin:      Yes" in result.output

    def test_third_new_user_is_not_auto_admin(self, temp_db):
        _generate("first@example.com", "first")
        _generate("second@example.com", "second")
        result = _generate("third@example.com", "third")
        assert "Admin:      No" in result.output

        db = temp_db()
        try:
            user = UserCRUD.get_by_email(db, "third@example.com")
            assert user.is_admin is False
        finally:
            db.close()


class TestExplicitAdminFlag:
    def test_explicit_admin_flag_grants_even_past_single_user_threshold(self, temp_db):
        _generate("first@example.com", "first")
        _generate("second@example.com", "second")
        result = _generate("third@example.com", "third", "--admin")
        assert "Admin:      Yes" in result.output
        assert "explicit --admin" in result.output

    def test_explicit_no_admin_denies_even_for_first_user(self, temp_db):
        result = _generate("first@example.com", "first", "--no-admin")
        assert "Admin:      No" in result.output

    def test_explicit_admin_promotes_existing_non_admin_user(self, temp_db):
        _generate("first@example.com", "first")
        _generate("second@example.com", "second")
        _generate("third@example.com", "third")  # non-admin (3rd user)

        result = _generate("third@example.com", "third", "--admin")
        assert "Granted administrator privileges for existing user" in result.output
        assert "Admin:      Yes" in result.output

        db = temp_db()
        try:
            user = UserCRUD.get_by_email(db, "third@example.com")
            assert user.is_admin is True
        finally:
            db.close()

    def test_explicit_no_admin_demotes_existing_admin_user(self, temp_db):
        result1 = _generate("first@example.com", "first")
        assert "Admin:      Yes" in result1.output  # auto-admin, first user

        result2 = _generate("first@example.com", "first", "--no-admin")
        assert "Revoked administrator privileges for existing user" in result2.output
        assert "Admin:      No" in result2.output

    def test_re_generating_for_existing_user_without_flag_leaves_status_unchanged(
        self, temp_db
    ):
        _generate("first@example.com", "first")  # auto-admin
        result = _generate("first@example.com", "first")  # no flag, existing user
        assert "Admin:      Yes" in result.output
        assert "Granted" not in result.output
        assert "Revoked" not in result.output
