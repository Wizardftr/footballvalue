"""Accounts and sign-in.

Deliberately small. This is a private app shared with a handful of people, so it
has accounts, not a signup funnel: the owner creates them with ``fv user add`` or
from the People page, and there is no public registration, no password reset
emails, and no third-party identity provider to configure.

Two decisions worth stating:

* **Sessions live in Streamlit's session state, not in a cookie.** That means a
  browser refresh signs you out again, which is mildly annoying and entirely
  deliberate — a persistent cookie needs a signing secret, rotation, and a
  revocation story, and none of that is worth building for a group this size.
  Nothing about the login is stored client-side.
* **Lockout state lives on the user row.** If it lived in memory, restarting the
  app would hand an attacker a fresh budget of attempts.
"""

from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import bcrypt
from sqlalchemy import func, select

from fv.config import Config, load_config
from fv.db.models import User
from fv.db.session import session_scope

MIN_PASSWORD_LENGTH = 10
MAX_FAILED_LOGINS = 8
LOCKOUT = timedelta(minutes=15)
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    """Anything that should be shown to whoever is trying to sign in."""


@dataclass(frozen=True)
class Account:
    """The signed-in user, as the UI needs it. Never carries the password hash."""

    id: int
    email: str
    display_name: str
    role: str

    @property
    def is_owner(self) -> bool:
        return self.role == "owner"


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------

def _prepare(password: str) -> bytes:
    """bcrypt silently truncates at 72 bytes, so hash the password down to a fixed
    length first. Without this, two long passphrases sharing a 72-byte prefix would
    be the same password."""
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_prepare(password), stored_hash.encode("ascii"))
    except (ValueError, TypeError):
        return False


def check_password_strength(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if password.lower() in {"password12", "footballvalue", "1234567890"}:
        raise AuthError("Pick something less guessable.")


# ---------------------------------------------------------------------------
# Account management
# ---------------------------------------------------------------------------

def normalise_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not EMAIL.match(email):
        raise AuthError(f"'{email}' is not a valid email address.")
    return email


def create_user(
    email: str,
    password: str,
    display_name: str | None = None,
    role: str = "member",
    cfg: Config | None = None,
) -> Account:
    cfg = cfg or load_config()
    if role not in ("owner", "member"):
        raise AuthError(f"Unknown role: {role}")
    email = normalise_email(email)
    check_password_strength(password)

    with session_scope(cfg) as s:
        if s.scalar(select(User).where(User.email == email)) is not None:
            raise AuthError(f"An account already exists for {email}.")
        user = User(
            email=email,
            display_name=(display_name or email.split("@")[0]).strip()[:64],
            password_hash=hash_password(password),
            role=role,
            created_at=datetime.utcnow(),
        )
        s.add(user)
        s.flush()
        return Account(user.id, user.email, user.display_name, user.role)


def authenticate(email: str, password: str, cfg: Config | None = None) -> Account:
    """Return the account, or raise AuthError. Failures are counted and throttled."""
    cfg = cfg or load_config()
    try:
        email = normalise_email(email)
    except AuthError:
        raise AuthError("Wrong email or password.") from None

    # The failure is recorded, then raised *after* the session has committed.
    # Raising inside session_scope rolls the transaction back, which would silently
    # discard the very counter the lockout depends on — the throttle would look
    # present in the code and do nothing at all.
    problem: str | None = None
    account: Account | None = None

    with session_scope(cfg) as s:
        user = s.scalar(select(User).where(User.email == email))
        # The same message whether the account is missing or the password is wrong:
        # a distinct "no such account" tells a stranger which emails are registered.
        if user is None:
            problem = "Wrong email or password."
        elif not user.is_active:
            problem = "That account has been disabled."
        elif user.locked_until and user.locked_until > datetime.utcnow():
            wait = int((user.locked_until - datetime.utcnow()).total_seconds() // 60) + 1
            problem = f"Too many failed attempts. Try again in {wait} minute(s)."
        elif not verify_password(password, user.password_hash):
            user.failed_logins = (user.failed_logins or 0) + 1
            if user.failed_logins >= MAX_FAILED_LOGINS:
                user.locked_until = datetime.utcnow() + LOCKOUT
                user.failed_logins = 0
            problem = "Wrong email or password."
        else:
            user.failed_logins = 0
            user.locked_until = None
            user.last_login_at = datetime.utcnow()
            account = Account(user.id, user.email, user.display_name, user.role)

    if problem is not None:
        raise AuthError(problem)
    return account


def set_password(user_id: int, password: str, cfg: Config | None = None) -> None:
    cfg = cfg or load_config()
    check_password_strength(password)
    with session_scope(cfg) as s:
        user = s.get(User, user_id)
        if user is None:
            raise AuthError(f"No account with id {user_id}.")
        user.password_hash = hash_password(password)
        user.failed_logins = 0
        user.locked_until = None


def change_password(user_id: int, current: str, new: str, cfg: Config | None = None) -> None:
    """Changing your own password requires proving you know the old one."""
    cfg = cfg or load_config()
    with session_scope(cfg) as s:
        user = s.get(User, user_id)
        if user is None or not verify_password(current, user.password_hash):
            raise AuthError("Current password is not correct.")
    set_password(user_id, new, cfg)


def set_active(user_id: int, active: bool, cfg: Config | None = None) -> None:
    """Disable an account. The last active owner cannot be disabled."""
    cfg = cfg or load_config()
    with session_scope(cfg) as s:
        user = s.get(User, user_id)
        if user is None:
            raise AuthError(f"No account with id {user_id}.")
        if not active and user.role == "owner":
            owners = s.scalar(
                select(func.count(User.id)).where(User.role == "owner", User.is_active.is_(True))
            )
            if owners <= 1:
                raise AuthError("That is the only owner — disabling it would lock everyone out.")
        user.is_active = active


def list_users(cfg: Config | None = None) -> list[dict]:
    cfg = cfg or load_config()
    with session_scope(cfg) as s:
        rows = s.scalars(select(User).order_by(User.id)).all()
        return [
            {
                "id": u.id,
                "email": u.email,
                "display_name": u.display_name,
                "role": u.role,
                "is_active": u.is_active,
                "created_at": u.created_at,
                "last_login_at": u.last_login_at,
            }
            for u in rows
        ]


def user_count(cfg: Config | None = None) -> int:
    cfg = cfg or load_config()
    with session_scope(cfg) as s:
        return int(s.scalar(select(func.count(User.id))) or 0)


# ---------------------------------------------------------------------------
# The account the command line acts as
# ---------------------------------------------------------------------------

LOCAL_EMAIL = "local@footballvalue.invalid"
# Not a bcrypt hash, so checkpw raises and verify_password returns False. The local
# account therefore exists for ownership of rows and can never be signed into.
UNUSABLE_HASH = "!"


def local_user_id(cfg: Config | None = None) -> int:
    """Whose bets the CLI is placing.

    The command line has no login — whoever can run `fv` already has the database.
    It acts as the first owner account, and if none exists yet it creates a
    placeholder that owns the local data. The placeholder cannot be signed into
    from the web app, so adding accounts later never grants access to the CLI's
    history by accident.
    """
    cfg = cfg or load_config()
    with session_scope(cfg) as s:
        owner = s.scalar(
            select(User)
            .where(User.role == "owner", User.is_active.is_(True))
            .order_by(User.id)
            .limit(1)
        )
        if owner is not None:
            return owner.id
        user = User(
            email=LOCAL_EMAIL,
            display_name="Local",
            password_hash=UNUSABLE_HASH,
            role="owner",
            created_at=datetime.utcnow(),
        )
        s.add(user)
        s.flush()
        return user.id


def get_account(user_id: int, cfg: Config | None = None) -> Account | None:
    cfg = cfg or load_config()
    with session_scope(cfg) as s:
        user = s.get(User, user_id)
        if user is None or not user.is_active:
            return None
        return Account(user.id, user.email, user.display_name, user.role)
