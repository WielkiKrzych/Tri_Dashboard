"""
Athlete Profile Management.

SQLite-backed CRUD store for managing multiple athlete profiles,
each with their own physiological parameters.
"""

import uuid
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from modules.db.base import BaseStore

logger = logging.getLogger(__name__)


@dataclass
class AthleteProfile:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    rider_weight: float = 97.0
    rider_height: float = 180.0
    rider_age: int = 30
    is_male: bool = True
    cp: float = 420.0
    w_prime: float = 15600.0
    vt1_watts: float = 320.0
    vt2_watts: float = 410.0
    vt1_vent: float = 73.0
    vt2_vent: float = 105.0
    crank_length: float = 165.0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_settings_dict(self) -> dict:
        """Convert to dict matching SettingsManager.default_settings keys."""
        return {
            "rider_weight": self.rider_weight,
            "rider_height": self.rider_height,
            "rider_age": self.rider_age,
            "is_male": self.is_male,
            "cp": self.cp,
            "w_prime": self.w_prime,
            "vt1_watts": self.vt1_watts,
            "vt2_watts": self.vt2_watts,
            "vt1_vent": self.vt1_vent,
            "vt2_vent": self.vt2_vent,
            "crank_length": self.crank_length,
        }

    @classmethod
    def from_settings_dict(
        cls, name: str, settings: dict, profile_id: Optional[str] = None
    ) -> "AthleteProfile":
        """Create from SettingsManager-compatible dict."""
        return cls(
            id=profile_id or uuid.uuid4().hex[:12],
            name=name,
            rider_weight=settings.get("rider_weight", 97.0),
            rider_height=settings.get("rider_height", 180.0),
            rider_age=settings.get("rider_age", 30),
            is_male=settings.get("is_male", True),
            cp=settings.get("cp", 420.0),
            w_prime=settings.get("w_prime", 15600.0),
            vt1_watts=settings.get("vt1_watts", 320.0),
            vt2_watts=settings.get("vt2_watts", 410.0),
            vt1_vent=settings.get("vt1_vent", 73.0),
            vt2_vent=settings.get("vt2_vent", 105.0),
            crank_length=settings.get("crank_length", 165.0),
        )


class AthleteProfileStore(BaseStore):
    """SQLite-backed CRUD store for athlete profiles."""

    table_name = "athlete_profiles"
    _schema_sql = """
        CREATE TABLE IF NOT EXISTS athlete_profiles (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            rider_weight REAL DEFAULT 97.0,
            rider_height REAL DEFAULT 180.0,
            rider_age INTEGER DEFAULT 30,
            is_male INTEGER DEFAULT 1,
            cp REAL DEFAULT 420.0,
            w_prime REAL DEFAULT 15600.0,
            vt1_watts REAL DEFAULT 320.0,
            vt2_watts REAL DEFAULT 410.0,
            vt1_vent REAL DEFAULT 73.0,
            vt2_vent REAL DEFAULT 105.0,
            crank_length REAL DEFAULT 165.0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """

    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> AthleteProfile:
        return AthleteProfile(
            id=row["id"],
            name=row["name"],
            rider_weight=row["rider_weight"],
            rider_height=row["rider_height"],
            rider_age=row["rider_age"],
            is_male=bool(row["is_male"]),
            cp=row["cp"],
            w_prime=row["w_prime"],
            vt1_watts=row["vt1_watts"],
            vt2_watts=row["vt2_watts"],
            vt1_vent=row["vt1_vent"],
            vt2_vent=row["vt2_vent"],
            crank_length=row["crank_length"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create_profile(self, profile: AthleteProfile) -> str:
        """Insert new profile. Returns profile.id."""
        self.execute(
            """
            INSERT INTO athlete_profiles
                (id, name, rider_weight, rider_height, rider_age, is_male,
                 cp, w_prime, vt1_watts, vt2_watts, vt1_vent, vt2_vent,
                 crank_length, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                profile.id,
                profile.name,
                profile.rider_weight,
                profile.rider_height,
                profile.rider_age,
                int(profile.is_male),
                profile.cp,
                profile.w_prime,
                profile.vt1_watts,
                profile.vt2_watts,
                profile.vt1_vent,
                profile.vt2_vent,
                profile.crank_length,
                profile.created_at,
                profile.updated_at,
            ),
        )
        return profile.id

    def get_profile(self, profile_id: str) -> Optional[AthleteProfile]:
        """Fetch single profile by id. Returns None if not found."""
        row = self.query_one("SELECT * FROM athlete_profiles WHERE id = ?", (profile_id,))
        return self._row_to_profile(row) if row else None

    def get_all_profiles(self) -> list[AthleteProfile]:
        """Return all profiles ordered by name ASC."""
        rows = self.query("SELECT * FROM athlete_profiles ORDER BY name ASC")
        return [self._row_to_profile(r) for r in rows]

    def update_profile(self, profile: AthleteProfile) -> bool:
        """Update existing profile. Returns True if row changed."""
        cursor = self.execute(
            """
            UPDATE athlete_profiles SET
                name=?, rider_weight=?, rider_height=?, rider_age=?,
                is_male=?, cp=?, w_prime=?, vt1_watts=?, vt2_watts=?,
                vt1_vent=?, vt2_vent=?, crank_length=?, updated_at=CURRENT_TIMESTAMP
            WHERE id=?
        """,
            (
                profile.name,
                profile.rider_weight,
                profile.rider_height,
                profile.rider_age,
                int(profile.is_male),
                profile.cp,
                profile.w_prime,
                profile.vt1_watts,
                profile.vt2_watts,
                profile.vt1_vent,
                profile.vt2_vent,
                profile.crank_length,
                profile.id,
            ),
        )
        return cursor.rowcount > 0

    def delete_profile(self, profile_id: str) -> bool:
        """Delete profile. Reassigns orphaned sessions to 'default'."""
        self.execute(
            "UPDATE sessions SET athlete_id = 'default' WHERE athlete_id = ?",
            (profile_id,),
        )
        return self.delete("id", profile_id)

    def profile_name_exists(self, name: str, exclude_id: Optional[str] = None) -> bool:
        """Check if name is already taken (case-insensitive)."""
        if exclude_id:
            row = self.query_one(
                "SELECT 1 FROM athlete_profiles WHERE LOWER(name)=? AND id!=?",
                (name.lower(), exclude_id),
            )
        else:
            row = self.query_one(
                "SELECT 1 FROM athlete_profiles WHERE LOWER(name)=?",
                (name.lower(),),
            )
        return row is not None

    def get_or_create_default(self) -> AthleteProfile:
        """Return profile with id='default', creating it from hardcoded defaults if needed."""
        existing = self.get_profile("default")
        if existing:
            return existing
        from modules.settings import SettingsManager

        defaults = SettingsManager().default_settings
        default_profile = AthleteProfile(
            id="default",
            name="Domyślny",
            rider_weight=defaults.get("rider_weight", 97.0),
            rider_height=defaults.get("rider_height", 180.0),
            rider_age=defaults.get("rider_age", 30),
            is_male=defaults.get("is_male", True),
            cp=defaults.get("cp", 420.0),
            w_prime=defaults.get("w_prime", 15600.0),
            vt1_watts=defaults.get("vt1_watts", 320.0),
            vt2_watts=defaults.get("vt2_watts", 410.0),
            vt1_vent=defaults.get("vt1_vent", 73.0),
            vt2_vent=defaults.get("vt2_vent", 105.0),
            crank_length=defaults.get("crank_length", 165.0),
        )
        self.create_profile(default_profile)
        return default_profile
