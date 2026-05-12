import psycopg2
import psycopg2.extras
from typing import Optional


class DbClient:
    """
    Thin wrapper around psycopg2 for writing sensor readings to TimescaleDB.
    Maintains a single persistent connection with auto-reconnect on failure.
    """

    def __init__(self, *, host: str, port: int, dbname: str, user: str, password: str, sslmode: str = "require"):
        self._dsn = (
            f"host={host} port={port} dbname={dbname} "
            f"user={user} password={password} sslmode={sslmode}"
        )
        self._conn: Optional[psycopg2.extensions.connection] = None

    def connect(self):
        self._conn = psycopg2.connect(self._dsn)
        self._conn.autocommit = True
        print("[DB] connected to TimescaleDB")

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _ensure_connected(self):
        if self._conn is None or self._conn.closed:
            print("[DB] reconnecting...")
            self.connect()

    def init_db(self):
        """
        Creates user_devices and sensor_readings tables if they don't exist.
        Converts sensor_readings into a TimescaleDB hypertable if not already one.
        Safe to call every time on startup.
        """
        self._ensure_connected()
        with self._conn.cursor() as cur:

            # user_devices — maps Clerk user IDs to Pi device IDs
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_devices (
                    id         SERIAL PRIMARY KEY,
                    user_id    TEXT NOT NULL,
                    device_id  TEXT NOT NULL UNIQUE,
                    name       TEXT DEFAULT 'My Device',
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            # sensor_readings — time-series table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sensor_readings (
                    time                TIMESTAMPTZ NOT NULL,
                    user_id             TEXT NOT NULL,
                    device_id           TEXT NOT NULL,
                    room_temperature_c  DOUBLE PRECISION,
                    room_humidity_rh    DOUBLE PRECISION,
                    breathing_rate_bpm  DOUBLE PRECISION,
                    heart_rate_bpm      DOUBLE PRECISION,
                    body_temperature_c  DOUBLE PRECISION,
                    mock_fields         TEXT[],
                    source              TEXT
                )
            """)

            # Convert to hypertable — skips silently if already one
            cur.execute("""
                SELECT create_hypertable(
                    'sensor_readings', 'time',
                    if_not_exists => TRUE
                )
            """)

            # Index for per-user queries ordered by time
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_sensor_readings_user_time
                ON sensor_readings (user_id, time DESC)
            """)

        print("[DB] tables and hypertable ready")

    def get_user_id(self, device_id: str) -> Optional[str]:
        """Look up the Clerk user_id paired to this device_id."""
        self._ensure_connected()
        with self._conn.cursor() as cur:
            cur.execute(
                "SELECT user_id FROM user_devices WHERE device_id = %s LIMIT 1",
                (device_id,),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def insert_reading(
        self,
        *,
        user_id: str,
        device_id: str,
        ts_ms: int,
        room_temperature_c: Optional[float],
        room_humidity_rh: Optional[float],
        breathing_rate_bpm: Optional[float],
        heart_rate_bpm: Optional[float],
        body_temperature_c: Optional[float],
        mock_fields: list,
        source: str,
    ):
        """Insert one row into sensor_readings."""
        self._ensure_connected()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sensor_readings (
                    time,
                    user_id,
                    device_id,
                    room_temperature_c,
                    room_humidity_rh,
                    breathing_rate_bpm,
                    heart_rate_bpm,
                    body_temperature_c,
                    mock_fields,
                    source
                ) VALUES (
                    to_timestamp(%s / 1000.0),
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    ts_ms,
                    user_id,
                    device_id,
                    room_temperature_c,
                    room_humidity_rh,
                    breathing_rate_bpm,
                    heart_rate_bpm,
                    body_temperature_c,
                    mock_fields or [],
                    source,
                ),
            )

    # ──────────────────────────────────────────────
    #  Cry alerts
    # ──────────────────────────────────────────────

    def init_cry_alerts_table(self):
        """
        Creates the cry_alerts table if it doesn't exist.
        Safe to call on every startup.
        """
        self._ensure_connected()
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS cry_alerts (
                    id          SERIAL PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    device_id   TEXT NOT NULL,
                    started_at  TIMESTAMPTZ NOT NULL,
                    ended_at    TIMESTAMPTZ,
                    duration_s  DOUBLE PRECISION
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_cry_alerts_user
                ON cry_alerts (user_id, started_at DESC)
            """)
        print("[DB] cry_alerts table ready")

    def insert_cry_alert_start(self, *, user_id: str, device_id: str, started_at_ms: int) -> int:
        """
        Insert a new crying alert with no end time yet.
        Returns the alert id so we can close it later.
        """
        self._ensure_connected()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO cry_alerts (user_id, device_id, started_at)
                VALUES (%s, %s, to_timestamp(%s / 1000.0))
                RETURNING id
                """,
                (user_id, device_id, started_at_ms),
            )
            row = cur.fetchone()
            return row[0]

    def update_cry_alert_end(self, *, alert_id: int, ended_at_ms: int):
        """
        Close an open alert by setting ended_at and computing duration_s.
        """
        self._ensure_connected()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE cry_alerts
                SET
                    ended_at   = to_timestamp(%s / 1000.0),
                    duration_s = EXTRACT(EPOCH FROM (
                                     to_timestamp(%s / 1000.0) - started_at
                                 ))
                WHERE id = %s
                """,
                (ended_at_ms, ended_at_ms, alert_id),
            )

    # ──────────────────────────────────────────────
    #  Risky posture alerts
    # ──────────────────────────────────────────────

    def insert_risky_posture_alert(self, *, user_id: str, device_id: str,
                                    detected_at_ms: int, nose_confidence: float,
                                    face_found: bool, eyes_visible: int):
        self._ensure_connected()
        with self._conn.cursor() as cur:
            cur.execute("""
                INSERT INTO risky_posture_alerts
                    (user_id, device_id, detected_at, nose_confidence, face_found, eyes_visible)
                VALUES (%s, %s, to_timestamp(%s / 1000.0), %s, %s, %s)
                RETURNING id
            """, (user_id, device_id, detected_at_ms,
                  nose_confidence, face_found, eyes_visible))
            return cur.fetchone()[0]

    # ──────────────────────────────────────────────
    #  Sleep alerts
    # ──────────────────────────────────────────────

    def insert_sleep_alert_start(self, *, user_id: str, device_id: str,
                                  started_at_ms: int, ear_start: float) -> int:
        self._ensure_connected()
        with self._conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sleep_alerts
                    (user_id, device_id, started_at, ear_start)
                VALUES (%s, %s, to_timestamp(%s / 1000.0), %s)
                RETURNING id
            """, (user_id, device_id, started_at_ms, ear_start))
            return cur.fetchone()[0]

    def update_sleep_alert_end(self, *, alert_id: int, ended_at_ms: int, ear_end: float):
        self._ensure_connected()
        with self._conn.cursor() as cur:
            cur.execute("""
                UPDATE sleep_alerts
                SET
                    ended_at   = to_timestamp(%s / 1000.0),
                    ear_end    = %s,
                    duration_s = EXTRACT(EPOCH FROM (
                                     to_timestamp(%s / 1000.0) - started_at
                                 )),
                    updated_at = NOW()
                WHERE id = %s
            """, (ended_at_ms, ear_end, ended_at_ms, alert_id))
