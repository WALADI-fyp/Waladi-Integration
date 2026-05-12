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
                                    face_found: bool, eyes_visible: int) -> int:
        import uuid as _uuid
        self._ensure_connected()
        with self._conn.cursor() as cur:
            cur.execute("""
                INSERT INTO risky_posture_alerts
                    (alert_id, user_id, device_id, detected_at,
                     nose_confidence, face_found, eyes_visible)
                VALUES (%s, %s, %s, to_timestamp(%s / 1000.0), %s, %s, %s)
                RETURNING id
            """, (str(_uuid.uuid4()), user_id, device_id, detected_at_ms,
                  nose_confidence, face_found, eyes_visible))
            return cur.fetchone()[0]

    # ──────────────────────────────────────────────
    #  Sleep alerts
    # ──────────────────────────────────────────────

    def insert_sleep_alert_start(self, *, user_id: str, device_id: str,
                                  started_at_ms: int, ear_start: float) -> int:
        import uuid as _uuid
        self._ensure_connected()
        with self._conn.cursor() as cur:
            cur.execute("""
                INSERT INTO sleep_alerts
                    (alert_id, user_id, device_id, started_at, ear_start)
                VALUES (%s, %s, %s, to_timestamp(%s / 1000.0), %s)
                RETURNING id
            """, (str(_uuid.uuid4()), user_id, device_id, started_at_ms, ear_start))
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

    # ──────────────────────────────────────────────
    #  Temperature alerts
    # ──────────────────────────────────────────────

    def init_temperature_alerts_table(self):
        """
        Creates/migrates temperature_alerts for event-based temperature alerts.
        Safe to call on every startup. If an older start/end style table exists,
        it keeps the table but relaxes old required columns so event inserts work.
        """
        self._ensure_connected()
        with self._conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS temperature_alerts (
                    id             SERIAL PRIMARY KEY,
                    user_id        TEXT NOT NULL,
                    device_id      TEXT NOT NULL,
                    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    temperature_c  DOUBLE PRECISION NOT NULL,
                    severity       TEXT NOT NULL
                )
            """)

            # If the old start/end temperature table exists, add the new columns.
            cur.execute("""
                ALTER TABLE temperature_alerts
                ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW()
            """)
            cur.execute("""
                ALTER TABLE temperature_alerts
                ADD COLUMN IF NOT EXISTS temperature_c DOUBLE PRECISION
            """)
            cur.execute("""
                ALTER TABLE temperature_alerts
                ADD COLUMN IF NOT EXISTS severity TEXT
            """)

            # Existing old columns such as started_at/temp_start_c may be NOT NULL.
            # Drop NOT NULL on columns no longer used by event-based temp alerts.
            cur.execute("""
                DO $$
                DECLARE col_name TEXT;
                BEGIN
                    FOREACH col_name IN ARRAY ARRAY[
                        'alert_id', 'started_at', 'ended_at', 'duration_s',
                        'temp_start_c', 'temp_peak_c', 'temp_end_c', 'updated_at'
                    ]
                    LOOP
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'temperature_alerts'
                              AND column_name = col_name
                        ) THEN
                            EXECUTE format(
                                'ALTER TABLE temperature_alerts ALTER COLUMN %I DROP NOT NULL',
                                col_name
                            );
                        END IF;
                    END LOOP;
                END $$
            """)

            cur.execute("""
                UPDATE temperature_alerts
                SET created_at = COALESCE(created_at, NOW())
                WHERE created_at IS NULL
            """)

            # Remove older severity CHECK constraints, then install the event-based one.
            cur.execute("""
                DO $$
                DECLARE constraint_name TEXT;
                BEGIN
                    FOR constraint_name IN
                        SELECT conname
                        FROM pg_constraint
                        WHERE conrelid = 'temperature_alerts'::regclass
                          AND contype = 'c'
                    LOOP
                        EXECUTE format(
                            'ALTER TABLE temperature_alerts DROP CONSTRAINT IF EXISTS %I',
                            constraint_name
                        );
                    END LOOP;
                END $$
            """)
            cur.execute("""
                ALTER TABLE temperature_alerts
                ADD CONSTRAINT temperature_alerts_severity_check
                CHECK (severity IN ('normal_high', 'moderately_high', 'severe'))
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_temperature_alerts_user_created_at
                ON temperature_alerts (user_id, created_at DESC)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_temperature_alerts_device_created_at
                ON temperature_alerts (device_id, created_at DESC)
            """)

        print("[DB] temperature_alerts table ready")

    def insert_temperature_alert(self, *, user_id: str, device_id: str,
                                 created_at_ms: int, temperature_c: float,
                                 severity: str) -> int:
        """Insert one event-based temperature alert and return its row id."""
        self._ensure_connected()
        with self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO temperature_alerts
                    (user_id, device_id, created_at, temperature_c, severity)
                VALUES (%s, %s, to_timestamp(%s / 1000.0), %s, %s)
                RETURNING id
                """,
                (user_id, device_id, created_at_ms, temperature_c, severity),
            )
            row = cur.fetchone()
            return row[0]

