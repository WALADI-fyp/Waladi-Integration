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
