"""
FIT File Exporter.

Creates Garmin FIT format files for compatibility with:
- TrainingPeaks
- Strava
- Garmin Connect
- Intervals.icu
"""

import struct
from io import BytesIO
from datetime import datetime, timedelta
from typing import Optional, List

import pandas as pd


# FIT Protocol constants
FIT_HEADER_SIZE = 14
FIT_PROTOCOL_VERSION = 0x20  # 2.0
FIT_PROFILE_VERSION = 0x0821  # 20.81

# Message types
MSG_FILE_ID = 0
MSG_RECORD = 20
MSG_SESSION = 18
MSG_ACTIVITY = 34

# Field types
FIT_UINT8 = 0x00
FIT_UINT16 = 0x84
FIT_UINT32 = 0x86


class FitExporter:
    """Creates FIT files from training data."""

    # FIT timestamp epoch (1989-12-31)
    FIT_EPOCH = datetime(1989, 12, 31, 0, 0, 0)

    # Default serial number; override per-export for platform deduplication
    DEFAULT_SERIAL_NUMBER = 0

    def __init__(self, serial_number: Optional[int] = None) -> None:
        self._buffer = BytesIO()
        self._data_size = 0
        self._serial_number = (
            serial_number if serial_number is not None else self.DEFAULT_SERIAL_NUMBER
        )

    def export(
        self,
        df: pd.DataFrame,
        metrics: dict,
        start_time: Optional[datetime] = None,
        sport: str = "cycling",
    ) -> bytes:
        """Export DataFrame to FIT format.

        Args:
            df: Training data with time, watts, heartrate, cadence, etc.
            metrics: Calculated metrics dict
            start_time: Activity start time (default: now)
            sport: Sport type (cycling, running, etc.)

        Returns:
            FIT file bytes
        """
        self._buffer = BytesIO()
        self._data_size = 0

        if start_time is None:
            start_time = datetime.now()

        # Write placeholder header (will update later)
        self._write_header_placeholder()

        # Write file ID message
        self._write_file_id(start_time)

        # Write record messages (main data)
        self._write_records(df, start_time)

        # Write session summary
        self._write_session(df, metrics, start_time, sport)

        # Write activity message
        self._write_activity(start_time)

        # Update header with actual data size
        self._update_header()

        # Add CRC
        self._add_crc()

        return self._buffer.getvalue()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_header_placeholder(self) -> None:
        """Write 14-byte FIT header placeholder."""
        header = struct.pack(
            "<BBHI4sH",
            FIT_HEADER_SIZE,  # Header size
            FIT_PROTOCOL_VERSION,
            FIT_PROFILE_VERSION,
            0,  # Data size (placeholder)
            b".FIT",  # Signature
            0,  # CRC (placeholder)
        )
        self._buffer.write(header)

    def _update_header(self) -> None:
        """Update header with actual data size."""
        self._buffer.seek(4)
        self._buffer.write(struct.pack("<I", self._data_size))
        self._buffer.seek(0, 2)  # Return to end

    def _add_crc(self) -> None:
        """Calculate and append CRC."""
        self._buffer.seek(0)
        data = self._buffer.read()
        crc = self._calculate_crc(data)
        self._buffer.write(struct.pack("<H", crc))

    @staticmethod
    def _calculate_crc(data: bytes) -> int:
        """Calculate FIT CRC16."""
        crc_table = [
            0x0000,
            0xCC01,
            0xD801,
            0x1400,
            0xF001,
            0x3C00,
            0x2800,
            0xE401,
            0xA001,
            0x6C00,
            0x7800,
            0xB401,
            0x5000,
            0x9C01,
            0x8801,
            0x4400,
        ]

        crc = 0
        for byte in data:
            tmp = crc_table[crc & 0xF]
            crc = (crc >> 4) & 0x0FFF
            crc = crc ^ tmp ^ crc_table[byte & 0xF]

            tmp = crc_table[crc & 0xF]
            crc = (crc >> 4) & 0x0FFF
            crc = crc ^ tmp ^ crc_table[(byte >> 4) & 0xF]

        return crc

    def _write_file_id(self, start_time: datetime) -> None:
        """Write File ID message."""
        timestamp = int((start_time - self.FIT_EPOCH).total_seconds())

        # Definition message
        definition = self._create_definition(
            MSG_FILE_ID,
            [
                (0, FIT_UINT8, 1),  # type (activity = 4)
                (1, FIT_UINT16, 1),  # manufacturer
                (2, FIT_UINT16, 1),  # product
                (3, FIT_UINT32, 1),  # serial number
                (4, FIT_UINT32, 1),  # time created
            ],
        )
        self._write_message(definition)

        # Data message
        data = struct.pack(
            "<BHHII",
            4,  # type = activity
            1,  # manufacturer = Garmin
            1,  # product
            self._serial_number,  # serial number
            timestamp,  # time created
        )
        self._write_data(MSG_FILE_ID, data)

    def _write_records(self, df: pd.DataFrame, start_time: datetime) -> None:  # noqa: C901
        """Write record messages for each data point."""
        has_power = "watts" in df.columns
        has_hr = "heartrate" in df.columns
        has_cadence = "cadence" in df.columns
        has_speed = "velocity_smooth" in df.columns or "speed" in df.columns

        # Create definition based on available fields
        fields: list[tuple[int, int, int]] = [
            (253, FIT_UINT32, 1),  # timestamp
        ]

        if has_power:
            fields.append((7, FIT_UINT16, 1))  # power
        if has_hr:
            fields.append((3, FIT_UINT8, 1))  # heart_rate
        if has_cadence:
            fields.append((4, FIT_UINT8, 1))  # cadence
        if has_speed:
            fields.append((6, FIT_UINT16, 1))  # speed (m/s * 1000)

        definition = self._create_definition(MSG_RECORD, fields)
        self._write_message(definition)

        time_col = "time" if "time" in df.columns else None
        n = len(df)

        time_values = df[time_col].values if time_col else None
        power_values = df["watts"].values if has_power else None
        hr_values = df["heartrate"].values if has_hr else None
        cadence_values = df["cadence"].values if has_cadence else None

        speed_col = "velocity_smooth" if "velocity_smooth" in df.columns else "speed"
        speed_values = None
        if has_speed and speed_col in df.columns:
            speed_values = df[speed_col].values

        for i in range(n):
            elapsed = float(time_values[i]) if time_values is not None else float(i)
            timestamp = int(
                (start_time + timedelta(seconds=elapsed) - self.FIT_EPOCH).total_seconds()
            )

            data_parts = [struct.pack("<I", timestamp)]

            if has_power and power_values is not None:
                power = int(float(power_values[i]))
                data_parts.append(struct.pack("<H", max(0, min(65535, power))))

            if has_hr and hr_values is not None:
                hr = int(float(hr_values[i]))
                data_parts.append(struct.pack("<B", max(0, min(255, hr))))

            if has_cadence and cadence_values is not None:
                cadence = int(float(cadence_values[i]))
                data_parts.append(struct.pack("<B", max(0, min(255, cadence))))

            if speed_values is not None:
                speed = float(speed_values[i]) * 1000
                data_parts.append(struct.pack("<H", max(0, min(65535, int(speed)))))

            self._write_data(MSG_RECORD, b"".join(data_parts))

    def _write_session(
        self,
        df: pd.DataFrame,
        metrics: dict,
        start_time: datetime,
        sport: str,
    ) -> None:
        """Write session summary message."""
        sport_num = {"cycling": 2, "running": 1, "swimming": 5}.get(sport, 2)

        timestamp = int((start_time - self.FIT_EPOCH).total_seconds())
        duration = len(df)  # seconds

        avg_power = int(metrics.get("avg_watts", 0))
        avg_hr = int(metrics.get("avg_hr", 0))
        max_hr = int(df["heartrate"].max()) if "heartrate" in df.columns else 0
        total_work = int(metrics.get("work_kj", 0) * 1000)  # kJ to J
        np_val = int(metrics.get("np", avg_power))
        tss = int(metrics.get("tss", 0) * 10)  # TSS * 10

        definition = self._create_definition(
            MSG_SESSION,
            [
                (253, FIT_UINT32, 1),  # timestamp
                (2, FIT_UINT32, 1),  # start_time
                (7, FIT_UINT32, 1),  # total_elapsed_time
                (8, FIT_UINT32, 1),  # total_timer_time
                (5, FIT_UINT8, 1),  # sport
                (20, FIT_UINT16, 1),  # avg_power
                (21, FIT_UINT16, 1),  # max_power
                (16, FIT_UINT8, 1),  # avg_heart_rate
                (17, FIT_UINT8, 1),  # max_heart_rate
                (48, FIT_UINT32, 1),  # total_work (joules)
                (34, FIT_UINT16, 1),  # normalized_power
                (35, FIT_UINT16, 1),  # training_stress_score
            ],
        )
        self._write_message(definition)

        max_power = int(df["watts"].max()) if "watts" in df.columns else 0

        data = struct.pack(
            "<IIIIBHHBBIHH",
            timestamp,
            timestamp,
            duration * 1000,  # total_elapsed_time ms
            duration * 1000,  # total_timer_time ms
            sport_num,
            avg_power,
            max_power,
            avg_hr,
            max_hr,
            total_work,
            np_val,
            tss,
        )
        self._write_data(MSG_SESSION, data)

    def _write_activity(self, start_time: datetime) -> None:
        """Write activity message."""
        timestamp = int((start_time - self.FIT_EPOCH).total_seconds())

        definition = self._create_definition(
            MSG_ACTIVITY,
            [
                (253, FIT_UINT32, 1),  # timestamp
                (1, FIT_UINT32, 1),  # total_timer_time
                (2, FIT_UINT16, 1),  # num_sessions
                (3, FIT_UINT8, 1),  # type (manual = 0)
                (4, FIT_UINT8, 1),  # event (activity = 26)
                (5, FIT_UINT8, 1),  # event_type (stop = 1)
            ],
        )
        self._write_message(definition)

        data = struct.pack(
            "<IIHBBB",
            timestamp,
            0,  # total_timer_time (already in session)
            1,  # num_sessions
            0,  # type = manual
            26,  # event = activity
            1,  # event_type = stop
        )
        self._write_data(MSG_ACTIVITY, data)

    @staticmethod
    def _create_definition(
        global_msg_num: int,
        fields: List[tuple],
    ) -> bytes:
        """Create definition message for a global message type."""
        num_fields = len(fields)

        # Definition header: reserved, arch (little endian), global msg num, num fields
        header = struct.pack("<xBHB", 0, global_msg_num, num_fields)

        # Field definitions
        field_defs = b""
        for field_num, field_type, field_size in fields:
            field_defs += struct.pack("<BBB", field_num, field_size, field_type)

        return header + field_defs

    def _write_message(self, definition: bytes) -> None:
        """Write a definition message."""
        header = 0x40
        self._buffer.write(struct.pack("B", header))
        self._buffer.write(definition)
        self._data_size += 1 + len(definition)

    def _write_data(self, local_msg: int, data: bytes) -> None:
        """Write a data message."""
        header = local_msg & 0x0F  # Local message, data bit clear
        self._buffer.write(struct.pack("B", header))
        self._buffer.write(data)
        self._data_size += 1 + len(data)


# ------------------------------------------------------------------
# Convenience wrapper
# ------------------------------------------------------------------


class PlatformSync:
    """Sync data to external platforms."""

    def __init__(self) -> None:
        self.fit_exporter = FitExporter()

    def export_to_fit(
        self,
        df: pd.DataFrame,
        metrics: dict,
        filename: Optional[str] = None,
    ) -> bytes:
        """Export to FIT format."""
        return self.fit_exporter.export(df, metrics)

    def prepare_strava_description(
        self,
        metrics: dict,
        notes: str = "",
    ) -> str:
        """Create description text for Strava upload.

        Args:
            metrics: Calculated metrics
            notes: Optional user notes

        Returns:
            Formatted description string
        """
        lines = ["📊 Pro Athlete Dashboard Analysis\n"]

        if metrics.get("np"):
            lines.append(f"⚡ NP: {metrics['np']:.0f} W")
        if metrics.get("avg_watts"):
            lines.append(f"💪 Avg Power: {metrics['avg_watts']:.0f} W")
        if metrics.get("avg_hr"):
            lines.append(f"❤️ Avg HR: {metrics['avg_hr']:.0f} bpm")
        if metrics.get("work_kj"):
            lines.append(f"🔋 Work: {metrics['work_kj']:.0f} kJ")
        if metrics.get("carbs_total"):
            lines.append(f"🍎 Carbs: {metrics['carbs_total']:.0f} g")

        if notes:
            lines.append(f"\n📝 Notes:\n{notes}")

        return "\n".join(lines)

    def prepare_intervals_icu_data(
        self,
        df: pd.DataFrame,
        metrics: dict,
    ) -> dict:
        """Prepare data for Intervals.icu API.

        Returns dict compatible with Intervals.icu wellness/activity API.
        """
        return {
            "type": "Ride",
            "icu_training_load": metrics.get("tss", 0),
            "icu_intensity": metrics.get("if_factor", 0),
            "average_watts": metrics.get("avg_watts", 0),
            "weighted_average_watts": metrics.get("np", 0),
            "average_heartrate": metrics.get("avg_hr", 0),
            "max_heartrate": df["heartrate"].max() if "heartrate" in df.columns else 0,
            "moving_time": len(df),
            "joules": int(metrics.get("work_kj", 0) * 1000),
        }
