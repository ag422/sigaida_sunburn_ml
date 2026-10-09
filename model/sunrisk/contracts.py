"""Shared data formats between firmware, model and app. Prose version: docs/contracts.md.

Everything here is plain data plus pure (bytes <-> object) conversion, so it maps directly to
TypeScript interfaces and a DataView-based decoder.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

CONTRACT_VERSION = 1

# ---------------------------------------------------------------------------------------
# Body parts and event vocabularies
# ---------------------------------------------------------------------------------------

# "forearm" is the sensor site (device on the wrist, facing out from the back of the forearm).
BODY_PARTS = ("forearm", "face", "neck", "shoulders", "upper_back", "chest", "thighs", "shins")

ENVIRONMENTS = ("sun", "shade", "cloud", "indoor")
ACTIVITIES = ("lying", "sitting", "standing", "walking", "light_activity", "vigorous", "swimming")

# ---------------------------------------------------------------------------------------
# BLE packet (device -> phone), little-endian, 164 bytes
# ---------------------------------------------------------------------------------------
#
# One packet per UV conversion (400 ms at 20-bit), carrying the IMU samples taken during that
# window. 164 bytes fits in a single notification with the iOS default ATT MTU (185).

IMU_SAMPLES_PER_PACKET = 8          # 20 Hz IMU x 0.4 s
IMU_RATE_HZ = 20

FLAG_HAS_MAG = 1 << 0               # magnetometer fields are valid
FLAG_UV_SATURATED = 1 << 1          # LTR390 reading hit full scale; lower gain needed
FLAG_LOW_BATTERY = 1 << 2
FLAG_UV_INVALID = 1 << 3            # no new UV conversion in this packet

_HEADER = struct.Struct("<BBHI" "IBB" "HH" "BB")   # 20 bytes
_IMU = struct.Struct("<9h")                         # 18 bytes per sample
PACKET_SIZE = _HEADER.size + IMU_SAMPLES_PER_PACKET * _IMU.size

# Fixed-point scales for the int16 IMU fields
ACC_MG_PER_LSB = 1.0       # accel in milli-g          (+-32 g range)
GYRO_DPS_PER_LSB = 0.1     # gyro in 0.1 deg/s         (+-3276 deg/s)
MAG_UT_PER_LSB = 0.1       # magnetometer in 0.1 uT    (+-3276 uT)

UV_GAINS = (1, 3, 6, 9, 18)
UV_RESOLUTIONS_BITS = (13, 16, 17, 18, 19, 20)


@dataclass
class ImuSample:
    acc_g: tuple[float, float, float]
    gyro_dps: tuple[float, float, float]
    mag_ut: tuple[float, float, float] | None = None


@dataclass
class BlePacket:
    seq: int                    # uint16, wraps; gaps = dropped packets
    t_device_ms: int            # uint32 ms since boot, time of the UV conversion end
    uv_counts: int              # raw LTR390 UVS counts (20-bit max)
    uv_gain: int                # one of UV_GAINS
    uv_resolution_bits: int     # one of UV_RESOLUTIONS_BITS
    battery_mv: int
    uvi_device: float           # firmware's own estimate (display only; phone recomputes)
    imu: list[ImuSample]
    flags: int = 0
    version: int = CONTRACT_VERSION

    def encode(self) -> bytes:
        if len(self.imu) != IMU_SAMPLES_PER_PACKET:
            raise ValueError(f"need exactly {IMU_SAMPLES_PER_PACKET} IMU samples")
        if self.uv_gain not in UV_GAINS or self.uv_resolution_bits not in UV_RESOLUTIONS_BITS:
            raise ValueError("invalid gain/resolution")
        head = _HEADER.pack(
            self.version, self.flags, self.seq & 0xFFFF, self.t_device_ms & 0xFFFFFFFF,
            self.uv_counts, self.uv_gain, self.uv_resolution_bits,
            self.battery_mv, round(self.uvi_device * 100),
            IMU_SAMPLES_PER_PACKET, IMU_RATE_HZ,
        )
        body = b"".join(_IMU.pack(*_imu_to_ints(s)) for s in self.imu)
        return head + body

    @classmethod
    def decode(cls, data: bytes) -> "BlePacket":
        if len(data) != PACKET_SIZE:
            raise ValueError(f"expected {PACKET_SIZE} bytes, got {len(data)}")
        (version, flags, seq, t_ms, counts, gain, res, batt, uvi100, n, rate) = _HEADER.unpack_from(data, 0)
        if version != CONTRACT_VERSION:
            raise ValueError(f"unsupported packet version {version}")
        has_mag = bool(flags & FLAG_HAS_MAG)
        imu = [_ints_to_imu(_IMU.unpack_from(data, _HEADER.size + i * _IMU.size), has_mag) for i in range(n)]
        return cls(seq, t_ms, counts, gain, res, batt, uvi100 / 100, imu, flags, version)


def _imu_to_ints(s: ImuSample) -> tuple[int, ...]:
    mag = s.mag_ut or (0.0, 0.0, 0.0)
    return (*(_i16(v / ACC_MG_PER_LSB * 1000) for v in s.acc_g),
            *(_i16(v / GYRO_DPS_PER_LSB) for v in s.gyro_dps),
            *(_i16(v / MAG_UT_PER_LSB) for v in mag))


def _ints_to_imu(v: tuple[int, ...], has_mag: bool) -> ImuSample:
    acc = tuple(x * ACC_MG_PER_LSB / 1000 for x in v[0:3])
    gyro = tuple(x * GYRO_DPS_PER_LSB for x in v[3:6])
    mag = tuple(x * MAG_UT_PER_LSB for x in v[6:9]) if has_mag else None
    return ImuSample(acc, gyro, mag)


def _i16(x: float) -> int:
    return max(-32768, min(32767, round(x)))


# ---------------------------------------------------------------------------------------
# Recorded session CSV (one row per IMU sample, 20 Hz)
# ---------------------------------------------------------------------------------------

SESSION_CSV_COLUMNS = (
    "t_unix_s",         # phone-aligned UTC time of this IMU sample (float seconds)
    "t_device_ms",      # device clock of the packet
    "seq",              # packet sequence number
    "ax_g", "ay_g", "az_g",
    "gx_dps", "gy_dps", "gz_dps",
    "mx_ut", "my_ut", "mz_ut",      # empty if no magnetometer
    "uv_counts", "uv_gain", "uv_res_bits",
    "uvi",              # phone-computed UVI using the session's calibration
    "uv_new",           # 1 on the last IMU row of each packet (when the UV conversion ended)
    "battery_mv",
)

EVENTS_CSV_COLUMNS = ("t_unix_s", "kind", "value")
# kind: "label_environment" (value in ENVIRONMENTS), "label_activity" (value in ACTIVITIES),
#       "sunscreen" (value "spf=30;amount=0.8;parts=face|neck"), "note" (free text)


# ---------------------------------------------------------------------------------------
# Risk engine interface (what the app calls)
# ---------------------------------------------------------------------------------------

@dataclass
class MedObservation:
    effective_dose_j_m2: float      # dose to the observed skin site, after sunscreen
    burned: float                   # 0/1, or a probability from the redness classifier


@dataclass
class UserProfile:
    fitzpatrick: int                # 1-6
    med_observations: list[MedObservation] = field(default_factory=list)


@dataclass
class SunscreenApplication:
    t_unix_s: float
    spf_label: float
    amount_mg_cm2: float | None = None          # None -> typical (assumption)
    body_parts: tuple[str, ...] | None = None   # None -> all exposed parts


@dataclass
class ForecastPoint:
    t_unix_s: float
    uvi: float                      # forecast all-sky UVI (e.g. Open-Meteo hourly)


@dataclass
class RiskStep:
    """One engine update, typically every few seconds."""
    t_unix_s: float
    uvi_ambient: float              # sensor-derived, orientation-corrected ambient UVI
    environment: str = "sun"        # one of ENVIRONMENTS
    activity: str = "standing"      # one of ACTIVITIES
    exposure_ratios: dict[str, float] | None = None  # per body part, relative to ambient; None -> 1.0


@dataclass
class Quantiles:
    p10: float
    p50: float
    p90: float


@dataclass
class BodyPartRisk:
    body_part: str
    minutes_left: Quantiles         # alert on p10 (pessimistic)
    fraction_med_used: Quantiles
    effective_spf: float            # median across particles
    beyond_horizon: bool = False    # True if even p10 did not reach MED within the horizon


@dataclass
class RiskOutput:
    t_unix_s: float
    parts: list[BodyPartRisk]
    first_to_burn: str              # body part with the lowest p10 minutes_left
    alert: bool                     # p10 minutes_left below the alert threshold
    projection: str                 # "constant" or "forecast"
    contract_version: int = CONTRACT_VERSION
