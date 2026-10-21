"""WANFAN EEG v1.0 wire format (manual dated 2026-05-20).

CRC follows CCITT-FALSE (unreflected, xorout=0); the manual specifies its
polynomial/init/coverage but does not explicitly specify reflection/xorout.
"""
from dataclasses import dataclass
import binascii
import struct

import numpy as np

SAMPLE_RATES = {250: 0x96, 500: 0x95, 1000: 0x94, 2000: 0x93, 4000: 0x92, 8000: 0x91}
HEADER = b"\xaa\xbb\xcc\xdd"
FRAMES_PER_PACKET = 50
UV_PER_COUNT = 4.5 / 24 / 8388607 * 1e6


class PacketError(ValueError):
    """Malformed or inconsistent packet."""


class CRCError(PacketError):
    """Packet checksum does not match."""


@dataclass(frozen=True)
class Packet:
    sequence: int
    samples: np.ndarray
    status: np.ndarray


def crc16(data: bytes) -> int:
    return binascii.crc_hqx(data, 0xFFFF)


def parse_packet(payload: bytes, channels: int) -> Packet:
    if not isinstance(channels, int) or not 1 <= channels <= 8:
        raise ValueError("channels must be between 1 and 8")
    expected = 10 + FRAMES_PER_PACKET * (3 + channels * 3)
    if len(payload) != expected:
        raise PacketError(f"Expected {expected} bytes, received {len(payload)}")
    if payload[:4] != HEADER:
        raise PacketError("Invalid EEG packet header")
    if crc16(payload[4:-2]) != int.from_bytes(payload[-2:], "big"):
        raise CRCError("EEG packet CRC mismatch")
    sequence = struct.unpack_from(">I", payload, 4)[0]
    frames = np.frombuffer(payload, dtype=np.uint8, offset=8, count=expected - 10)
    frames = frames.reshape(FRAMES_PER_PACKET, 3 + 3 * channels)
    codes = frames[:, 3:].reshape(FRAMES_PER_PACKET, channels, 3).astype(np.int32)
    raw = (codes[:, :, 0] << 16) | (codes[:, :, 1] << 8) | codes[:, :, 2]
    raw = np.where(raw & 0x800000, raw - 0x1000000, raw)
    return Packet(sequence, raw.astype(np.float64) * UV_PER_COUNT, frames[:, :3].copy())


def encode_packet(sequence: int, raw_samples: np.ndarray, status=None) -> bytes:
    """Create a test packet from integer ADC counts, never from microvolts."""
    raw = np.asarray(raw_samples)
    if raw.ndim != 2 or raw.shape[0] != 50 or not 1 <= raw.shape[1] <= 8:
        raise ValueError("raw_samples must have shape (50, 1..8)")
    if raw.dtype.kind not in "iu" or np.any(raw < -8388608) or np.any(raw > 8388607):
        raise ValueError("Samples must be signed 24-bit integer ADC counts")
    if not 0 <= sequence <= 0xFFFFFFFF:
        raise ValueError("Sequence must fit an unsigned 32-bit integer")
    values = raw.astype(np.int64) & 0xFFFFFF
    encoded = np.stack(((values >> 16) & 255, (values >> 8) & 255, values & 255), axis=2)
    statuses = np.zeros((50, 3), dtype=np.uint8) if status is None else np.asarray(status)
    if statuses.shape != (50, 3) or statuses.dtype.kind not in "iu" or np.any(statuses < 0) or np.any(statuses > 255):
        raise ValueError("status must contain 50 three-byte status fields")
    frames = np.concatenate((statuses.astype(np.uint8), encoded.reshape(50, -1).astype(np.uint8)), axis=1)
    body = sequence.to_bytes(4, "big") + frames.tobytes()
    return HEADER + body + crc16(body).to_bytes(2, "big")
