import pytest

from sunrisk import contracts as c


def make_packet(mag=True):
    imu = [c.ImuSample((0.01 * i, -0.98, 0.123), (1.5, -20.3, 250.0),
                       (22.5, -5.1, 40.0) if mag else None) for i in range(c.IMU_SAMPLES_PER_PACKET)]
    return c.BlePacket(seq=65535, t_device_ms=123456789, uv_counts=18400, uv_gain=18,
                       uv_resolution_bits=20, battery_mv=3912, uvi_device=8.0, imu=imu,
                       flags=c.FLAG_HAS_MAG if mag else 0)


def test_packet_fits_ios_default_mtu():
    assert c.PACKET_SIZE == 164
    assert c.PACKET_SIZE <= 185 - 3   # ATT MTU 185 minus 3-byte notification header


@pytest.mark.parametrize("mag", [True, False])
def test_packet_roundtrip(mag):
    p = make_packet(mag)
    q = c.BlePacket.decode(p.encode())
    assert (q.seq, q.t_device_ms, q.uv_counts, q.uv_gain, q.uv_resolution_bits, q.battery_mv) == \
           (p.seq, p.t_device_ms, p.uv_counts, p.uv_gain, p.uv_resolution_bits, p.battery_mv)
    assert q.uvi_device == pytest.approx(8.0)
    for a, b in zip(p.imu, q.imu):
        assert a.acc_g == pytest.approx(b.acc_g, abs=1e-3)
        assert a.gyro_dps == pytest.approx(b.gyro_dps, abs=0.05)
        if mag:
            assert a.mag_ut == pytest.approx(b.mag_ut, abs=0.05)
        else:
            assert b.mag_ut is None


def test_packet_rejects_bad_input():
    p = make_packet()
    with pytest.raises(ValueError):
        c.BlePacket.decode(p.encode()[:-1])
    p.uv_gain = 2
    with pytest.raises(ValueError):
        p.encode()


def test_imu_values_saturate_not_wrap():
    p = make_packet()
    p.imu[0] = c.ImuSample((100.0, 0, 0), (0, 0, 0))   # 100 g exceeds int16 milli-g range
    assert c.BlePacket.decode(p.encode()).imu[0].acc_g[0] == pytest.approx(32.767)
