import time
from smbus2 import SMBus


class SHT31:
    CMD_MEASURE = (0x2C, 0x06)  # single shot, high repeatability, clock stretching

    def __init__(self, bus_id: int = 1, address: int = 0x44):
        self.bus_id = bus_id
        self.address = address

    def _crc8(self, data) -> int:
        crc = 0xFF
        for b in data:
            crc ^= b
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ 0x31) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        return crc

    def read(self):
        with SMBus(self.bus_id) as bus:
            bus.write_i2c_block_data(self.address, self.CMD_MEASURE[0], [self.CMD_MEASURE[1]])
            time.sleep(0.05)
            raw = bus.read_i2c_block_data(self.address, 0x00, 6)

        t_data = raw[0:2]
        t_crc = raw[2]
        rh_data = raw[3:5]
        rh_crc = raw[5]

        if self._crc8(t_data) != t_crc or self._crc8(rh_data) != rh_crc:
            raise ValueError("CRC check failed")

        t_raw = (t_data[0] << 8) | t_data[1]
        rh_raw = (rh_data[0] << 8) | rh_data[1]

        temp_c = -45 + 175 * (t_raw / 65535.0)
        humidity_rh = 100 * (rh_raw / 65535.0)
        humidity_rh = max(0.0, min(100.0, humidity_rh))

        return temp_c, humidity_rh
