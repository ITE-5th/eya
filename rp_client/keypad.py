import time

from pad4pi import rpi_gpio


class Keypad:
    def __init__(self, keys=None, row_pins=None, col_pins=None):

        # Setup Keypad
        if keys is None:
            self.KEYS = [
                [1, 2, 3, "A"],
                [4, 5, 6, "B"],
                [7, 8, 9, "C"],
                ["*", 0, "#", "D"]
            ]
        if row_pins is None:
            self.row_pins = [5, 6, 13, 19]  # BCM numbering
        if col_pins is None:
            self.col_pins = [26, 16, 20, 21]  # BCM numbering
        factory = rpi_gpio.KeypadFactory()
        self.keypad = factory.create_keypad(keypad=self.KEYS, row_pins=self.row_pins, col_pins=self.col_pins)

    def start(self, callback_fn):
        self.keypad.registerKeyPressHandler(callback_fn)

    def cleanup(self):
        self.keypad.cleanup()


def print_key(key):
    print(key)


if __name__ == '__main__':
    keypad_client = Keypad()
    keypad_client.start(print_key)
    try:
        while True:
            time.sleep(0.1)
    finally:
        keypad_client.cleanup()
