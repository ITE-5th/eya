import os
import time


class TTS:
    def __init__(self, festival=True, espeak=True, pico=True):
        self.Festival = festival
        self.Espeak = espeak
        self.Pico = pico

    def say(self, message):
        if self.Festival:
            print('Festival Text to Speech')
            os.system('echo "' + message + '" | festival --tts')

        if self.Espeak:
            print('Espeak Text to Speech')
            os.system('espeak -ven+f3 -k5 -s150 "' + message + '"')

        if self.Pico:
            print('Pico Text to Speech')
            fname = time.strftime("%Y%m%d-%H%M%S") + '.wav'
            cmd = 'pico2wave -w ' + fname + ' "' + message + '" && aplay ' + fname + '.wav'
            os.system(cmd)
            # os.remove(fname)


if __name__ == '__main__':
    TTS(festival=False, espeak=False, pico=True).say('Hello World')
