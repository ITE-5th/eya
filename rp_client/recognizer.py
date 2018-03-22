import signal

from rp_client import snowboydecoder

interrupted = False


class Recognizer:
    def __init__(self, pmdl_path=None, sensitivity=0.38, sphinx=True, google=True, server=None):
        if pmdl_path is None:
            pmdl_path = [
                './client/resources/models/question.pmdl',
                './client/resources/models/face.pmdl',
                './client/resources/models/caption.pmdl',
                './client/resources/models/snowboy.umdl',
                './client/resources/models/alexa_02092017.umdl',
            ]

        model = pmdl_path
        # capture SIGINT signal, e.g., Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        self.detector = snowboydecoder.HotwordDetector(model, sensitivity=sensitivity)
        self.sphinx = sphinx
        self.google = google
        self.server = server

    def start(self, callback_function):
        # main loop
        print('Listening... Press Ctrl+C to exit')
        callbacks = [lambda: [snowboydecoder.play_audio_file(snowboydecoder.DETECT_DING)],
                     lambda: [
                         snowboydecoder.play_audio_file(snowboydecoder.DETECT_DONG),
                         callback_function(data_id='caption')
                     ],
                     lambda: [
                         snowboydecoder.play_audio_file(snowboydecoder.DETECT_DONG),
                         callback_function(data_id='ocr')
                     ],
                     lambda: [
                         snowboydecoder.play_audio_file(snowboydecoder.DETECT_DONG),
                         callback_function(data_id='face')
                     ],
                     lambda: [
                         snowboydecoder.play_audio_file(snowboydecoder.DETECT_DONG),
                         callback_function(data_id='capture_face')
                     ]
                     ]

        acallbacks = [lambda fname: callback_function(fname, data_id='vqa'), None, None, None, None]

        self.detector.start(detected_callback=callbacks,
                            audio_recorder_callback=acallbacks,
                            interrupt_check=self.interrupt_callback,
                            sleep_time=0.01)

        self.detector.terminate()

    def signal_handler(self, signal, frame):
        global interrupted
        interrupted = True

    def interrupt_callback(self):
        global interrupted
        return interrupted


if __name__ == '__main__':
    r = Recognizer()
    r.start()
