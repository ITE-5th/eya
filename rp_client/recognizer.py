import signal

from rp_client import snowboydecoder

interrupted = False
Running = True


class Recognizer:
    def __init__(self, pmdl_path=None, sensitivity=0.38, sphinx=True, google=True, server=None, callback_function=None):
        if pmdl_path is None:
            pmdl_path = [
                # start listening
                './rp_client/resources/models/alexa.umdl',
                # visual question answering
                './rp_client/resources/models/question.pmdl',
                # face recognition
                './rp_client/resources/models/face.pmdl',
                # image captioning
                './rp_client/resources/models/caption.pmdl',
                # OCR
                './rp_client/resources/models/snowboy.umdl',
                # new person
                './rp_client/resources/models/alexa_02092017.umdl',
                # remove person
                './rp_client/resources/models/smart_mirror.umdl'
            ]

        model = pmdl_path
        # capture SIGINT signal, e.g., Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        self.start_detector = snowboydecoder.HotwordDetector(model[0], sensitivity=sensitivity)
        self.detector = snowboydecoder.HotwordDetector(model[1:], sensitivity=sensitivity)
        self.sphinx = sphinx
        self.google = google
        self.server = server
        self.callback_function = callback_function

    def start(self):
        global Running
        callbacks = [lambda: [snowboydecoder.play_audio_file(snowboydecoder.DETECT_DING)],
                     lambda: [
                         snowboydecoder.play_audio_file(snowboydecoder.DETECT_DONG),
                         self.callback_function(data_id='face-recognition'),
                         self.set_interrupted(value=False)

                     ],
                     lambda: [
                         snowboydecoder.play_audio_file(snowboydecoder.DETECT_DONG),
                         self.callback_function(data_id='image-to-text'),
                         self.set_interrupted(value=False)

                     ],
                     lambda: [
                         snowboydecoder.play_audio_file(snowboydecoder.DETECT_DONG),
                         self.callback_function(data_id='ocr'),
                         self.set_interrupted(value=False)

                     ],
                     # set-last-person
                     lambda: [
                         snowboydecoder.play_audio_file(snowboydecoder.DETECT_DING),
                         self.set_interrupted(value=False)

                     ],
                     # remove-person
                     lambda: [
                         snowboydecoder.play_audio_file(snowboydecoder.DETECT_DING),

                     ]
                     ]

        acallbacks = [
            lambda fname: [
                self.callback_function(fname, data_id='visual-question-answering'),
                self.set_interrupted(value=False)
            ],
            None, None, None,
            lambda fname: [
                self.callback_function(fname, data_id='set-last-person'),
                self.set_interrupted(value=False)
            ],
            lambda fname: [
                self.callback_function(fname, data_id='remove-person')],
            self.set_interrupted(value=False)
        ]

        # main loop
        while Running:
            print('\033[93m' + 'Listening... ' + '\033[0m')

            self.start_detector.start(detected_callback=[
                lambda: [self.start_detected_callback(), snowboydecoder.play_audio_file(snowboydecoder.DETECT_DING)]],
                                      interrupt_check=self.interrupt_start_callback,
                                      sleep_time=0.01)

            self.start_detector.terminate()
            self.detector.start(detected_callback=callbacks,
                                audio_recorder_callback=acallbacks,
                                interrupt_check=self.interrupt_callback,
                                sleep_time=0.01)
            self.detector.terminate()

    def start_detected_callback(self):
        global interrupted
        self.set_interrupted(value=True)

    def signal_handler(self, signal, frame):
        global Running
        print(Running)
        Running = False

    def set_interrupted(self, value=None):
        global interrupted
        interrupted = not interrupted if value is None else value

    def interrupt_callback(self):
        global interrupted, Running
        return not Running or not interrupted

    def interrupt_start_callback(self):
        global interrupted, Running
        return not Running or interrupted


if __name__ == '__main__':
    r = Recognizer()
    r.start()
