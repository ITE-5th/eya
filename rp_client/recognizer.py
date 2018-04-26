import signal
import time

from rp_client import snowboydecoder

interrupted = False
Recognizer_Running = True
Pause = False


class Recognizer:
    def __init__(self, pmdl_path=None, sensitivity=0.38, sphinx=True, google=True, server=None, callback_function=None):
        if pmdl_path is None:
            pmdl_path = [
                # start listening
                './rp_client/resources/models/Open_Sesame.pmdl',
                # visual question answering
                './rp_client/resources/models/question.pmdl',
                # face recognition
                './rp_client/resources/models/face.pmdl',
                # image captioning
                './rp_client/resources/models/caption.pmdl',
                # OCR
                './rp_client/resources/models/Read.pmdl',
                # new person
                './rp_client/resources/models/add_person.pmdl',
                # remove person
                './rp_client/resources/models/remove_person.pmdl'
            ]

        model = pmdl_path
        # capture SIGINT signal, e.g., Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        # self.start_detector = snowboydecoder.HotwordDetector(model[0], sensitivity=sensitivity)
        self.detector = snowboydecoder.HotwordDetector(model[1:], sensitivity=sensitivity)
        self.sphinx = sphinx
        self.google = google
        self.server = server
        self.callback_function = callback_function

    def start(self):
        global Recognizer_Running, Pause
        Recognizer_Running = True
        self.set_interrupted(False)
        callbacks = [lambda: [snowboydecoder.play_audio_file(snowboydecoder.DETECT_DING)],
                     lambda: [
                         snowboydecoder.play_audio_file(snowboydecoder.DETECT_DONG),
                         self.callback_function(data_id='face-recognition'),
                         # self.set_interrupted(value=False)

                     ],
                     lambda: [
                         snowboydecoder.play_audio_file(snowboydecoder.DETECT_DONG),
                         self.callback_function(data_id='image-to-text'),
                         # self.set_interrupted(value=False)

                     ],
                     lambda: [
                         snowboydecoder.play_audio_file(snowboydecoder.DETECT_DONG),
                         self.callback_function(data_id='ocr'),
                         # self.set_interrupted(value=False)

                     ],
                     # set-last-person
                     lambda: [
                         snowboydecoder.play_audio_file(snowboydecoder.DETECT_DING),
                         # self.set_interrupted(value=False)

                     ],
                     # remove-person
                     lambda: [
                         snowboydecoder.play_audio_file(snowboydecoder.DETECT_DING),
                     ]
                     ]

        acallbacks = [
            lambda fname: [
                self.callback_function(fname, data_id='visual-question-answering'),
                # self.set_interrupted(value=False)
            ],
            None, None, None,
            lambda fname: [
                self.callback_function(fname, data_id='set-last-person'),
                # self.set_interrupted(value=False)
            ],
            lambda fname: [
                self.callback_function(fname, data_id='remove-person')],
            # self.set_interrupted(value=False)
        ]

        # main loop

        while Recognizer_Running:
            if Pause:
                time.sleep(0.3)
            else:

                print('\033[93m' + 'Listening... ' + '\033[0m')
                #
                # self.start_detector.start(detected_callback=[
                #     lambda: [self.start_detected_callback(), snowboydecoder.play_audio_file(snowboydecoder.DETECT_DING)]],
                #     interrupt_check=self.interrupt_start_callback,
                #     sleep_time=0.01)
                # self.start_detector.terminate()
                self.detector.start(detected_callback=callbacks,
                                    audio_recorder_callback=acallbacks,
                                    interrupt_check=self.interrupt_callback,
                                    sleep_time=0.01)
                # self.detector.terminate()

    def start_detected_callback(self):
        global interrupted
        self.set_interrupted(value=True)

    def signal_handler(self, signal, frame):
        global Recognizer_Running
        print(Recognizer_Running)
        Recognizer_Running = False

    def pause(self):
        global Pause
        Pause = True
        self.detector.terminate()

    def resume(self):
        global Pause
        Pause = False

    def set_interrupted(self, value=None):
        global interrupted
        interrupted = not interrupted if value is None else value

    def interrupt_callback(self):
        global interrupted, Recognizer_Running
        # return not Recognizer_Running or not interrupted
        return not Recognizer_Running or interrupted

    def interrupt_start_callback(self):
        global interrupted, Recognizer_Running
        return not Recognizer_Running or interrupted


if __name__ == '__main__':
    r = Recognizer()
    r.start()
