import socket
import threading
import time

import RPi.GPIO as GPIO
import speech_recognition as sr

# from rp_client.speaker import Speaker
# from rp_client.speaker import SpeakersModel
from helper import Helper
from rp_client.TTS import TTS
from rp_client.camera import Camera
from rp_client.ocr import OCR
from rp_client.recognizer import Recognizer

Running = True


class ClientAPI:
    def __init__(self, speaker_name, host=socket.gethostname(), port=8888):
        self.host = host
        self.port = port
        self.speaker_name = speaker_name
        self.cam = Camera()
        self.tts = TTS(festival=False, espeak=False, pico=True)
        self.recognizer = Recognizer(server=self, callback_function=self.data_callback)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.last_person = None

    def handle_capture_button(self):
        global Running
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        try:
            images = 0
            while Running:
                button_state = GPIO.input(23)
                if not button_state:
                    if self.last_person is None:
                        print('please say add person')
                    else:
                        self.data_callback(data_id='add-person')
                        # images count per user
                        images += 1
                        if images >= 10:
                            self.last_person = None
                            images = 0
                    time.sleep(1)
        except Exception as e:
            print('\033[93m' + 'capture thread stopped' + '\033[0m')
            print(e)
            GPIO.cleanup()

    def start(self):
        global Running
        try:
            self.socket.connect((self.host, self.port))
            print('connected to server ' + self.host + ':' + str(self.port))
            self.data_callback(data_id='register-face-recognition')
            capture_handler = threading.Thread(
                target=self.handle_capture_button,
            )
            capture_handler.start()
            #     start recogniser
            self.recognizer.start()
        finally:
            Running = False
            print('closing camera')
            self.cam.close()
            Helper.send_json(self.socket, {'type': 'close'})
            print('closing socket')
            self.socket.close()

    def data_callback(self, fname=None, data_id=None):
        """
        data_callback is called when capture button is pressed
        or when hot-word detected
        :param fname: is recorded audio path after hot-word is detected
                    'currently contains question audio File else None'
        :param data_id: callback message type
        """
        message = None
        if data_id == 'visual-question-answering' or data_id == 'set-last-person':
            # verify speaker
            threshold = 0.5
            if self.get_speaker(fname) > threshold:
                print("converting audio to text")
                speech = self.speech_to_text(fname)
                if data_id == 'set-last-person':
                    if speech != '':
                        self.last_person = speech
                else:
                    message = self._build_message(data_id, text_from_speech=speech)
                # os.remove(fname)
            else:
                print('speaker is not verified')

        else:
            message = self._build_message(data_id)

        if message is not None:
            print(message)
            self.communicate_with_server(message)

    def get_speaker(self, fname):
        # Speaker() used for import speaker class only
        # Speaker(name='test')
        # model: SpeakersModel = SpeakersModel.load("models/gmms.model")
        # return model.verify_speaker(fname, self.speaker_name.title())
        return 1

    def speech_to_text(self, fname=None):
        r = sr.Recognizer()

        with sr.AudioFile(fname) as source:
            audio = r.record(source)  # read the entire audio file
        # recognize speech using Google Speech Recognition
        try:
            # print("Sphinx thinks you said " + r.recognize_sphinx(audio))
            print("")
        except sr.UnknownValueError:
            print("Sphinx could not understand audio")

        except sr.RequestError as e:
            print("Sphinx error; {0}".format(e))
        googleSTT = ''
        try:
            googleSTT = r.recognize_google(audio)
            print(googleSTT)
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return googleSTT

    def close(self):
        self.socket.close()

    def communicate_with_server(self, message):
        if message['type'] == 'OCR':
            response = OCR.get_text()
        else:
            Helper.send_json(self.socket, message)
            response = Helper.receive_json(self.socket)
        print(response)
        self.tts.say(response['result'])

    def _build_message(self, type, text_from_speech=None):

        json_data = {
            "type": type,
        }
        image_file = None
        if type == 'add-person':
            image_file = self.cam.take_image(face_count=1)
            if image_file == -1:
                print('Sorry,Please Take a new Image.')
                self.tts.say('Sorry  Please Take a new Image.')
                return None
            json_data["name"] = self.last_person

        elif type == 'register-face-recognition':
            json_data["name"] = self.speaker_name

        elif type == "remove-person":
            json_data["name"] = text_from_speech

        elif type == 'image-to-text' or \
                type == 'visual-question-answering' or \
                type == 'face-recognition':
            image_file = self.cam.take_image()

        json_data["image"] = image_file

        if text_from_speech is not None:
            json_data["question"] = text_from_speech

        return json_data


if __name__ == '__main__':
    api = ClientAPI(speaker_name='zaher', host='192.168.1.3')
    try:
        api.start()
    finally:
        api.close()
