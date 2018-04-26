import argparse
import configparser
import os
import socket
import threading
import time

import RPi.GPIO as GPIO
import speech_recognition as sr

# from rp_client.speaker import Speaker
# from rp_client.speaker import SpeakersModel
from misc.connection_helper import ConnectionHelper
from misc.text_normalizer import to_uniform
from rp_client.TTS import TTS
from rp_client.camera import Camera
from rp_client.keypad import Keypad
from rp_client.ocr import OCR
from rp_client.recognizer import Recognizer
from server.message.add_person_message import AddPersonMessage
from server.message.close_message import CloseMessage
from server.message.end_add_person_message import EndAddPersonMessage
from server.message.face_recognition_message import FaceRecognitionMessage
from server.message.image_to_text_message import ImageToTextMessage
from server.message.ocr_message import OcrMessage
from server.message.register_face_recognition_message import RegisterFaceRecognitionMessage
from server.message.remove_person_message import RemovePersonMessage
from server.message.start_face_recognition_message import StartFaceRecognitionMessage
from server.message.vqa_message import VqaMessage

Running = True
# MESSAGE TYPES
END_ADD_PERSON = 'end-add-person'
VQA = 'visual-question-answering'
IMAGE_TO_TEXT = 'image-to-text'
OCR_MSG = 'ocr'
FACE_RECOGNITION = 'face-recognition'
REMOVE_PERSON = 'remove-person'
ADD_PERSON = 'add-person'
UNKNOWN = 'Unknown'
SET_LAST_PERSON = 'set-last-person'

# user identify
START_FACE = 'start-face-recognition'
REGISTER_FACE = 'register-face-recognition'


class ClientAPI:
    def __init__(self, host=socket.gethostname(), port=8888, configFilePath=r'./config.ini'):
        self.host = host
        self.port = port
        self.speaker_name = 'Default'
        self.id = 'Default'
        self.cam = Camera(width=800, height=600)
        self.tts = TTS(festival=False, espeak=False, pico=True)
        self.recognizer = Recognizer(server=self, callback_function=self.data_callback)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.last_person = None
        self.configParser = configparser.RawConfigParser()
        self.configParser.read(configFilePath)
        self.last_msg = None
        self.keypad_client = Keypad()
        self._images_counter = 0

    def handle_capture_button(self):
        global Running
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        try:
            while Running:
                button_state = GPIO.input(23)
                if not button_state:
                    self.add_person()
                    time.sleep(1)
                time.sleep(0.05)
        except Exception as e:
            print('\033[93m' + 'capture thread stopped' + '\033[0m')
            print(e)
            GPIO.cleanup()

    def add_person(self):
        if self.last_person is None:
            success = self.data_callback(data_id=SET_LAST_PERSON)
        else:
            self.tts.say(f'image number is . {self._images_counter} . for user . {self.last_person} . ')
            success = self.data_callback(data_id=ADD_PERSON)
            if success:
                self.tts.say(f'image successfully added . ')

                # images count per user
                if self._images_counter >= 10:
                    self.data_callback(data_id=END_ADD_PERSON)
                    self.last_person = None
                    self._images_counter = 0

                self._images_counter += 1

    def start(self):
        global Running
        try:
            self.socket.connect((self.host, self.port))
            print('connected to server ' + self.host + ':' + str(self.port))
            if self.configParser.get('user-data', 'id') == "":
                # First Run
                self.data_callback(data_id=REGISTER_FACE)
                self.tts.say('Please Say your Name .')
                self.speaker_name = self.speech_to_text('./temp/' + self.id + '.wav', mic=True)
                self.write_config('id', self.id)
                self.write_config('u_name', self.speaker_name)
            else:

                self.id = self.configParser.get('user-data', 'id')
                self.speaker_name = self.configParser.get('user-data', 'u_name')
                self.data_callback(data_id=START_FACE)

            self.tts.say(f'Welcome {self.speaker_name}')

            capture_handler = threading.Thread(
                target=self.handle_capture_button,
            )
            self.keypad_client.start(self.keypad_callback)

            capture_handler.start()
            # start recogniser
            self.recognizer.start()
        finally:
            self.tts.say('System Shutdown . Good bye')
            Running = False
            self.recognizer.set_interrupted(True)
            print('closing camera')
            self.keypad_client.cleanup()
            self.cam.close()
            ConnectionHelper.send_pickle(self.socket, CloseMessage())
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
        if data_id in [VQA, SET_LAST_PERSON, REMOVE_PERSON]:
            # verify speaker
            print("converting audio to text")
            speech = self.speech_to_text(fname)
            if data_id == SET_LAST_PERSON:
                if speech != '':
                    self.last_person = speech
                    self.tts.say(f'selected user is . {speech}')
                    return True
            else:
                message = self._build_message(data_id, text_from_speech=speech)
            # os.remove(fname)
        else:
            message = self._build_message(data_id)

        if message is not None:
            return self.communicate_with_server(message)
        return False

    def get_speaker(self, fname):
        # Speaker() used for import speaker class only
        # Speaker(name='test')
        # model: SpeakersModel = SpeakersModel.load("models/gmms.model")
        # return model.verify_speaker(fname, self.speaker_name.title())
        return 1

    def speech_to_text(self, fname=None, mic=False):
        r = sr.Recognizer()
        if fname is None:
            fname = './temp/mic_' + time.strftime("%Y%m%d-%H%M%S") + '.wav'
            mic = True
        if mic:
            self.recognizer.pause()
            print('Recording')
            with sr.Microphone() as source:
                audio = r.listen(source)
                # write audio to a WAV file
                with open(fname, "wb") as f:
                    f.write(audio.get_wav_data())
            self.recognizer.resume()

        else:
            with sr.AudioFile(fname) as source:
                audio = r.record(source)  # read the entire audio file
        threshold = 0.5

        if self.get_speaker(fname) > threshold:
            try:
                # print("Sphinx thinks you said " + r.recognize_sphinx(audio))
                print("")
            except sr.UnknownValueError:
                print("Sphinx could not understand audio")

            except sr.RequestError as e:
                print("Sphinx error; {0}".format(e))
            # recognize speech using Google Speech Recognition
            googleSTT = ''
            try:
                googleSTT = r.recognize_google(audio)
                print(googleSTT)
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
            return to_uniform(googleSTT)
        else:
            print('speaker is not verified')
            return ''

    def close(self):
        self.socket.close()

    def communicate_with_server(self, message):
        if self.last_msg == OCR_MSG:
            response = OCR.get_text(message.image)
        else:
            ConnectionHelper.send_pickle(self.socket, message)
            response = ConnectionHelper.receive_json(self.socket)

        if self.last_msg == REGISTER_FACE:
            self.id = response['result']
            # register face response
            print('registering')
        print(response)
        self.say_message(response['result'])
        return True

    def write_config(self, field, data):
        self.configParser.set('user-data', field, data)
        with open('config.ini', 'w') as f:
            self.configParser.write(f)

    def say_message(self, response):
        phrase = 'we recognise . '
        print(response)
        if response.strip() != '':
            if self.last_msg == FACE_RECOGNITION:
                persons = response.split(',')
                unk_count = sum([x.split(' ').count(UNKNOWN) for i, x in enumerate(persons)])

                # remove Unknown
                persons = [x for i, x in enumerate(persons) if x.split(' ')[0] != UNKNOWN]
                for idx, person in enumerate(persons):
                    person = person.split(' ')
                    persons[idx] = person[0].replace('_', ' ').title()
                    persons[idx] += ' . '
                    # persons[idx] += ' With probability of {} Percent . '.format(person[1])

                if unk_count > 0:
                    persons.append(str(unk_count) + ' Unknown persons . ')

                persons_count = len(persons)
                for i in range(persons_count):
                    phrase += persons[i]
                    phrase += ' And ' if i == persons_count - 2 and persons_count > 1 else ''

            elif self.last_msg == IMAGE_TO_TEXT or self.last_msg == OCR_MSG:
                phrase += response
            elif self.last_msg == VQA:
                answers = response.split(',')
                answers_count = len(answers)
                for i in range(answers_count):
                    phrase += answers[i].capitalize() + (
                        ' . Or ' if i == answers_count - 2 and answers_count > 1 else ' . ')
            else:
                phrase = ''
            self.tts.say(phrase)

    def take_image(self, face_count=0):
        self.tts.say('Taking Photo')
        img = self.cam.take_image(face_count=face_count)
        if img == -1:
            print('Sorry,Please Take a new Image.')
            self.tts.say('Sorry  Please Take a new Image.')
            return None
        return img

    def _build_message(self, type, text_from_speech=None):
        self.last_msg = type
        if type == ADD_PERSON:
            image, _ = self.take_image(face_count=1)
            return AddPersonMessage(image) if image is not None else None

        if type == OCR_MSG:
            _, fname = self.take_image()
            return OcrMessage(fname)

        elif type == END_ADD_PERSON:
            return EndAddPersonMessage(self.last_person)

        elif type == REGISTER_FACE:
            return RegisterFaceRecognitionMessage(self.id)

        elif type == START_FACE:
            return StartFaceRecognitionMessage(self.id)

        elif type == REMOVE_PERSON:
            return RemovePersonMessage(text_from_speech)

        elif type == IMAGE_TO_TEXT:
            image, _ = self.take_image()
            return ImageToTextMessage(image) if image is not None else None

        elif type == VQA:
            image, _ = self.take_image()
            return VqaMessage(image, text_from_speech) if image is not None else None

        elif type == FACE_RECOGNITION:
            image, _ = self.take_image()
            return FaceRecognitionMessage(image) if image is not None else None

        return None

    def keypad_callback(self, key):
        global Running
        print(key)
        callbacks = [
            lambda: self.data_callback(data_id=FACE_RECOGNITION),
            lambda: self.data_callback(data_id=VQA),
            lambda: self.data_callback(data_id=IMAGE_TO_TEXT),
            lambda: self.data_callback(data_id=OCR_MSG),
            lambda: self.add_person(),
            lambda: self.data_callback(data_id=REMOVE_PERSON),
        ]
        if key in range(1, 6):
            print(callbacks[key - 1])
            callbacks[key - 1]()
        elif key == 'A':
            Running = False
            time.sleep(5)
            os.execv('/home/pi/myFolder/RestartMySelf.py', [''])
        elif key == 'B':
            Running = False
            os.system('sudo reboot')
        elif key == 'C':
            Running = False
            os.system('sudo shutdown now')


def main(args):
    api = ClientAPI(host=args.host, port=args.port)
    try:
        api.start()
    finally:
        api.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str,
                        default='192.168.1.4')
    parser.add_argument('--port', type=int,
                        default=8888)
    parser.add_argument('--config', type=str,
                        default='config.ini')

    arguments = parser.parse_args()
    main(arguments)
