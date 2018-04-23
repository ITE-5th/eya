import configparser
import socket
import threading
import time

import RPi.GPIO as GPIO
import speech_recognition as sr

# from rp_client.speaker import Speaker
# from rp_client.speaker import SpeakersModel
from misc.connection_helper import ConnectionHelper
from rp_client.TTS import TTS
from rp_client.camera import Camera
from rp_client.ocr import OCR
from rp_client.recognizer import Recognizer

Running = True
# MESSAGE TYPES
VQA = 'visual-question-answering'
START_FACE = 'start-face-recognition'
REGISTER_FACE = 'register-face-recognition'
IMAGE_TO_TEXT = 'image-to-text'
OCR = 'OCR'
FACE_RECOGNITION = 'face-recognition'
REMOVE_PERSON = 'remove-person'
ADD_PERSON = 'add-person'
UNKNOWN = 'Unknown'


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

                        self.data_callback(data_id=ADD_PERSON)
                        # images count per user
                        images += 1
                        if images >= 10:
                            self.last_person = None
                            images = 0
                    time.sleep(1)
                time.sleep(0.05)
        except Exception as e:
            print('\033[93m' + 'capture thread stopped' + '\033[0m')
            print(e)
            GPIO.cleanup()

    def start(self):
        global Running
        try:
            self.socket.connect((self.host, self.port))
            print('connected to server ' + self.host + ':' + str(self.port))
            if self.configParser.get('user-data', 'id') == "":
                # First Run
                self.data_callback(data_id=REGISTER_FACE)
                self.tts.say('Please Say your Name .')
                self.speaker_name = self.speech_to_text(self.id + '.wav', mic=True)
                self.write_config('id', self.id)
                self.write_config('u_name', self.speaker_name)
            else:

                self.id = self.configParser.get('user-data', 'id')
                self.speaker_name = self.configParser.get('user-data', 'u_name')
                self.data_callback(data_id=START_FACE)

            self.tts.say('Welcome ' + self.speaker_name)

            capture_handler = threading.Thread(
                target=self.handle_capture_button,
            )

            capture_handler.start()
            # start recogniser
            self.recognizer.start()
        finally:
            Running = False
            print('closing camera')
            self.cam.close()
            ConnectionHelper.send_json(self.socket, {'type': 'close'})
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
        if data_id in [VQA, 'set-last-person', REMOVE_PERSON]:
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

    def speech_to_text(self, fname=None, mic=False):
        r = sr.Recognizer()

        if mic:
            print('Recording')
            with sr.Microphone() as source:
                audio = r.listen(source)
                # write audio to a WAV file
                with open(fname, "wb") as f:
                    f.write(audio.get_wav_data())
        else:
            with sr.AudioFile(fname) as source:
                audio = r.record(source)  # read the entire audio file

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

    def close(self):
        self.socket.close()

    def communicate_with_server(self, message):
        if message['type'] == OCR:
            response = OCR.get_text()
        else:
            ConnectionHelper.send_json(self.socket, message)
            response = ConnectionHelper.receive_json(self.socket)
        if message['type'] == REGISTER_FACE:
            self.id = response['result']
            # register face response
            print('registering')
        print(response)
        self.say_message(response['result'], message['type'])

    def write_config(self, field, data):
        self.configParser.set('user-data', field, data)
        with open('config.ini', 'w') as f:
            self.configParser.write(f)

    def say_message(self, response, m_type):
        phrase = 'we recognise . '
        print(response)
        print(m_type)
        if response.strip() != '':
            if m_type == FACE_RECOGNITION:
                persons = response.split(',')
                unk_count = sum([x.split(' ').count(UNKNOWN) for i, x in enumerate(persons)])

                # remove Unknown
                persons = [x for i, x in enumerate(persons) if x.split(' ')[0] != UNKNOWN]
                for idx, person in enumerate(persons):
                    person = person.split(' ')
                    persons[idx] = person[0].replace('_', ' ').title()
                    persons[idx] += ' With probability of {} Percent . '.format(person[1])

                if unk_count > 0:
                    persons.append(str(unk_count) + ' Unknown persons . ')

                persons_count = len(persons)
                for i in range(persons_count):
                    phrase += persons[i]
                    phrase += ' And ' if i == persons_count - 2 and persons_count > 1 else ''

            elif m_type == IMAGE_TO_TEXT or m_type == OCR:
                phrase += response
            elif m_type == VQA:
                answers = response.split(',')
                answers_count = len(answers)
                for i in range(answers_count):
                    phrase += answers[i].capitalize() + (
                        ' . Or ' if i == answers_count - 2 and answers_count > 1 else ' . ')
            self.tts.say(phrase)

    def _build_message(self, type, text_from_speech=None):

        json_data = {
            "type": type,
        }
        image_file = None
        if type == ADD_PERSON:
            image_file = self.cam.take_image(face_count=1)
            if image_file == -1:
                print('Sorry,Please Take a new Image.')
                self.tts.say('Sorry  Please Take a new Image.')
                return None
            json_data["name"] = self.last_person

        elif type == REGISTER_FACE or type == START_FACE:
            json_data["name"] = self.id

        elif type == REMOVE_PERSON:
            json_data["name"] = text_from_speech

        elif type == IMAGE_TO_TEXT or \
                type == VQA or \
                type == FACE_RECOGNITION:
            image_file = self.cam.take_image()

        json_data["image"] = image_file

        if text_from_speech is not None:
            json_data["question"] = text_from_speech

        return json_data


appos = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "I would",
    "i'll": "I will",
    "i'm": "I am",
    "isn't": "is not",
    "it's": "it is",
    "it'll": "it will",
    "i've": "I have",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "weren't": "were not",
    "we've": "we have",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "'re": " are",
    "wasn't": "was not",
    "we'll": " will"
}


def to_uniform(word):
    words = word.split()
    reformed = [appos[word] if word in appos else word for word in words]
    return " ".join(reformed)


if __name__ == '__main__':
    api = ClientAPI(host='192.168.1.4')
    try:
        api.start()
    finally:
        api.close()
