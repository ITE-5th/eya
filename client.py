import socket

import speech_recognition as sr

# from rp_client.speaker import Speaker
# from rp_client.speaker import SpeakersModel
from helper import Helper
from rp_client.TTS import TTS
from rp_client.camera import Camera
from rp_client.ocr import OCR
from rp_client.recognizer import Recognizer


class ClientAPI:
    def __init__(self, speaker_name, host=socket.gethostname(), port=1231):
        self.host = host
        self.port = port
        self.speaker_name = speaker_name
        self.cam = Camera()
        self.tts = TTS(festival=False, espeak=False, pico=True)
        self.recognizer = Recognizer(server=self)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def start(self):

        self.socket.connect((self.host, self.port))
        print('connected to server ' + self.host + ':' + str(self.port))
        #     start recogniser
        self.recognizer.start(self.audio_recorder_callback)

    def audio_recorder_callback(self, fname=None, hotword_id=None):
        message = None
        if hotword_id == 'vqa':
            # verify speaker
            threshold = 0.5
            if self.get_speaker(fname) > threshold:
                print("converting audio to text")
                speech = self.speech_to_text(fname)
                message = self._build_message(hotword_id, question=speech)
                # os.remove(fname)
            else:
                print('speaker is not verified')
        else:
            message = self._build_message(hotword_id)

        if message is not None:
            self.communicate_with_server(message)

    def get_speaker(self, fname):
        # Speaker() used for import speaker class only
        # Speaker(name='test')
        # model: SpeakersModel = SpeakersModel.load("models/gmms.model")
        # return model.verify_speaker(fname, self.speaker_name.title())
        return 1

    def speech_to_text(self, fname):
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

    def _build_message(self, type, question=None):
        # type == "visual-question-answering"
        # type == "face-recognition"
        # type == "image-to-text"
        if (type == 'capture_face'):
            image_file = self.cam.take_image(face_count=1)

            if (image_file == -1):
                print('Sorry,Please Take a new Image.')
                self.tts.say('Sorry  Please Take a new Image.')
                return None;
        else:
            image_file = self.cam.take_image()
        json_data = {
            "type": type,
            "image": image_file,
        }
        if question is not None:
            json_data["question"] = question
        # print('json_data')
        # print(json_data)
        return json_data


if __name__ == '__main__':
    api = ClientAPI(speaker_name='zaher', host='192.168.1.3')
    try:
        api.start()
    finally:
        api.close()
