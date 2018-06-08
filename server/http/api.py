from flask import Flask, request
from flask_json import FlaskJSON, json_response

from face_recognition_model import FaceRecognitionModel
from image_to_text_model import ImageToTextModel
from misc.converter import Converter
from server.http.face_model_map import FaceModelMap
from server.http.route_names import Names
from vqa_model import VqaModel

vqa_model = VqaModel()
itt_model = ImageToTextModel()
face_models = FaceModelMap()

app = Flask(__name__)
FlaskJSON(app)

PORT = 9999


@app.route(Names.VQA_ROUTE, methods=["post", "put"])
def vqa():
    data = request.get_json(force=True)
    image = Converter.to_image(data["image"])
    result = vqa.predict(data["question"], image)
    return json_response(result=result)


@app.route(Names.ITT_ROUTE, methods=["post", "put"])
def itt():
    data = request.get_json(force=True)
    image = Converter.to_image(data["image"])
    result = itt_model.predict(image)
    return json_response(result=result)


common_face_route = f"{Names.FACE_RECOGNITION_ROUTE}/<user_name>"
common_target_face_route = f"{common_face_route}/<target_name>"


@app.route(common_face_route, methods=["post"])
def register_face_recognition(user_name):
    FaceRecognitionModel.register(user_name, remove_dir=False)
    return json_response(result="success", registered=True)


@app.route(common_face_route, methods=['head'])
def start_face_recognition(user_name):
    model = FaceRecognitionModel(user_name)
    face_models[user_name] = model, []
    return json_response(result="success")


@app.route(common_face_route, methods=["post"])
def face_recognition(user_name):
    model, _ = face_models[user_name]
    data = request.get_json(force=True)
    image = Converter.to_image(data["image"])
    result = model.predict(image)
    return json_response(result=result)


@app.route(common_target_face_route, methods=["delete"])
def remove_person(user_name, target_name):
    try:
        model, _ = face_models[user_name]
        model.remove_person(target_name)
        return json_response(result="success")
    except:
        return json_response(result="error")


@app.route(common_target_face_route, methods=["put"])
def add_face(user_name):
    model, images = face_models[user_name]
    data = request.get_json(force=True)
    image = Converter.to_image(data["image"])
    images.append(image)
    face_models[user_name] = model, images
    return json_response(result="success")


@app.route(common_target_face_route, methods=["post"])
def end_add_face(user_name, target_name):
    model, images = face_models[user_name]
    model.add_person(target_name, images)
    face_models[user_name] = model, []
    return json_response(result="success")


if __name__ == '__main__':
    app.run(port=PORT)
