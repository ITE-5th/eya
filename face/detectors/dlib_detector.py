import dlib

from face.detectors.base_detector import BaseDetector
from face.misc.utils import Utils
from file_path_manager import FilePathManager


class DLibDetector(BaseDetector):

    def __init__(self, scale=1, use_cnn=False):
        super().__init__()
        self.scale = scale
        self.use_cnn = use_cnn
        self.detector = dlib.get_frontal_face_detector() if not use_cnn else dlib.cnn_face_detection_model_v1(
            FilePathManager.resolve("face/data/mmod_human_face_detector.dat"))

    def forward(self, image):
        temp = self.detector(image, self.scale)
        items = temp if not self.use_cnn else [item.rect for item in temp]
        return items, image

    def postprocess(self, inputs):
        items, image = inputs
        return Utils.rects2points(items), image


if __name__ == '__main__':
    pass
    # FilePathManager.clear_dir("output")
    #
    # path = FilePathManager.resolve("output")
    # faces = sorted(glob.glob(FilePathManager.resolve("faces/*/*")))
    #
    # pipeline = Pipeline([DLibDetector(scale=1), Scale(0.2), Crop()])
    #
    # for i, face in enumerate(faces):
    #     print(face)
    #     face = cv2.imread(face)
    #     cropped_output, _ = pipeline(face)
    #     for j, cropped_image in enumerate(cropped_output):
    #         cv2.imwrite(path + "/test{}-{}.jpeg".format(i, j), cropped_image)
