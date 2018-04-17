from multiprocessing import cpu_count
from multiprocessing.pool import Pool

import openface

from aligners.base_aligner import BaseAligner
from face_recognition.misc.utils import Utils
from file_path_manager import FilePathManager


class OneMillisecondAligner(BaseAligner):
    def __init__(self, size=224):
        self.size = size
        self.aligner = openface.AlignDlib(
            FilePathManager.resolve("face_recognition/data/shape_predictor_68_face_landmarks.dat"))
        self.image = None

    def align(self, rect):
        return self.aligner.align(self.size, self.image, rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    def forward(self, inputs):
        rects, self.image = inputs

        rects = Utils.points2rects(rects)
        with Pool(cpu_count()) as pool:
            result = pool.map(self.align, rects)

        return result, self.image


if __name__ == '__main__':
    pass
    # FilePathManager.clear_dir("output")
    #
    # path = FilePathManager.resolve("output")
    # faces = sorted(glob.glob(FilePathManager.resolve("images/*")))
    #
    # pipeline = Pipeline([DLibDetector(), OneMillisecondAligner()])
    #
    # for i, face in enumerate(faces):
    #     face = cv2.imread(face)
    #
    #     cropped_output, image = pipeline(face)
    #     for j, cropped_image in enumerate(cropped_output):
    #         cv2.imwrite(path + "/test{}-{}.jpeg".format(i, j), cropped_image)
