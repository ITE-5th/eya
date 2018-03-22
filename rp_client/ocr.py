import pytesseract
from PIL import Image


class OCR:
    @staticmethod
    def get_text(image_path='client/temp/Image.jpg', lang='eng'):
        return {
            'result': pytesseract.image_to_string(Image.open(image_path), lang=lang)
        }
