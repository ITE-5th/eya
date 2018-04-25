import pytesseract
from PIL import Image


class OCR:
    @staticmethod
    def get_text(image_path, lang='eng'):
        result = ''
        try:
            if image_path is not None:
                result = pytesseract.image_to_string(Image.open(image_path), lang=lang)
        except Exception as e:
            print(e)
        finally:
            return {'result': result}
