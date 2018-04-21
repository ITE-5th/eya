from server.handler import Handler


class ImageToTextHandler:
    class __ImageToTextHandler(Handler):

        def __init__(self, image_to_text):
            self.image_to_text = image_to_text

        def handle(self, image, question, type, name):
            return {"result": self.image_to_text.predict(image)}

    instance = None

    def __init__(self, image_to_text):
        if not ImageToTextHandler.instance:
            ImageToTextHandler.instance = ImageToTextHandler.__ImageToTextHandler(image_to_text)
        else:
            ImageToTextHandler.instance.image_to_text = image_to_text

    def __getattr__(self, name):
        return getattr(self.instance, name)
