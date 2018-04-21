from server.handler import Handler


class VqaHandler:
    class __VqaHandler(Handler):
        def __init__(self, vqa):
            self.vqa = vqa

        def handle(self, image, question, type, name):
            return {
                "result": self.vqa.predict(question, image)
            }

    instance = None

    def __init__(self, vqa):
        if not VqaHandler.instance:
            VqaHandler.instance = VqaHandler.__VqaHandler(vqa)
        else:
            VqaHandler.instance.vqa = vqa

    def __getattr__(self, name):
        return getattr(self.instance, name)
