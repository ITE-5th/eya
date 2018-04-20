#
# class ClosedPredictor:
#     pipeline = Pipeline([
#         DLibDetector(scale=1),
#         OneMillisecondAligner(224),
#         VggExtractor()
#     ])
#
#     def __init__(self, model_path):
#         self.model_path = model_path
#         self.reload()
