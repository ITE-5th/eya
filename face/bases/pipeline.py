from typing import List

from face.bases.stage import Stage
from face.misc.time_logger import TimeLogger


class Pipeline(object):
    def __init__(self, stages: List[Stage]) -> None:
        super().__init__()
        self.stages = stages

    def __call__(self, inputs, verbose: bool = False):
        output = inputs

        if verbose:
            logger = TimeLogger()

        for i, stage in enumerate(self.stages):

            output = stage(output, verbose=verbose)
            if output[0] is None or len(output[0]) == 0 or output[1] is None:
                break

        if verbose:
            logger.log()
            print("Pipeline: {} stages, time {}".format(i + 1, logger))
        if len(output) < 3:
            output = output[0], output[1], None
        return output
