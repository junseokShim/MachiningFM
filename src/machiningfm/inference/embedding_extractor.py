from machiningfm.inference.predictor import MachiningPredictor


class EmbeddingExtractor:
    def __init__(self, predictor: MachiningPredictor) -> None:
        self.predictor = predictor

    def __call__(self, sample: dict) -> list[float]:
        return self.predictor.embed(sample)
