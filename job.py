import dataclasses
import uuid


class Job:
    id_uuid: uuid.UUID
    datasets: list
    preprocessing_steps: list
    feature_extraction_method: str
    feature_selection_percent: int

    def __init__(self, datasets: list, preprocessing_steps: list, feature_extraction_method: str, feature_selection_percent: int) -> object:
        self.feature_extraction_method = feature_extraction_method
        self.preprocessing_steps = preprocessing_steps
        self.datasets = datasets
        self.id_uuid = uuid.uuid4()
        self.feature_selection_percent = feature_selection_percent

    def __repr__(self):
        return f"id_uuid = {str(self.id_uuid)} \n" \
               f"datasets: {self.datasets} \n" \
               f"preprocessing: {self.preprocessing_steps} \n" \
               f"feature extraction: {self.feature_extraction_method} \n" \
               f"feature selection: {self.feature_selection_percent} \n"







