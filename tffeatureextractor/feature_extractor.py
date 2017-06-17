import numpy as np

#from tffeatureextractor_detector import DlibFaceDetector
from kolasimagesearch.impl.feature_engine.feature_extractor import FeatureExtractor, Descriptor


class TFFeatureExtractor(FeatureExtractor):
    def __init__(self, *args, **kwargs):
        super(self, TFFeatureExtractor).__init__(*args, **kwargs)
        self.tensorflow_proxy = TensorflowProxy()

    def calculate(self, image: np.ndarray) -> Descriptor:
        image_encoded = encode_image(image)
        feature_vector = self.tensorflow_proxy.get_desctiptor(image_encoded)
        return Descriptor(feature_vector)

