import numpy as np
from mock import patch

from feature_extractor import TFFeatureExtractor, Descriptor


class TestTFFeatureExtractor:
    @patch("feature_extractor.TensorflowProxy")
    @patch("feature_extractor.ImageEncoder")
    def test_calculate(self, image_encoder, tensorflow_proxy):
        expected_vector = np.arange(100)
        image_as_array = np.ones([10, 20])
        image_as_bytes = b"123345"
        tensorflow_proxy.return_value.get_desctiptor.return_value = expected_vector
        image_encoder.return_value.numpy_to_binary.return_value = image_as_bytes

        result = TFFeatureExtractor().calculate(image_as_array)

        tensorflow_proxy.assert_called_once_with()
        tensorflow_proxy.return_value.get_desctiptor.assert_called_once_with(image_as_bytes)
        image_encoder.assert_called_once_with(image_format="jpeg")
        image_encoder.return_value.numpy_to_binary.assert_called_once_with(image_as_array)
        assert result == Descriptor(expected_vector)
