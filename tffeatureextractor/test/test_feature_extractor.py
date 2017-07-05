import numpy as np
from unittest import mock

from tffeatureextractor.feature_extractor import TFFeatureExtractor
from tffeatureextractor.feature_extractor import Descriptor, DEFAULT_SETTINGS, TensorflowProxy, ServerSettings

port = 232323
url_with_port = "http://url:{}".format(port)
image_as_array = np.ones([10, 20])
image_as_bytes = b"123345"
feature_vector = np.arange(200).reshape([2, 100])


class TestTFFeatureExtractor:
    @mock.patch("tffeatureextractor.feature_extractor.TensorflowProxy")
    @mock.patch("tffeatureextractor.feature_extractor.ImageEncoder")
    def test_calculate(self, image_encoder, tensorflow_proxy):
        expected_vector = np.arange(100)
        tensorflow_proxy.return_value.get_descriptor.return_value = expected_vector
        image_encoder.return_value.numpy_to_binary.return_value = image_as_bytes

        result = TFFeatureExtractor(url_with_port).calculate(image_as_array)

        tensorflow_proxy.assert_called_once_with(url_with_port, server_settings=DEFAULT_SETTINGS)
        tensorflow_proxy.return_value.get_descriptor.assert_called_once_with(image_as_bytes)
        image_encoder.assert_called_once_with(image_format="jpeg")
        image_encoder.return_value.numpy_to_binary.assert_called_once_with(image_as_array)
        assert result == Descriptor(expected_vector)


class TestTensorflowProxy:
    settings = ServerSettings(None, None, None, None, "123123")

    @mock.patch("tffeatureextractor.feature_extractor.TFConnection")
    def test_get_descriptor(self, tf_connection):
        tf_connection.return_value.predict.return_value = feature_vector

        proxy = TensorflowProxy(url_with_port, self.settings)
        descriptor = proxy.get_descriptor(image_as_bytes)

        tf_connection.assert_called_once_with(url_with_port, self.settings)
        tf_connection.return_value.predict.assert_called_once_with(image_as_bytes)
        assert descriptor == Descriptor(vector=feature_vector)
