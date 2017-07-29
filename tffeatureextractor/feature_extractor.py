from __future__ import print_function

from grpc.beta import implementations
import tensorflow as tf

from tffeatureextractor.tensorflow_serving.apis import predict_pb2, prediction_service_pb2

import numpy as np

from kolasimagecommon import FeatureExtractor, Descriptor
from kolasimagestorage.image_encoder import ImageEncoder
from collections import namedtuple
import logging
from typing import Iterable

ServerSettings = namedtuple("ServerSettings",
                            ["timeout_in_seconds", "model_name", "signature_name", "inputs_name", "outputs_name"])

DEFAULT_SETTINGS = ServerSettings(timeout_in_seconds=10.0, model_name="inception", signature_name="predict_images2",
                                  inputs_name="images", outputs_name="res2")

sieve_step = 16


class TFFeatureExtractor(FeatureExtractor):
    def __init__(self, server_url_with_port, *args, **kwargs):
        logger = logging.getLogger(__name__)
        logger.info("Init TFFeatureExtractor with server_url_with_port {}".format(server_url_with_port))
        self.tensorflow_proxy = TensorflowProxy(server_url_with_port, server_settings=DEFAULT_SETTINGS)
        self.image_encoder = ImageEncoder(image_format="jpeg")

    def calculate(self, image: np.ndarray) -> Descriptor:
        image_encoded = self.image_encoder.numpy_to_binary(image)
        return self.tensorflow_proxy.get_descriptor(image_encoded)

    def descriptor_shape(self) -> Iterable[int]:
        return 1, 8, 8, 2048 // sieve_step


class TensorflowProxy:
    def __init__(self, server_url_with_port: str, server_settings: ServerSettings):
        logger = logging.getLogger(__name__)
        logger.info("Init TensorflowProxy "
                    "with server_url_with_port {}, server_settings {}".format(server_url_with_port, server_settings))
        self._tf_connection = TFConnection(server_url_with_port, server_settings)

    def get_descriptor(self, image_encoded: bytes) -> Descriptor:
        vector = self._tf_connection.predict(image_encoded)
        return Descriptor(vector[:, :, :, ::sieve_step])


class TFConnection:
    def __init__(self, server_url_with_port: str, server_settings: ServerSettings):
        channel = implementations.insecure_channel(server_url_with_port, None)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        self.server_settings = server_settings

    def _prepare_request(self, bin_image: bytes):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.server_settings.model_name
        request.model_spec.signature_name = self.server_settings.signature_name
        proto = tf.contrib.util.make_tensor_proto(bin_image, shape=[1])
        request.inputs[self.server_settings.inputs_name].CopyFrom(proto)
        return request

    def predict(self, bin_image: bytes) -> np.ndarray:
        request = self._prepare_request(bin_image)
        response = self.stub.Predict(request, self.server_settings.timeout_in_seconds)

        return tf.contrib.util.make_ndarray(response.outputs[self.server_settings.outputs_name])


def main():
    path_image = "./test.jpg"
    with open(path_image, 'rb') as f:
        data = f.read()

    tf_connection = TFConnection("localhost:9000", DEFAULT_SETTINGS)
    print(tf_connection.predict(data))


if __name__ == '__main__':
    main()
