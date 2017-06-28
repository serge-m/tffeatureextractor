from __future__ import print_function

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2, prediction_service_pb2

import numpy as np

from kolasimagesearch.impl.feature_engine.feature_extractor import FeatureExtractor, Descriptor
from kolasimagestorage.image_encoder import ImageEncoder
from collections import namedtuple

ServerSettings = namedtuple("ServerSettings",
                            ["timeout_in_seconds", "model_name", "signature_name", "inputs_name", "outputs_name"])

DEFAULT_SETTINGS = ServerSettings(timeout_in_seconds=10.0, model_name="inception", signature_name="predict_images2",
                                  inputs_name="images", outputs_name="res2")


class TFFeatureExtractor(FeatureExtractor):
    def __init__(self, server_url_with_port, *args, **kwargs):
        self.tensorflow_proxy = TensorflowProxy(server_url_with_port, server_settings=DEFAULT_SETTINGS)
        self.image_encoder = ImageEncoder(image_format="jpeg")

    def calculate(self, image: np.ndarray) -> Descriptor:
        image_encoded = self.image_encoder.numpy_to_binary(image)
        feature_vector = self.tensorflow_proxy.get_descriptor(image_encoded)
        return Descriptor(feature_vector)


class TensorflowProxy:
    def __init__(self, server_url_with_port: str, server_settings: ServerSettings):
        self._tf_connection = TFConnection(server_url_with_port, server_settings)

    def get_descriptor(self, image_encoded: bytes) -> Descriptor:
        vector = self._tf_connection.predict(image_encoded)
        return Descriptor(vector)


class TFConnection:
    def __init__(self, server_url_with_port: str, server_settings: ServerSettings):
        host, port = server_url_with_port.split(':')
        channel = implementations.insecure_channel(host, int(port))
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
