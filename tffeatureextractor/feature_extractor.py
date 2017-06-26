from __future__ import print_function

from grpc.beta import implementations
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2, prediction_service_pb2

import numpy as np

from kolasimagesearch.impl.feature_engine.feature_extractor import FeatureExtractor, Descriptor
from kolasimagestorage.image_encoder import ImageEncoder


class TensorflowProxy(object):
    def get_desctiptor(self, image_encoded):
        return Descriptor([0])


class TFFeatureExtractor(FeatureExtractor):
    def __init__(self, *args, **kwargs):
        self.tensorflow_proxy = TensorflowProxy()
        self.image_encoder = ImageEncoder(image_format="jpeg")

    def calculate(self, image: np.ndarray) -> Descriptor:
        image_encoded = self.image_encoder.numpy_to_binary(image)
        feature_vector = self.tensorflow_proxy.get_desctiptor(image_encoded)
        return Descriptor(feature_vector)


class TFConnection:
    def __init__(self, server_url_with_port="localhost:9000"):
        host, port = server_url_with_port.split(':')
        channel = implementations.insecure_channel(host, int(port))
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        self.timeout = 10.0

    def predict(self, bin_image):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'inception'
        request.model_spec.signature_name = 'predict_images2'
        request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(bin_image, shape=[1]))
        response = self.stub.Predict(request, self.timeout)

        res1 = tf.contrib.util.make_ndarray(response.outputs["res1"])
        res2 = tf.contrib.util.make_ndarray(response.outputs["res2"])

        print(res1.shape, res2.shape)

        return response


def main():
    path_image = "./test.jpg"
    with open(path_image, 'rb') as f:
        data = f.read()

    tf_connection = TFConnection()
    print(tf_connection.predict(data))


if __name__ == '__main__':
    main()
