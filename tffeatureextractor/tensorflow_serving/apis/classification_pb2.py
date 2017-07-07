# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tensorflow_serving/apis/classification.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from tffeatureextractor.tensorflow_serving.apis import input_pb2 as tensorflow__serving_dot_apis_dot_input__pb2
from tffeatureextractor.tensorflow_serving.apis import model_pb2 as tensorflow__serving_dot_apis_dot_model__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='tensorflow_serving/apis/classification.proto',
  package='tensorflow.serving',
  syntax='proto3',
  serialized_pb=_b('\n,tensorflow_serving/apis/classification.proto\x12\x12tensorflow.serving\x1a#tensorflow_serving/apis/input.proto\x1a#tensorflow_serving/apis/model.proto\"%\n\x05\x43lass\x12\r\n\x05label\x18\x01 \x01(\t\x12\r\n\x05score\x18\x02 \x01(\x02\"=\n\x0f\x43lassifications\x12*\n\x07\x63lasses\x18\x01 \x03(\x0b\x32\x19.tensorflow.serving.Class\"T\n\x14\x43lassificationResult\x12<\n\x0f\x63lassifications\x18\x01 \x03(\x0b\x32#.tensorflow.serving.Classifications\"t\n\x15\x43lassificationRequest\x12\x31\n\nmodel_spec\x18\x01 \x01(\x0b\x32\x1d.tensorflow.serving.ModelSpec\x12(\n\x05input\x18\x02 \x01(\x0b\x32\x19.tensorflow.serving.Input\"R\n\x16\x43lassificationResponse\x12\x38\n\x06result\x18\x01 \x01(\x0b\x32(.tensorflow.serving.ClassificationResultB\x03\xf8\x01\x01\x62\x06proto3')
  ,
  dependencies=[tensorflow__serving_dot_apis_dot_input__pb2.DESCRIPTOR,tensorflow__serving_dot_apis_dot_model__pb2.DESCRIPTOR,])
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_CLASS = _descriptor.Descriptor(
  name='Class',
  full_name='tensorflow.serving.Class',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='label', full_name='tensorflow.serving.Class.label', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='score', full_name='tensorflow.serving.Class.score', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=142,
  serialized_end=179,
)


_CLASSIFICATIONS = _descriptor.Descriptor(
  name='Classifications',
  full_name='tensorflow.serving.Classifications',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='classes', full_name='tensorflow.serving.Classifications.classes', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=181,
  serialized_end=242,
)


_CLASSIFICATIONRESULT = _descriptor.Descriptor(
  name='ClassificationResult',
  full_name='tensorflow.serving.ClassificationResult',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='classifications', full_name='tensorflow.serving.ClassificationResult.classifications', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=244,
  serialized_end=328,
)


_CLASSIFICATIONREQUEST = _descriptor.Descriptor(
  name='ClassificationRequest',
  full_name='tensorflow.serving.ClassificationRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_spec', full_name='tensorflow.serving.ClassificationRequest.model_spec', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='input', full_name='tensorflow.serving.ClassificationRequest.input', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=330,
  serialized_end=446,
)


_CLASSIFICATIONRESPONSE = _descriptor.Descriptor(
  name='ClassificationResponse',
  full_name='tensorflow.serving.ClassificationResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='result', full_name='tensorflow.serving.ClassificationResponse.result', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=448,
  serialized_end=530,
)

_CLASSIFICATIONS.fields_by_name['classes'].message_type = _CLASS
_CLASSIFICATIONRESULT.fields_by_name['classifications'].message_type = _CLASSIFICATIONS
_CLASSIFICATIONREQUEST.fields_by_name['model_spec'].message_type = tensorflow__serving_dot_apis_dot_model__pb2._MODELSPEC
_CLASSIFICATIONREQUEST.fields_by_name['input'].message_type = tensorflow__serving_dot_apis_dot_input__pb2._INPUT
_CLASSIFICATIONRESPONSE.fields_by_name['result'].message_type = _CLASSIFICATIONRESULT
DESCRIPTOR.message_types_by_name['Class'] = _CLASS
DESCRIPTOR.message_types_by_name['Classifications'] = _CLASSIFICATIONS
DESCRIPTOR.message_types_by_name['ClassificationResult'] = _CLASSIFICATIONRESULT
DESCRIPTOR.message_types_by_name['ClassificationRequest'] = _CLASSIFICATIONREQUEST
DESCRIPTOR.message_types_by_name['ClassificationResponse'] = _CLASSIFICATIONRESPONSE

Class = _reflection.GeneratedProtocolMessageType('Class', (_message.Message,), dict(
  DESCRIPTOR = _CLASS,
  __module__ = 'tensorflow_serving.apis.classification_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.serving.Class)
  ))
_sym_db.RegisterMessage(Class)

Classifications = _reflection.GeneratedProtocolMessageType('Classifications', (_message.Message,), dict(
  DESCRIPTOR = _CLASSIFICATIONS,
  __module__ = 'tensorflow_serving.apis.classification_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.serving.Classifications)
  ))
_sym_db.RegisterMessage(Classifications)

ClassificationResult = _reflection.GeneratedProtocolMessageType('ClassificationResult', (_message.Message,), dict(
  DESCRIPTOR = _CLASSIFICATIONRESULT,
  __module__ = 'tensorflow_serving.apis.classification_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.serving.ClassificationResult)
  ))
_sym_db.RegisterMessage(ClassificationResult)

ClassificationRequest = _reflection.GeneratedProtocolMessageType('ClassificationRequest', (_message.Message,), dict(
  DESCRIPTOR = _CLASSIFICATIONREQUEST,
  __module__ = 'tensorflow_serving.apis.classification_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.serving.ClassificationRequest)
  ))
_sym_db.RegisterMessage(ClassificationRequest)

ClassificationResponse = _reflection.GeneratedProtocolMessageType('ClassificationResponse', (_message.Message,), dict(
  DESCRIPTOR = _CLASSIFICATIONRESPONSE,
  __module__ = 'tensorflow_serving.apis.classification_pb2'
  # @@protoc_insertion_point(class_scope:tensorflow.serving.ClassificationResponse)
  ))
_sym_db.RegisterMessage(ClassificationResponse)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\370\001\001'))
# @@protoc_insertion_point(module_scope)
