# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

__all__ = [
    "ProtoSerializer",
    "register_proto_serializer",
]

import abc
import typing
from typing import Any, Collection, Optional, TypeVar

import google.protobuf.message
import google.protobuf.text_format

import onnx

_Proto = TypeVar("_Proto", bound=google.protobuf.message.Message)
# Encoding used for serializing and deserializing text files
_ENCODING = "utf-8"

registered_serializers: dict[str, ProtoSerializer] = {}


class ProtoSerializer(abc.ABC):
    """A serializer-deserializer to and from in-memory Protocol Buffers representations."""

    supported_formats: Collection[str]

    @abc.abstractmethod
    def serialize(self, proto: _Proto) -> Any:
        """Serialize a in-memory proto to a serialized data type."""

    @abc.abstractmethod
    def deserialize(self, serialized: Any, proto: _Proto) -> _Proto:
        """Parse a serialized data type into a in-memory proto."""


class _ProtobufSerializer(ProtoSerializer):
    """Serialize and deserialize protobuf message."""

    supported_formats = ("protobuf",)

    def serialize(self, proto: _Proto) -> bytes:
        if hasattr(proto, "SerializeToString") and callable(proto.SerializeToString):
            try:
                result = proto.SerializeToString()
            except ValueError as e:
                if proto.ByteSize() >= onnx.checker.MAXIMUM_PROTOBUF:
                    raise ValueError(
                        "The proto size is larger than the 2 GB limit. "
                        "Please use save_as_external_data to save tensors separately from the model file."
                    ) from e
                raise
            return result  # type: ignore
        raise TypeError(
            f"No SerializeToString method is detected.\ntype is {type(proto)}"
        )

    def deserialize(self, serialized: bytes, proto: _Proto) -> _Proto:
        if not isinstance(serialized, bytes):
            raise TypeError(
                f"Parameter 'serialized' must be bytes, but got type: {type(serialized)}"
            )
        decoded = typing.cast(Optional[int], proto.ParseFromString(serialized))
        if decoded is not None and decoded != len(serialized):
            raise google.protobuf.message.DecodeError(
                f"Protobuf decoding consumed too few bytes: {decoded} out of {len(serialized)}"
            )
        return proto


class _TextProtoSerializer(ProtoSerializer):
    """Serialize and deserialize text proto."""

    supported_formats = ("textproto",)

    def serialize(self, proto: _Proto) -> bytes | str:
        textproto = google.protobuf.text_format.MessageToString(proto)
        return textproto.encode(_ENCODING)

    def deserialize(self, serialized: bytes | str, proto: _Proto) -> _Proto:
        if not isinstance(serialized, (bytes, str)):
            raise TypeError(
                f"Parameter 'serialized' must be bytes or str, but got type: {type(serialized)}"
            )
        if isinstance(serialized, bytes):
            serialized = serialized.decode(_ENCODING)
        assert isinstance(serialized, str)
        return google.protobuf.text_format.Parse(serialized, proto)


def register_proto_serializer(serializer: ProtoSerializer) -> None:
    """Register a serializer to the ONNX serialization framework."""
    for fmt in serializer.supported_formats:
        registered_serializers[fmt] = serializer


def check_format(fmt: str) -> None:
    """Check if the serialization format is supported."""
    if fmt not in registered_serializers:
        raise ValueError(
            f"Unsupported format: '{fmt}'. Supported formats are: {registered_serializers.keys()}"
        )


# Register default serializers
register_proto_serializer(_ProtobufSerializer())
register_proto_serializer(_TextProtoSerializer())
