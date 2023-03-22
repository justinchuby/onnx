# SPDX-License-Identifier: Apache-2.0

from typing import Union

import onnx
import onnx.onnx_cpp2py_export.printer as c_printer


def to_text(proto: Union[onnx.ModelProto, onnx.FunctionProto, onnx.GraphProto]) -> str:
    if isinstance(proto, onnx.ModelProto):
        return c_printer.model_to_text(proto.SerializeToString())
    if isinstance(proto, onnx.FunctionProto):
        return c_printer.function_to_text(proto.SerializeToString())
    if isinstance(proto, onnx.GraphProto):
        return c_printer.graph_to_text(proto.SerializeToString())
    raise TypeError("Unsupported argument type.")
