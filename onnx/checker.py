# SPDX-License-Identifier: Apache-2.0
"""onnx checker

This implements graphalities that allows us to check whether a serialized
proto is legal.
"""

import functools
import sys
from typing import Any, Callable, Optional, Type, TypeVar, Union, cast

from google.protobuf.message import Message

import onnx.defs
import onnx.onnx_cpp2py_export.checker as c_checker
import onnx.shape_inference
from onnx import (
    IR_VERSION,
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    NodeProto,
    SparseTensorProto,
    TensorProto,
    ValueInfoProto,
    helper,
)

# Limitation of single protobuf file is 2GB
MAXIMUM_PROTOBUF = 2000000000

# TODO: This thing where we reserialize the protobuf back into the
# string, only to deserialize it at the call site, is really goofy.
# Stop doing that.


# NB: Please don't edit this context!
DEFAULT_CONTEXT = c_checker.CheckerContext()
DEFAULT_CONTEXT.ir_version = IR_VERSION
# TODO: Maybe ONNX-ML should also be defaulted?
DEFAULT_CONTEXT.opset_imports = {"": onnx.defs.onnx_opset_version()}


FuncType = TypeVar("FuncType", bound=Callable[..., Any])


# TODO: This really doesn't seem worth the metaprogramming...
def _create_checker(proto_type: Type[Message]) -> Callable[[FuncType], FuncType]:
    def decorator(py_func: FuncType) -> FuncType:
        @functools.wraps(py_func)
        def checker(proto: Message, ctx: c_checker.CheckerContext = DEFAULT_CONTEXT) -> Any:
            if not isinstance(proto, proto_type):
                raise RuntimeError(
                    f"You cannot pass an object that is not of type {proto_type.__name__}"
                )
            return getattr(c_checker, py_func.__name__)(proto.SerializeToString(), ctx)

        return cast(FuncType, checker)

    return decorator


@_create_checker(ValueInfoProto)
def check_value_info(
    value_info: ValueInfoProto, ctx: c_checker.CheckerContext = DEFAULT_CONTEXT
) -> None:
    pass


@_create_checker(TensorProto)
def check_tensor(tensor: TensorProto, ctx: c_checker.CheckerContext = DEFAULT_CONTEXT) -> None:
    pass


@_create_checker(AttributeProto)
def check_attribute(
    attr: AttributeProto, ctx: c_checker.CheckerContext = DEFAULT_CONTEXT
) -> None:
    pass


@_create_checker(NodeProto)
def check_node(node: NodeProto, ctx: c_checker.CheckerContext = DEFAULT_CONTEXT) -> None:
    pass


def check_function(
    function: FunctionProto, ctx: Optional[c_checker.CheckerContext] = None
) -> None:
    if ctx is None:
        ctx = c_checker.CheckerContext()
        ctx.ir_version = helper.find_min_ir_version_for(
            list(function.opset_import), True
        )
        function_opset_dic = {}
        for domain_version in function.opset_import:
            function_opset_dic[domain_version.domain] = domain_version.version
        ctx.opset_imports = function_opset_dic
    c_checker.check_function(function.SerializeToString(), ctx)


@_create_checker(GraphProto)
def check_graph(graph: GraphProto, ctx: c_checker.CheckerContext = DEFAULT_CONTEXT) -> None:
    pass


def check_sparse_tensor(
    sparse: SparseTensorProto, ctx: c_checker.CheckerContext = DEFAULT_CONTEXT
) -> None:
    c_checker.check_sparse_tensor(sparse.SerializeToString(), ctx)


def check_model(model: Union[ModelProto, str, bytes], full_check: bool = False) -> None:
    """Check the consistency of a model. An exception is raised if the test fails.

    Arguments:
        model (ModelProto): model to check
        full_check (bool): if True, the function checks shapes can be inferred
    """
    # If model is a path instead of ModelProto
    if isinstance(model, str):
        c_checker.check_model_path(model, full_check)
    else:
        protobuf_string = (
            model if isinstance(model, bytes) else model.SerializeToString()
        )
        # If the protobuf is larger than 2GB,
        # remind users should use the model path to check
        if sys.getsizeof(protobuf_string) > MAXIMUM_PROTOBUF:
            raise ValueError(
                "This protobuf of onnx model is too large (>2GB). Call check_model with model path instead."
            )
        c_checker.check_model(protobuf_string, full_check)


ValidationError = c_checker.ValidationError
