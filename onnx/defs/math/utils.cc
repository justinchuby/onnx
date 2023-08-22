/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "onnx/defs/math/utils.h"

#include <string>

#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {
namespace defs::math::utils {
template <typename T>
T GetScalarValueFromTensor(const ONNX_NAMESPACE::TensorProto* t) {
  if (t == nullptr) {
    return T{};
  }

  auto data_type = t->data_type();
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto::FLOAT:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<float>(t).at(0));
    case ONNX_NAMESPACE::TensorProto::DOUBLE:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<double>(t).at(0));
    case ONNX_NAMESPACE::TensorProto::INT32:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<int32_t>(t).at(0));
    case ONNX_NAMESPACE::TensorProto::INT64:
      return static_cast<T>(ONNX_NAMESPACE::ParseData<int64_t>(t).at(0));
    default:
      fail_shape_inference("Unsupported input data type of ", data_type);
  }
}
std::function<void(OpSchema&)>
SoftmaxFamilyDocGenerator(const char* name, const char* description, const char* equation) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
The operator computes the {description} values for the given input:

 {equation}

The "axis" attribute indicates the dimension along which {name}
will be performed. The output tensor has the same shape
and contains the {name} values of the corresponding input.
)DOC";
                        ReplaceAll(doc, "{name}", name);
                        ReplaceAll(doc, "{description}", description);
                        ReplaceAll(doc, "{equation}", equation););
    std::string axis_attr;
    POPULATE_OP_DOC_STR(axis_attr = R"DOC(
Describes the dimension {name} will be performed on.
Negative value means counting dimensions
from the back. Accepted range is [-r, r-1] where r = rank(input).
)DOC";
                        ReplaceAll(axis_attr, "{name}", name););
    schema.SetDoc(doc);
    schema.Attr("axis", axis_attr, AttributeProto::INT, static_cast<int64_t>(-1));
    schema.Input(
        0, "input", "The input tensor of rank >= axis.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable);
    schema.Output(
        0,
        "output",
        "The output values with the same shape as the input tensor.",
        "T",
        OpSchema::Single,
        true,
        1,
        OpSchema::Differentiable);
    schema.TypeConstraint(
        "T",
        {"tensor(float16)", "tensor(float)", "tensor(double)", "tensor(bfloat16)"},
        "Constrain input and output types to float tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // Type inference
      propagateElemTypeFromInputToOutput(ctx, 0, 0);

      // Shape inference starts
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }

      // Validate the value of 'axis'
      const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
      int r = input_shape.dim_size();
      int axis = static_cast<int>(getAttribute(ctx, "axis", -1));
      if (axis < -r || axis >= r) {
        fail_shape_inference("'axis' must be in [", -r, " , ", (r - 1), "]. Its actual value is: ", axis);
      }

      // Shape inference
      propagateShapeFromInputToOutput(ctx, 0, 0);
    });
  };
}
} // namespace defs::math::utils

} // namespace ONNX_NAMESPACE
