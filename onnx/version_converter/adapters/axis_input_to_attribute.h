// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE {
namespace version_conversion {

class AxisInputToAttribute : public Adapter {
 public:
  // Convert axis from input to attribute.
  // axis_index: index of the axis input
  // default_axis: default value of axis
  explicit AxisInputToAttribute(
      const std::string& op_name,
      const OpSetID& initial,
      const OpSetID& target,
      int64_t axis_index,
      int64_t default_axis)
      : Adapter(op_name, initial, target), axis_index(axis_index), default_axis(default_axis) {}

  Node* adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    // Identify if axis is statically determined; if so, feed as attribute
    const ArrayRef<Value*>& inputs = node->inputs();

    // 1. Handle when axis is not given
    if !(inputs.size() > this->axis_index && inputs[this->axis_index]->node()->kind() != kUndefined) {
      node->i_(kaxis, this->default_axis);
      return EnsureAndReturnNode(node);
    }

    // 2. Get axis from constant operator
    Value* index_val = inputs[this->axis_index];
    Node* node = index_val->node();
    // Identify whether we have a Constant Op or an Initializer
    if (node->kind() == kConstant) {
      // Get value attribute of kConstant
      const std::vector<int64_t>& int64s = node->t(kvalue).int64s();
      if (int64s.empty()) {
        // Also handle raw data
        std::string raw_data = node->t(kvalue).raw();
        ONNX_ASSERTM(
            raw_data.size() != 0 && raw_data.size() % 8 == 0,
            "Raw Data must be non-empty and size must be a multiple of 8");
        // TODO(justinchuby): Why cast to char* first?
        int64_t* raw = const_cast<int64_t*>(const_cast<char*>(raw_data.c_str()));
        node->i_(kaxis, static_cast<int64_t>(raw[0]));
      } else {
        node->i_(kaxis, int64s.at(0));
      }
      // If Constant node isn't used anywhere else, remove it
      node->removeInput(this->axis_index);
      if (index_val->uses().size() < 1) {
        node->destroy();
      }
      return EnsureAndReturnNode(node);
    }

    // 3. Get axis from initializer
    // Get Value name, find Initializer with same name
    for (const auto& initializer : graph->initializers()) {
      if (initializer.name() == inputs[this->axis_index]->uniqueName()) {
        node->i_(kaxis, initializer.int64s().at(0));
        node->removeInput(this->axis_index);
        // Remove initializer
        if (index_val->uses().size() < 1)
          graph->eraseInitializerAndInput(index_val);
        break;
      }
    }
    return EnsureAndReturnNode(node);
  }

 private:
  int64_t axis_index;
  int64_t default_axis;

  inline Node* EnsureAndReturnNode(Node* node) const {
    ONNX_ASSERTM(node->hasAttribute(kaxis), "Axis attribute not created. This may be a bug.");
    return node;
  }
};

} // namespace version_conversion
} // namespace ONNX_NAMESPACE
