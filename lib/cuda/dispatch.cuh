#pragma once

#include <stdexcept>
#include <torch/all.h>

template <at::ScalarType... SupportedTypes>
__forceinline__ void DISPATCH_AT_TENSOR_TYPES(at::ScalarType type,
                                              auto &&func) {
  bool found = ((type == SupportedTypes) || ...);
  if (!found) {
    throw std::invalid_argument("Unsupported type: " + static_cast<int>(type));
  }

  ((type == SupportedTypes ? (func.template operator()<SupportedTypes>(), false)
                           : false) ||
   ...);
}
