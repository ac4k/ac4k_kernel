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

template <int First, int... Rest>
__forceinline__ bool _TRY_DISPATCH_HEAD_DIM_SIZES(int hdim, auto &&func) {
  if (hdim <= First) {
    func.template operator()<First>();
    return true;
  }

  if constexpr (sizeof...(Rest) > 0) {
    return _TRY_DISPATCH_HEAD_DIM_SIZES<Rest...>(
        hdim, std::forward<decltype(func)>(func));
  }

  return false;
}

template <int... SupportedHeadDimSize>
__forceinline__ void DISPATCH_HEAD_DIM_SIZES(int hdim, auto &&func) {
  if (!_TRY_DISPATCH_HEAD_DIM_SIZES<SupportedHeadDimSize...>(hdim, func)) {
    throw std::invalid_argument("Unsupported hdim: " + std::to_string(hdim));
  }
}

template <bool... SupportedBools>
__forceinline__ void DISPATCH_BOOLEAN_VALUES(bool value, auto &&func) {
  constexpr bool support_true = ((SupportedBools == true) || ...);
  constexpr bool support_false = ((SupportedBools == false) || ...);

  if ((value == true && !support_true) || (value == false && !support_false)) {
    std::string value_str = value ? "true" : "false";
    throw std::invalid_argument("Unsupported bool value: " + value_str);
  }

  if (value == true) {
    func.template operator()<true>();
  } else {
    func.template operator()<false>();
  }
}
