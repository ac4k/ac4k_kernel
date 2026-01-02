#pragma once

#include <torch/all.h>

template <at::ScalarType ATDtype> struct AtDataTraits;

template <> struct AtDataTraits<at::ScalarType::BFloat16> {
  using Type = __nv_bfloat16;
  using Typex2 = __nv_bfloat162;
  static const int32_t BPE = sizeof(Type);
};

template <> struct AtDataTraits<at::ScalarType::Half> {
  using Type = __half;
  using Typex2 = __half2;
  static const int32_t BPE = sizeof(Type);
};

template <> struct AtDataTraits<at::ScalarType::Float> {
  using Type = float;
  using Typex2 = float2;
  static const int32_t BPE = sizeof(Type);
};
