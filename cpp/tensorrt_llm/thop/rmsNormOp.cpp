/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/rmsnormKernels.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <torch/extension.h>
#include <vector>

namespace torch_ext
{

#if 0
template <typename T, typename QuantT>
void invokeGeneralRmsNorm(T* out, T const* input, T const* gamma, T const* beta, float const eps, int const tokens,
    int const hidden_dim, QuantMode quantMode, cudaStream_t stream, float const* clampPtr, float const* scale,
    float* dynamic_scale, float* sum_per_token, QuantT* normed_output_quant)

else if (inputDesc[0].type == DataType::kFLOAT && mOutputType == DataType::kFP8)
    {
        dispatchDataType<float, __nv_fp8_e4m3>(
            nullptr, input, weight, bias, mEps, m, n, stream, clampValPtr, scale, dynamic_scale, sum_per_token, output);
    }
#endif

std::tuple<torch::Tensor, torch::Tensor> rms_norm_quant_fp8(torch::Tensor const& input, torch::Tensor const& residual,
    torch::Tensor const& norm_weight, double const eps, torch::Tensor const& scale)
{
    TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor [batch_size, hidden_dim].");
    TORCH_CHECK(input.sizes()[0] > 0, "Batch size must be greater than 0.");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "Only bf16 input is supported");
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensors.");

    auto normed_output_quant = torch::empty_like(input, torch::kFloat8_e4m3fn);
    auto out = torch::empty_like(input);

    auto const num_tokens = input.sizes()[0];
    auto const hidden_dim = input.sizes()[1];

    auto quant_mode = tensorrt_llm::common::QuantMode::fp8Qdq();
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

    using T = __nv_bfloat16;
    using QuantT = __nv_fp8_e4m3;

    tensorrt_llm::kernels::invokeGeneralRmsNorm<T, QuantT>(static_cast<T*>(out.mutable_data_ptr()),
        static_cast<T*>(input.data_ptr()), static_cast<T*>(norm_weight.data_ptr()),
        static_cast<T*>(residual.data_ptr()), eps, num_tokens, hidden_dim, quant_mode, stream, nullptr,
        static_cast<float*>(scale.data_ptr()), nullptr, nullptr, static_cast<QuantT*>(normed_output_quant.data_ptr()));

    return {normed_output_quant, out};
}

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "rms_norm_quant_fp8(Tensor input, Tensor residual, Tensor norm_weight, "
        "float eps, Tensor scale) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("rms_norm_quant_fp8", &torch_ext::rms_norm_quant_fp8);
}
