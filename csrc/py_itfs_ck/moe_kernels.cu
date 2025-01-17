// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include "py_itfs_common.h"

#include "fused_moe.hpp"

torch::Tensor ck_moe(torch::Tensor &hidden_states,          // [m, k], input token
                     torch::Tensor &w1,                     // [e, n, k]/[e, 2*n, k], pre-shuffle([e, nr, kr, w])
                     torch::Tensor &w2,                     // [e, n, k], pre-shuffle([e, nr, kr, w])
                     torch::Tensor &topk_weights,           // [tokens, topk]
                     torch::Tensor &topk_ids,               // [tokens, topk]
                     std::optional<torch::Tensor> w1_scale, // [e, 1, n], gate(up) scale
                     std::optional<torch::Tensor> w2_scale, // [e, 1, k], down scale
                     std::optional<torch::Tensor> a1_scale, // [m, 1], token scale
                     std::optional<torch::Tensor> a2_scale, // [e, 1, n], smooth-quant-scale for 2nd gemm input
                     std::optional<int> block_m = 32)
{
    auto device = hidden_states.device();
    int topk_ids_numel = topk_ids.numel();
    int experts = w1.size(0);
    int topk = topk_ids.size(1);
    int tokens = topk_ids.size(0);
    int hidden_size = w1.size(2);
    int shared_intermediate_size_0 = w1.size(1);
    int block_size = block_m.value();

    int max_num_tokens_padded = topk_ids_numel + experts * block_size - topk;
    int max_num_m_blocks = (max_num_tokens_padded + block_size - 1) / block_size;

    auto sorted_ids = torch::empty({max_num_tokens_padded}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto sorted_weights = torch::empty({max_num_tokens_padded}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto sorted_expert_ids = torch::empty({max_num_m_blocks}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto num_tokens_post_pad = torch::empty({1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
    auto out = torch::empty_like(hidden_states);

    auto prec_i = torchDTypeToStr(hidden_states.dtype());
    auto prec_w = torchDTypeToStr(w1.dtype());
    auto prec_o = torchDTypeToStr(out.dtype());
    auto prec_kw = torchDTypeToStr(topk_weights.dtype());

    int gate_only = 0;
    int activation = 0;
    int fused_quant = 0;
    if (shared_intermediate_size_0 == 2 * w2.size(-1))
    {
        gate_only = 1;
        activation = 1;
    }

    if (!w1_scale.has_value())
    {
        fused_quant = 0;
    }
    else if (a2_scale.has_value())
    {
        fused_quant = 1;
    }
    else
    {
        fused_quant = 2;
    }

    int stride = hidden_size;
    std::string prec_st = !a1_scale ? "fp32" : torchDTypeToStr(a1_scale->dtype());
    std::string prec_sw = !w1_scale ? "fp32" : torchDTypeToStr(w1_scale->dtype());
    std::string prec_sq = !a2_scale ? "fp32" : torchDTypeToStr(a2_scale->dtype());

    // std::cout << "gate_only" << gate_only << std::endl;
    // std::cout << "activation" << activation << std::endl;
    // std::cout << "fused_quant" << fused_quant << std::endl;
    // std::cout << "prec_i" << prec_i << std::endl;
    // std::cout << "prec_w" << prec_w << std::endl;
    // std::cout << "prec_o" << prec_o << std::endl;
    // std::cout << "block_size" << block_size << std::endl;
    // std::cout << "hidden_size" << hidden_size << std::endl;
    // std::cout << "shared_intermediate_size_0" << shared_intermediate_size_0 << std::endl;
    // std::cout << "tokens" << tokens << std::endl;
    // std::cout << "experts" << experts << std::endl;
    // std::cout << "topk" << topk << std::endl;
    // std::cout << "stride" << stride << std::endl;
    // std::cout << "max_num_tokens_padded" << max_num_tokens_padded << std::endl;
    // std::cout << "max_num_m_blocks" << max_num_m_blocks << std::endl;

    fused_moe_traits traits{prec_i,
                            prec_w,
                            prec_o,
                            prec_st,
                            prec_sw,
                            prec_sq,
                            prec_kw,
                            block_size,
                            activation,
                            gate_only,
                            fused_quant};

    fused_moe_args args{hidden_states.data_ptr(),
                        a1_scale.has_value() ? a1_scale->data_ptr() : nullptr,
                        w1.data_ptr(),
                        w2.data_ptr(),
                        w1_scale.has_value() ? w1_scale->data_ptr() : nullptr,
                        w2_scale.has_value() ? w2_scale->data_ptr() : nullptr,
                        a2_scale.has_value() ? a2_scale->data_ptr() : nullptr,
                        out.data_ptr(),

                        topk_ids.data_ptr(),
                        topk_weights.data_ptr(),
                        sorted_ids.data_ptr(),
                        sorted_weights.data_ptr(),
                        sorted_expert_ids.data_ptr(),
                        num_tokens_post_pad.data_ptr(),
                        
                        block_size,
                        hidden_size,
                        shared_intermediate_size_0,
                        tokens,
                        experts,
                        topk,
                        stride};

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    fused_moe(traits, args, {stream});
}