# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
from dataclasses import dataclass
import copy
from pathlib import Path
import pandas as pd
import argparse
import shutil
from grouped_gemm_bf16_common import kernelInstance, kernels_list, default_kernels_dict



class grouped_gemm_a8w8_fwd_codegen:
    def __init__(self, working_path, istune=False):
        self.working_path = working_path
        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.istune = istune

    def gen_instance(self, k: kernelInstance):
        INSTANCE_IMPL = f"""/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "bf16_grouped_common.h"

std::vector<at::Tensor>
{k.name}(
    at::TensorList A,
    at::TensorList B,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y)
{{{{
    // Check if this input needs to be padded.
#if 0
    bool pad = true;
	for (int i = 0; i < A.size(); i++) {{{{
		int M = A[i].size(0);
		int K = A[i].size(1);
		int N = B[i].size(0);
        pad = pad & (M % {k.MPerBLOCK} != 0) || (N % {k.NPerBLOCK} != 0) || (K % ({k.KPerBLOCK} * KBatch) != 0);
	}}}}
#else
    // disable padding for packed tensor
    bool pad = false;
#endif
    if (pad)
    {{{{
        // pad
        {{INSTANCE_CONTENT_pad}}
        // pad
    }}}}
    else
    {{{{
        // no pad
        {{INSTANCE_CONTENT_nopad}}
        // no pad
    }}}}
}}}}


"""
        INSTANCE_CONTENT_bias = f"""
        {{{{
           using DeviceGemmInstance = DeviceGemmHelper<
                DDataType, EDataType,
                {k.BLOCK_SIZE},
                {k.MPerBLOCK},
                {k.NPerBLOCK},
                {k.KPerBLOCK},
                {k.WAVE_TILE_M},
                {k.WAVE_TILE_N},
                {k.WAVE_MAP_M},
                {k.WAVE_MAP_N},
                S<{(", ").join(map(lambda x:str(x),k.ABLOCK_TRANSFER))}>,
                S<{(", ").join(map(lambda x:str(x),k.BBLOCK_TRANSFER))}>,
                S<{(", ").join(map(lambda x:str(x),k.CBLOCK_TRANSFER))}>,
                S<{(", ").join(map(lambda x:str(x),k.CBLOCK_SPV))}>,
                {k.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE},
                {k.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE},
                ck::BlockGemmPipelineScheduler::{k.LOOP_SCHED},
                ck::BlockGemmPipelineVersion::v{k.PIPELINE_VERSION},
                ck::tensor_operation::device::GemmSpecialization::{{GemmSpec}}>;
            // Run kernel instance.
            return bf16_grouped_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, Y, bias, KBatch);
        }}}}
"""
        INSTANCE_CONTENT_nobias = f"""using DeviceGemmInstance = DeviceGemmHelper<
            {k.BLOCK_SIZE},
            {k.MPerBLOCK},
            {k.NPerBLOCK},
            {k.KPerBLOCK},
            {k.WAVE_TILE_M},
            {k.WAVE_TILE_N},
            {k.WAVE_MAP_M},
            {k.WAVE_MAP_N},
            S<{(", ").join(map(lambda x:str(x),k.ABLOCK_TRANSFER))}>,
            S<{(", ").join(map(lambda x:str(x),k.BBLOCK_TRANSFER))}>,
            S<{(", ").join(map(lambda x:str(x),k.CBLOCK_TRANSFER))}>,
            S<{(", ").join(map(lambda x:str(x),k.CBLOCK_SPV))}>,
            {k.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE},
            {k.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE},
            ck::BlockGemmPipelineScheduler::{k.LOOP_SCHED},
            ck::BlockGemmPipelineVersion::v{k.PIPELINE_VERSION},
            ck::tensor_operation::device::GemmSpecialization::{{GemmSpec}}>;
        // Run kernel instance.
        return bf16_grouped_impl<DeviceGemmInstance>(A, B, kernel_args, Y);
"""
        if self.istune:
            INSTANCE_IMPL_str = INSTANCE_IMPL.format(INSTANCE_CONTENT_pad=(INSTANCE_CONTENT_nobias.format(GemmSpec="MNKPadding")),
                                                     INSTANCE_CONTENT_nopad=(INSTANCE_CONTENT_nobias.format(GemmSpec="Default")))
        else:
            INSTANCE_IMPL_str = INSTANCE_IMPL.format(INSTANCE_CONTENT_pad=INSTANCE_CONTENT_nobias.format(GemmSpec="MNKPadding"),
                                                     INSTANCE_CONTENT_nopad=INSTANCE_CONTENT_nobias.format(GemmSpec="Default"))

        Path(os.path.join(self.impl_path, f"{k.name}.hip")).write_text(
                INSTANCE_IMPL_str)

        INSTANCE_template = """// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "{name}.cuh"

template torch::Tensor
{name}<{dtypes}>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y,
    std::optional<torch::Tensor> bias,
    int KBatch);

"""
        INSTANCE_dBF16_eBF16 = INSTANCE_template.format(
            name=k.name, dtypes="B16")
        INSTANCE_dFP32_eBF16 = INSTANCE_template.format(
            name=k.name, dtypes="F32, B16")
        INSTANCE_dFP16_eFP16 = INSTANCE_template.format(
            name=k.name, dtypes="F16")
        INSTANCE_dFP32_eFP16 = INSTANCE_template.format(
            name=k.name, dtypes="F32, F16")

        if self.istune:
            Path(os.path.join(self.instances_path, f"{k.name}_dBF16_eBF16.cpp")).write_text(
                INSTANCE_dBF16_eBF16)
        else:
            Path(os.path.join(self.instances_path, f"{k.name}_dBF16_eBF16.cpp")).write_text(
                INSTANCE_dBF16_eBF16)
            Path(os.path.join(self.instances_path, f"{k.name}_dFP32_eBF16.cpp")).write_text(
                INSTANCE_dFP32_eBF16)
            Path(os.path.join(self.instances_path, f"{k.name}_dFP16_eFP16.cpp")).write_text(
                INSTANCE_dFP16_eFP16)
            Path(os.path.join(self.instances_path, f"{k.name}_dFP32_eFP16.cpp")).write_text(
                INSTANCE_dFP32_eFP16)

    def gen_lookup_dict(self, kernels_dict):
        LOOKUP_head = """
static const std::unordered_map<std::tuple<int, int, int, int>, GroupedKernel, IntTupleHash> bf16_grouped_lookup_dispatch = {"""
        LOOKUP_template = """
  {{{mnk},{kernel_name}}},"""

        LOOKUP_end = """
}
"""
        with open(os.path.join(self.working_path, "grouped_gemm_a8w8_lookup.h"), "w") as f:
            f.write(LOOKUP_head)
            for mnk, k in kernels_dict.items():
                #print((", ").join(map(lambda x: str(x), list(mnk))), ":", k.name)
                if not self.istune and (isinstance(mnk, tuple) and mnk[0] > 0):
                    f.write(LOOKUP_template.format(mnk="{"+(", ").join(
                        map(lambda x: str(x), list(mnk))) + "}", kernel_name=k.name))
                elif self.istune and isinstance(mnk, int):
                    f.write(LOOKUP_template.format(mnk=mnk, kernel_name=k.name))
            f.write(LOOKUP_end)

    def gen_manifest_head(self, kernels_dict):
        MAINFEST_head = """#pragma once
/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <string>
#include <unordered_map>

#include <ATen/ATen.h>

#define KERNEL_NAME_MAP_ENTRY(name) \
  { #name, name }

using GroupedKernel = std::function<std::vector<at::Tensor>(
    at::TensorList,
    at::TensorList,
    at::Tensor,
    std::vector<at::Tensor>)>;

"""
        MAINFEST_template = """
std::vector<at::Tensor>
{kernel_name}(
	at::TensorList XQ,
    at::TensorList WQ,
    at::Tensor kernel_args,
    std::vector<at::Tensor> Y);
"""
        MAINFEST_end = """
"""

        with open(os.path.join(self.working_path, "bf16_grouped_kernel_manifest.h"), "w") as f:
            f.write(MAINFEST_head)
            for mnk, k in kernels_dict.items():
                f.write(MAINFEST_template.format(kernel_name=k.name))
            f.write(MAINFEST_end)

    def gen_instances(self, kernels_dict):
        if os.path.exists(self.impl_path):
            shutil.rmtree(self.impl_path)
        os.mkdir(self.impl_path)
        if os.path.exists(self.instances_path):
            shutil.rmtree(self.instances_path)
        os.mkdir(self.instances_path)

        for mnk, k in kernels_dict.items():
            self.gen_instance(k)

        self.gen_lookup_dict(kernels_dict)
        self.gen_manifest_head(kernels_dict)


def get_tune_dict(tune_dict_csv):
    tune_dict = default_kernels_dict
    if os.path.exists(tune_dict_csv):
        tune_df = pd.read_csv(tune_dict_csv)
        for i in range(len(tune_df)):
            B = tune_df.loc[i, "B"]
            M = tune_df.loc[i, "M"]
            N = tune_df.loc[i, "N"]
            K = tune_df.loc[i, "K"]
            kid = tune_df.loc[i, "kernelId"]
            tune_dict[(B, M, N, K)] = kernels_list[kid]
    return tune_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK grouped gemm a8w8 kernel",
    )

    # the directory for list_blobs/gen_blobs to write files into
    parser.add_argument(
        "-w",
        "--working_path",
        default="./",
        required=False,
        help="the path where all the blobs are going to be generated"
    )

    parser.add_argument(
        "-f",
        "--tune_file",
        default="aiter/configs/a8w8_tuned_grouped_gemm.csv",
        required=False,
        help="tune_file include the result after run grouped_gemm_a8w8_tune.py"
    )

    parser.add_argument(
        "--tune",
        action='store_true',
        required=False,
        help="generated tune instanses"
    )

    # parser.add_argument(
    #     "--out_type",
    #     default="all",
    #     required=False,
    #     help="Specifie the type of scale\n \
    #         all: [bf16, fp16] \n  \
    #         bf16, fp16"
    # )

    # parser.add_argument(
    #     "--scale_type",
    #     default="all",
    #     required=False,
    #     help="Specifie the type of scale\n \
    #         all: [fp32, same as out] \n  \
    #         same: [same as out]"
    # )


    args = parser.parse_args()
    codegen = grouped_gemm_a8w8_fwd_codegen(args.working_path, args.tune)

    if args.tune:
        codegen.gen_instances(kernels_list)
    else:
        codegen.gen_instances(get_tune_dict(args.tune_file))
