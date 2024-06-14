/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ATB_SPEED_MODELS_MLP_W8A16_OPERATION_H
#define ATB_SPEED_MODELS_MLP_W8A16_OPERATION_H
#include <atb/atb_infer.h>
#include "atb_speed/log.h"

namespace atb_speed {
namespace contrib {
struct MlpW8A16Param {
    int rank = 0;
    int rankSize = 1;
    int rankRoot = 0;
    void *hcclComm = nullptr;
    atb::infer::ActivationType activationType;
    bool transposeB = false;
    bool isBias = true;
    bool isPack = false;
    std::string backend = "hccl";
    bool isQuant = true;
    bool isPrefill = false;
    std::string hcomm_info = "";   
};

atb::Status CreateMlpW8A16Operation(const MlpW8A16Param &param, atb::Operation **operation);
} // namespace contrib
} // namespace atb_speed
#endif

