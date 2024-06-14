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
#include <atb/atb_infer.h>
#include <memory>
#include "atb_speed/log.h"
#include "linear_parallel_w8a16.h"
#include "layers/plugin_op/w8a16_operation.h"
#include "layers/plugin_op/w16a16_operation.h"
#include "mlp_w8a16.h"

namespace atb_speed {
namespace contrib {

enum MlpW8A16TensorId {
    IN_INPUT = 0,
    IN_MLPGATEUPWEIGHT,
    IN_MLPGATEUPSCALE,
    IN_MLPGATEUPOFFSET,
    IN_MLPDOWNWEIGHT,
    IN_MLPDOWNSCALE,
    IN_MLPDOWNOFFSET,
    IN_MLPRESIDUALBIAS,
    OUT_MLPRESULT,
    INTERMIDATE_GATEUP_OUT,
    INTERMIDATE_SWIGLU_OUT
};

static const uint64_t IN_TENSOR_COUNT = 7;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 2;
static const uint64_t NODE_COUNT = 3;

atb::Status CreateMlpW8A16Operation(const MlpW8A16Param &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    size_t mlpOutId = OUT_MLPRESULT;
    size_t gateupOutId = INTERMIDATE_GATEUP_OUT;
    size_t swigluOutId = INTERMIDATE_SWIGLU_OUT;
    if (param.isBias) {
        opGraph.inTensorNum = IN_TENSOR_COUNT + 1;
    } else {
        opGraph.inTensorNum = IN_TENSOR_COUNT;
        mlpOutId--;
        gateupOutId--;
        swigluOutId--;
    }
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
    opGraph.name = "MlpW8A16";

    size_t nodeId = 0;
    atb::Node &linearGateUpNode = opGraph.nodes.at(nodeId++);
    atb::Node &swishNode = opGraph.nodes.at(nodeId++);
    atb::Node &linearDownNode = opGraph.nodes.at(nodeId++);

    if (param.isQuant) {
        // FOR W8A16
        // ATB_LOG(INFO) << "enter MLP isQuant = True";
        linearGateUpNode.operation = new atb_speed::common::W8A16Operation("MlpGateUpNode");
        linearGateUpNode.inTensorIds = {IN_INPUT, IN_MLPGATEUPWEIGHT, IN_MLPGATEUPSCALE, IN_MLPGATEUPOFFSET};
        linearGateUpNode.outTensorIds = {gateupOutId};
    } else {
        // FOR BF16, FP16
        // ATB_LOG(INFO) << "enter MLP isQuant = False";

        // linearGateUpNode.operation = new atb_speed::common::W16A16Operation("MlpGateUpNode", true);
        // linearGateUpNode.inTensorIds = {IN_INPUT, IN_MLPGATEUPWEIGHT};
        // linearGateUpNode.outTensorIds = {gateupOutId};
        // linearGateUpNode.inTensorReshapeFuncs.resize(linearGateUpNode.inTensorIds.size());
        // linearGateUpNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        //     newShape.dimNum = 4;
        //     newShape.dims[0] = oldShape.dims[1]/16;
        //     newShape.dims[1] = oldShape.dims[0]/16;
        //     newShape.dims[2] = 16;
        //     newShape.dims[3] = 16;
        // };        

        atb::infer::LinearParam linearParam = {false, true, false};
        // atb::infer::LinearParam linearParam = {
        CREATE_OPERATION(linearParam, &linearGateUpNode.operation);
        linearGateUpNode.inTensorIds = {IN_INPUT, IN_MLPGATEUPWEIGHT};
        linearGateUpNode.outTensorIds = {gateupOutId};
    }

    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
    activationParam.dim = -1;
    CREATE_OPERATION(activationParam, &swishNode.operation);
    swishNode.inTensorIds = {gateupOutId};
    swishNode.outTensorIds = {swigluOutId};

    atb_speed::contrib::LinearParallelW8A16Param linearParallelParam;
    linearParallelParam.transWeight = param.transposeB;
    linearParallelParam.rank = param.rank;
    linearParallelParam.rankSize = param.rankSize;
    linearParallelParam.rankRoot = param.rankRoot;
    linearParallelParam.isBias = param.isBias;
    linearParallelParam.parallelType = "RowParallel";
    linearParallelParam.backend = param.backend;
    linearParallelParam.isQuant = param.isQuant;
    linearParallelParam.isPrefill = param.isPrefill;
    linearParallelParam.hcomm_info = param.hcomm_info;
    CreateLinearParallelW8A16(linearParallelParam, &linearDownNode.operation, true);
    if (param.isBias) {
        linearDownNode.inTensorIds = {INTERMIDATE_SWIGLU_OUT, IN_MLPDOWNWEIGHT, IN_MLPDOWNSCALE, IN_MLPDOWNOFFSET, IN_MLPRESIDUALBIAS};
    } else {
        linearDownNode.inTensorIds = {INTERMIDATE_SWIGLU_OUT, IN_MLPDOWNWEIGHT, IN_MLPDOWNSCALE, IN_MLPDOWNOFFSET};
    }

    linearDownNode.outTensorIds = {OUT_MLPRESULT};

    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}
} // namespace contrib
} // namespace atb_speed

