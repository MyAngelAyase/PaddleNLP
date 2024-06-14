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
#include <cmath>
#include <numeric>
#include "atb_speed/log.h"
#include "layers/plugin_op/w8a16_operation.h"
#include "layers/plugin_op/w16a16_operation.h"
#include "layers/plugin_op/matmul_allreduce_operation.h"
#include "linear_parallel_w8a16.h"

namespace atb_speed {
namespace contrib {

enum LinearParallelW8A16TensorId {
    IN_INPUT = 0,
    IN_WEIGHT,
    IN_SCALE,
    IN_OFFSET,
    IN_BIAS,
    OUT_LINEAROUT,
};

static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 0;
static const uint64_t NODE_COUNT = 2;

atb::Status CreateLinearParallelW8A16Inner(const LinearParallelW8A16Param &param, atb::Operation **operation, bool transB)
{
    atb::GraphParam opGraph;
    size_t outTensorId = OUT_LINEAROUT;
    if (param.isBias) {
        opGraph.inTensorNum = IN_TENSOR_COUNT + 1;
        opGraph.nodes.resize(param.rank == 0 ? NODE_COUNT + 1 : NODE_COUNT + 2);
        // opGraph.nodes.resize(NODE_COUNT);
    }else{
        opGraph.inTensorNum = IN_TENSOR_COUNT;
        opGraph.nodes.resize(NODE_COUNT);
        outTensorId--;
    }   
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.name = "LinearParallelW8A16";

    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);

    if (param.isQuant) {
        // FOR W8A16
        // ATB_LOG(INFO) << "enter linearParallel isQuant = True";
        linearNode.operation = new atb_speed::common::W8A16Operation("LinearNode");
        linearNode.inTensorIds = {IN_INPUT, IN_WEIGHT, IN_SCALE, IN_OFFSET};
        linearNode.outTensorIds = {outTensorId};
    }else{
        // FOR BF16, FP16
        // ATB_LOG(INFO) << "enter linearParallel isQuant = False";
        // atb::infer::MatmulParam linearParam = {false, true};

        // linearNode.operation = new atb_speed::common::W16A16Operation("LinearNode", transB);
        // linearNode.inTensorIds = {IN_INPUT, IN_WEIGHT};
        // linearNode.outTensorIds = {outTensorId};
        // linearNode.inTensorReshapeFuncs.resize(linearNode.inTensorIds.size());
        // linearNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        //     newShape.dimNum = 4;
        //     newShape.dims[0] = oldShape.dims[1]/16;
        //     newShape.dims[1] = oldShape.dims[0]/16;
        //     newShape.dims[2] = 16;
        //     newShape.dims[3] = 16;
        // };

        atb::infer::LinearParam linearParam;
        linearParam.transposeA = false;
        linearParam.transposeB = true;
        linearParam.hasBias = false;
        CREATE_OPERATION(linearParam, &linearNode.operation);
        linearNode.inTensorIds = {IN_INPUT, IN_WEIGHT};
        linearNode.outTensorIds = {outTensorId};
    }

    if (param.isBias) {
        if(param.rank != 0){
            atb::Node &mulsNode = opGraph.nodes.at(nodeId++);
            atb::infer::ElewiseParam mulsParam;
            mulsParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MULS;
            mulsParam.mulsParam.varAttr = 0;
            CREATE_OPERATION(mulsParam, &mulsNode.operation);
            mulsNode.inTensorIds = {IN_BIAS};
            mulsNode.outTensorIds = {IN_BIAS};
        }
        atb::Node &residualNode = opGraph.nodes.at(nodeId++);
        atb::infer::ElewiseParam residualParam;
        residualParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
        CREATE_OPERATION(residualParam, &residualNode.operation);
        residualNode.inTensorIds = {outTensorId, IN_BIAS};
        residualNode.outTensorIds = {outTensorId};
    }

    atb::Node &allReduceNode = opGraph.nodes.at(nodeId++);
    atb::infer::AllReduceParam allReduceParam = {param.rank, param.rankSize, param.rankRoot,
                                                 "sum",      param.backend,  param.hcclComm};
    CREATE_OPERATION(allReduceParam, &allReduceNode.operation);
    allReduceNode.inTensorIds = {outTensorId};
    allReduceNode.outTensorIds = {outTensorId};

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

static const uint64_t ROW_PARALLEL_LCOC_NODE_COUNT = 1;

atb::Status CreateLinearParallelLcoc(const LinearParallelW8A16Param &param, atb::Operation **operation)
{
    // atb::GraphParam opGraph;
    // opGraph.inTensorNum = IN_TENSOR_COUNT + 1;
    // opGraph.outTensorNum = OUT_TENSOR_COUNT;

    // opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    // opGraph.nodes.resize(ROW_PARALLEL_LCOC_NODE_COUNT);

    // size_t nodeId = 0;

    // atb::Node &linearParallelNode = opGraph.nodes.at(nodeId++);
    // // atb::infer::LinearParallelParam linearParallelParam;
    // // linearParallelParam.transWeight = true;
    // // linearParallelParam.rank = param.rank;
    // // linearParallelParam.rankSize = param.rankSize;
    // // linearParallelParam.hasResidual = false;
    // // linearParallelParam.backend = "lcoc";
    // // CREATE_OPERATION(linearParallelParam, &linearParallelNode.operation);

    // // linearParallelNode.inTensorIds = { IN_INPUT, IN_WEIGHT};
    // // linearParallelNode.outTensorIds = { OUT_LINEAROUT };
    // linearParallelNode.operation = new atb_speed::common::MatmulAllreduceOperation("MatmulAllreduceOperation", param.hcomm_info);
    // linearParallelNode.inTensorIds = {IN_INPUT, IN_WEIGHT};
    // linearParallelNode.outTensorIds = {OUT_LINEAROUT};

    atb::GraphParam opGraph;
    size_t outTensorId = OUT_LINEAROUT;
    if (param.isBias) {
        opGraph.inTensorNum = IN_TENSOR_COUNT + 1;
        // opGraph.nodes.resize(param.rank == 0 ? NODE_COUNT + 1 : NODE_COUNT + 2);
        opGraph.nodes.resize(NODE_COUNT);
    }else{
        opGraph.inTensorNum = IN_TENSOR_COUNT;
        opGraph.nodes.resize(NODE_COUNT);
        outTensorId--;
    }   

    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.name = "LinearParallelW8A16";

    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);

    if (param.isQuant) {
        // FOR W8A16
        // ATB_LOG(INFO) << "enter linearParallel isQuant = True";
        linearNode.operation = new atb_speed::common::W8A16Operation("LinearNode");
        linearNode.inTensorIds = {IN_INPUT, IN_WEIGHT, IN_SCALE, IN_OFFSET};
        linearNode.outTensorIds = {outTensorId};
    }else{
        // FOR BF16, FP16
        // ATB_LOG(INFO) << "enter linearParallel isQuant = False";

        // linearNode.operation = new atb_speed::common::W16A16Operation("LinearNode", true);
        // linearNode.inTensorIds = {IN_INPUT, IN_WEIGHT};
        // linearNode.outTensorIds = {outTensorId};

        atb::infer::LinearParam linearParam;
        linearParam.transposeA = false;
        linearParam.transposeB = true;
        linearParam.hasBias = false;
        CREATE_OPERATION(linearParam, &linearNode.operation);
        linearNode.inTensorIds = {IN_INPUT, IN_WEIGHT};
        linearNode.outTensorIds = {outTensorId};
    }


    atb::Node &allReduceNode = opGraph.nodes.at(nodeId++);
    // std::string backend;
    // if (param.isPrefill)
    // {
    //     backend = "hccl";
    // } else {
    //     backend = "lccl";
    // }
    atb::infer::AllReduceParam allReduceParam = {param.rank, param.rankSize, param.rankRoot,
                                                 "sum",      param.backend,  param.hcclComm};
    CREATE_OPERATION(allReduceParam, &allReduceNode.operation);
    allReduceNode.inTensorIds = {outTensorId};
    allReduceNode.outTensorIds = {outTensorId};

    CREATE_OPERATION(opGraph, operation);
    return atb::NO_ERROR;
}

atb::Status CreateLinearParallelW8A16(const LinearParallelW8A16Param &param, atb::Operation **operation, bool transB)
{
    if (param.isQuant == false && param.isPrefill) { // FP16 & prefill 使用lcoc
        return CreateLinearParallelLcoc(param, operation);
    }

    return CreateLinearParallelW8A16Inner(param, operation, transB); // 原有逻辑，W8A16量化，及浮点
}

}// namespace contrib
}// namespace atb_speed

