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
#include "paged_attention_model.h"
#include "atb/atb_infer.h"
#include "atb_speed/log.h"
#include "models/contrib/layer/paged_attention_layer.h"
#include "nlohmann/json.hpp"
#include "parallel_lmhead.h"
#include "atb_speed/utils/model_factory.h"

namespace atb_speed {
namespace contrib {

REGISTER_MODEL(contrib, PagedAttentionModel);

const int WEIGHT_COUNT_PER_LAYER = 14;
const int OPERATION_COUNT_BEFORE_LAYER = 0;
const int OPERATION_COUNT_AFTER_LAYER = 2;
const int WEIGHT_COUNT_BEFORE_LAYER = 0;
const int FINALNORMNODE_WEIGHT_COUNT = 1;
const int OUT_LM_HEAD_WEIGHT_COUNT = 1;
const int INTERMEDIATETENSOR_COUNT_BEFORE_LAYER = 0;

enum InTensorId : int {
    IN_TENSOR_HIDDENSTATES = 0,
    IN_TENSOR_ATTENTIONMASK,
    IN_TENSOR_BLOCK_TABLES,
    IN_TENSOR_SLOTS,
    IN_TENSOR_SEQLEN,
    IN_TENSOR_LOGTIS_INDICES,
    IN_TENSOR_MAX,
};

enum OutTensorId : int {
    OUT_TENSOR_HIDDENSTATES = 0,
    OUT_TENSOR_MAX,
};

void PagedAttentionModel::Param::FromString(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    rmsNormEps = paramJson["rmsNormEps"].get<double>();
    headNum = paramJson["headNum"].get<int>();
    dk = paramJson["dk"].get<int>();
    layerNum = paramJson["layerNum"].get<int>();
    rank = paramJson["rank"].get<int>();
    rankSize = paramJson["rankSize"].get<int>();
    numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int>();
    if (paramJson.contains("isPrefill")) {
        isPrefill = paramJson["isPrefill"].get<bool>();
    }
    if (paramJson.contains("transposedWeight")) {
        transposedWeight = paramJson["transposedWeight"].get<bool>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        backend = paramJson["backend"].get<std::string>();
    }
    for (auto item : paramJson["tokenOffset"]) {
        tokenOffset.push_back(item.get<int>());
    }
    for (auto item : paramJson["seqLen"]) {
        seqLen.push_back(item.get<int>());
    }
    if (paramJson.contains("rotaryCoeff")) {
        rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    }
    if (paramJson.contains("isBF16")) {
        isBF16 = paramJson["isBF16"].get<bool>();
    }
    if (paramJson.contains("hcomm_info")) {
        hcomm_info = paramJson["hcomm_info"].get<std::string>();
    }
    ATB_LOG(INFO) << "PagedAttentionModel param rmsNormEps:" << rmsNormEps << ", headNum:" << headNum
                  << ", dk:" << dk << ", layerNum:" << layerNum << ", transposedWeight:" << transposedWeight
                  << ", rank:" << rank << ", rankSize:" << rankSize << ", backend:" << backend
                  << ", numHeadsPerPartition:" << numHeadsPerPartition << ", isBF16:" << isBF16;
}

PagedAttentionModel::PagedAttentionModel(const std::string &param) : Model("PagedAttentionModel", param)
{
    param_.FromString(param);
    modelName_ += param_.isPrefill ? "_Prefill" : "_Decoder";
}

PagedAttentionModel::~PagedAttentionModel() = default;

uint32_t PagedAttentionModel::GetInputNum() const { return graph_.inTensors.size(); }

uint32_t PagedAttentionModel::GetOutputNum() const { return graph_.outTensors.size(); }

atb::Status PagedAttentionModel::InferShape(const std::vector<atb::TensorDesc> &inTensorDescs,
                                            std::vector<atb::TensorDesc> &outTensorDescs)
{
    ATB_LOG(INFO) << "Enter PagedAttentionModel InferShape";
    if (outTensorDescs.size() != GetOutputNum()) {
        return atb::ERROR_INVALID_GRAPH;
    }

    const int64_t outDim = graph_.weightTensors.at(graph_.weightTensors.size() - 1).desc.shape.dims[0];
    outTensorDescs.at(0) = inTensorDescs.at(0);
    auto outDimNum = inTensorDescs.at(0).shape.dimNum;
    outTensorDescs.at(0).shape.dims[outDimNum - 1] = outDim * param_.rankSize;

    // change first dim
    if (param_.isPrefill) {
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(IN_TENSOR_LOGTIS_INDICES).shape.dims[0];
    }

    return atb::NO_ERROR;
}

int64_t PagedAttentionModel::BuildGraph()
{
    ATB_LOG(INFO) << "Enter PagedAttentionModel BuildGraph";
    const int weightTensorSize = WEIGHT_COUNT_BEFORE_LAYER + WEIGHT_COUNT_PER_LAYER * 1 +
                                 FINALNORMNODE_WEIGHT_COUNT + OUT_LM_HEAD_WEIGHT_COUNT;
    graph_.weightTensors.resize(weightTensorSize);

    graph_.kCacheTensors.resize(param_.layerNum);
    graph_.vCacheTensors.resize(param_.layerNum);

    graph_.inTensors.resize(IN_TENSOR_MAX);
    graph_.outTensors.resize(OUT_TENSOR_MAX);

    const int nodeSize = OPERATION_COUNT_BEFORE_LAYER + param_.layerNum + OPERATION_COUNT_AFTER_LAYER;
    ATB_LOG(INFO) << "PagedAttentionModel nodeSize is " << nodeSize;
    graph_.nodes.resize(nodeSize);

    const int internalTensorSize = graph_.nodes.size() - 1;
    graph_.internalTensors.resize(internalTensorSize);

    int nodeId = 0;
    atb::Operation *op = nullptr;
    atb::Tensor *firstInTensor = &graph_.inTensors.at(0);

    //////////////////////////////   LAYER 0 (FIRST LAYER) ///////////////////////////////////////
    ATB_LOG(INFO) << "FIRST LAYER ";
    auto &layerNode = graph_.nodes.at(nodeId++);
    atb_speed::contrib::PagedAttentionLayerParam layerParam;
    layerParam.rmsNormEps = param_.rmsNormEps;
    layerParam.headNum = param_.headNum;
    layerParam.dk = param_.dk;
    layerParam.transposedWeight = param_.transposedWeight;
    layerParam.rank = param_.rank;
    layerParam.rankSize = param_.rankSize;
    layerParam.numHeadsPerPartition = param_.numHeadsPerPartition;
    layerParam.isPrefill = param_.isPrefill;
    layerParam.backend = param_.backend;
    layerParam.isBF16 = param_.isBF16;
    layerParam.isQuant = false;
    layerParam.hcomm_info = param_.hcomm_info;
    atb_speed::contrib::PagedAttentionLayer(layerParam, &op);
    layerNode.operation.reset(op);
    layerNode.inTensors.resize(layerNode.operation->GetInputNum());
    size_t inTensorId = 0;
    layerNode.inTensors.at(inTensorId++) = firstInTensor;
    for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
        layerNode.inTensors.at(inTensorId++) = 
            &graph_.weightTensors.at(0 * WEIGHT_COUNT_PER_LAYER + weightTensorId + WEIGHT_COUNT_BEFORE_LAYER);
    }
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
    layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(0);
    layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(0);
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);
    layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen
    if (param_.isPrefill) {
        layerNode.outTensors = {firstInTensor};
        // layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER)};
    } else {
        layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER)};
    }
    ATB_LOG(INFO) << "FIRST LAYER5";
    firstInTensor = layerNode.outTensors.at(0);
    ///////////////////////////////////////////////////////////////////////////////////////////////
    ATB_LOG(INFO) << "FIRST LAYER6";
    for (int layerId = 1; layerId < param_.layerNum - 1; ++layerId) {
    //for (int layerId = 0; layerId < param_.layerNum; ++layerId) {
        auto &layerNode = graph_.nodes.at(nodeId++);

        atb_speed::contrib::PagedAttentionLayerParam layerParam;
        layerParam.rmsNormEps = param_.rmsNormEps;
        layerParam.headNum = param_.headNum;
        layerParam.dk = param_.dk;
        layerParam.transposedWeight = param_.transposedWeight;
        layerParam.rank = param_.rank;
        layerParam.rankSize = param_.rankSize;
        layerParam.numHeadsPerPartition = param_.numHeadsPerPartition;
        layerParam.isPrefill = param_.isPrefill;
        layerParam.backend = param_.backend;
        layerParam.isBF16 = param_.isBF16;
        layerParam.isQuant = false;
        layerParam.hcomm_info = param_.hcomm_info;
        atb_speed::contrib::PagedAttentionLayer(layerParam, &op);
        layerNode.operation.reset(op);
        layerNode.inTensors.resize(layerNode.operation->GetInputNum());

        size_t inTensorId = 0;
        layerNode.inTensors.at(inTensorId++) = firstInTensor;
        for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
            layerNode.inTensors.at(inTensorId++) =
                &graph_.weightTensors.at(0 * WEIGHT_COUNT_PER_LAYER + weightTensorId + WEIGHT_COUNT_BEFORE_LAYER);
        }
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
        layerNode.inTensors.at(inTensorId++) = &graph_.kCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.vCacheTensors.at(layerId);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);
        layerNode.inTensors.at(inTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen

        if (param_.isPrefill) {
            layerNode.outTensors = {firstInTensor};
            // layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};
        } else {
            layerNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + layerId)};
        }
        firstInTensor = layerNode.outTensors.at(0);
    }

    //////////////////////////////   LAYER 87 (LAST LAYER) ///////////////////////////////////////
    auto &layerLastNode = graph_.nodes.at(nodeId++);
    atb_speed::contrib::PagedAttentionLayerParam layerLastParam;
    layerLastParam.rmsNormEps = param_.rmsNormEps;
    layerLastParam.headNum = param_.headNum;
    layerLastParam.dk = param_.dk;
    layerLastParam.transposedWeight = param_.transposedWeight;
    layerLastParam.rank = param_.rank;
    layerLastParam.rankSize = param_.rankSize;
    layerLastParam.numHeadsPerPartition = param_.numHeadsPerPartition;
    layerLastParam.isPrefill = param_.isPrefill;
    layerLastParam.backend = param_.backend;
    layerLastParam.isBF16 = param_.isBF16;
    layerLastParam.isQuant = false;
    layerLastParam.hcomm_info = param_.hcomm_info;    
    atb_speed::contrib::PagedAttentionLayer(layerLastParam, &op);
    layerLastNode.operation.reset(op);
    layerLastNode.inTensors.resize(layerLastNode.operation->GetInputNum());
    size_t inLastTensorId = 0;
    layerLastNode.inTensors.at(inLastTensorId++) = firstInTensor;
    for (size_t weightTensorId = 0; weightTensorId < WEIGHT_COUNT_PER_LAYER; ++weightTensorId) {
        layerLastNode.inTensors.at(inLastTensorId++) =
            &graph_.weightTensors.at(0 * WEIGHT_COUNT_PER_LAYER + weightTensorId + WEIGHT_COUNT_BEFORE_LAYER);
    }
    layerLastNode.inTensors.at(inLastTensorId++) = &graph_.inTensors.at(IN_TENSOR_ATTENTIONMASK);
    layerLastNode.inTensors.at(inLastTensorId++) = &graph_.kCacheTensors.at(param_.layerNum - 1);
    layerLastNode.inTensors.at(inLastTensorId++) = &graph_.vCacheTensors.at(param_.layerNum - 1);
    layerLastNode.inTensors.at(inLastTensorId++) = &graph_.inTensors.at(IN_TENSOR_BLOCK_TABLES);
    layerLastNode.inTensors.at(inLastTensorId++) = &graph_.inTensors.at(IN_TENSOR_SLOTS);
    layerLastNode.inTensors.at(inLastTensorId++) = &graph_.inTensors.at(IN_TENSOR_SEQLEN); // seqLen

    if (param_.isPrefill) {
        layerLastNode.outTensors = {firstInTensor};
        // layerLastNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + 87)};
    } else {
        layerLastNode.outTensors = {&graph_.internalTensors.at(INTERMEDIATETENSOR_COUNT_BEFORE_LAYER + 87)};
    }

    firstInTensor = layerLastNode.outTensors.at(0);
    ///////////////////////////////////////////////////////////////////////////////////////////////

    auto &finalNormNode = graph_.nodes.at(nodeId++);
    atb::infer::RmsNormParam finalNormParam;
    finalNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    finalNormParam.normParam.epsilon = param_.rmsNormEps;
    CREATE_OPERATION(finalNormParam, &op);
    finalNormNode.operation.reset(op);
    const int finalLayerNormWeightTensorId =
        graph_.weightTensors.size() - FINALNORMNODE_WEIGHT_COUNT - OUT_LM_HEAD_WEIGHT_COUNT;
    const int finalLayerNormOutTensorId = internalTensorSize - 1;
    finalNormNode.inTensors = {firstInTensor, &graph_.weightTensors.at(finalLayerNormWeightTensorId)};
    if (param_.isPrefill) {
        finalNormNode.outTensors = {firstInTensor};
        // finalNormNode.outTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};
    } else {
        finalNormNode.outTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId)};
    }
    auto &lmHeadNode = graph_.nodes.at(nodeId++);
    atb_speed::common::ParallelLmHeadParam lmHeadParam;
    lmHeadParam.rank = param_.rank;
    lmHeadParam.rankSize = param_.rankSize;
    lmHeadParam.unpadInputs = true;
    lmHeadParam.gatherAhead = param_.isPrefill;
    lmHeadParam.backend = param_.backend;
    lmHeadParam.isBF16 = param_.isBF16;
    ParallelLmHead(lmHeadParam, &op);
    lmHeadNode.operation.reset(op);
    const int finalLinearWeightTensorId = graph_.weightTensors.size() - OUT_LM_HEAD_WEIGHT_COUNT;
    if (param_.isPrefill) {
        lmHeadNode.inTensors = {firstInTensor,
        // lmHeadNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                                &graph_.weightTensors.at(finalLinearWeightTensorId),
                                &graph_.inTensors.at(IN_TENSOR_LOGTIS_INDICES)};
    } else {
        lmHeadNode.inTensors = {&graph_.internalTensors.at(finalLayerNormOutTensorId),
                                &graph_.weightTensors.at(finalLinearWeightTensorId)};
    }
    lmHeadNode.outTensors = {&graph_.outTensors.at(0)};

    return atb::NO_ERROR;
}

atb::Status PagedAttentionModel::ParseParam(const std::string &param)
{
    nlohmann::json paramJson = nlohmann::json::parse(param);
    seqLen_.clear();
    for (auto item : paramJson["seqLen"]) {
        seqLen_.push_back(item.get<int>());
    }
    ATB_LOG(INFO) << "PagedAttentionModel ParseParam seqLen: " << seqLen_.capacity();
    return atb::NO_ERROR;
}

atb::Status PagedAttentionModel::BindParamHostTensor(uint32_t nodeId)
{
    ATB_LOG(INFO) << "BindParamHostTensor";
    ATB_LOG(INFO) << "nodeId = " << nodeId;
    ATB_LOG(INFO) << "param_.layerNum = " << param_.layerNum;

    if (nodeId < OPERATION_COUNT_BEFORE_LAYER || nodeId >= OPERATION_COUNT_BEFORE_LAYER + param_.layerNum) {
        return atb::NO_ERROR;
    }
    auto &node = graph_.nodes.at(nodeId);
    const uint32_t seqLenTensorId = atb_speed::contrib::PagedAttentionLayerTensorId::IN_SEQLEN;
    node.variantPack.inTensors.at(seqLenTensorId).hostData = seqLen_.data();
    ATB_LOG(INFO) << "BindParamHostTensor end";
    return atb::NO_ERROR;
}
} // namespace contrib
} // namespace atb_speed


