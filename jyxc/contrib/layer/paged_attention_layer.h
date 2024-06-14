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
#ifndef CONTRIB_PAGED_ATTENTION_LAYER_H
#define CONTRIB_PAGED_ATTENTION_LAYER_H

#include <vector>
#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "atb_speed/log.h"
#include "nlohmann/json.hpp"

namespace atb_speed {
namespace contrib {
struct PagedAttentionLayerParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int dk = 0;
    int rank = 0;
    int rankSize = 1;
    std::string model = "";
    std::string backend = "hccl";
    int numHeadsPerPartition = 0;
    std::vector<int> seqLen;
    std::vector<int> tokenOffset;
    int rotaryCoeff = 2;
    bool transposedWeight = false;
    bool isPrefill = false;
    bool isBF16 = false;
    bool isQuant = true;
    std::string hcomm_info = "";
};

enum PagedAttentionLayerTensorId : int {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_QKVMIXDWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPGATEUPWEIGHT,
    IN_MLPDOWNWEIGHT,
    IN_QKV_SCALE,
    IN_QKV_OFFSET,
    IN_SELFOUTLINEAR_SCALE,
    IN_SELFOUTLINEAR_OFFSET,
    IN_MLPGATEUP_SCALE,
    IN_MLPGATEUP_OFFSET,
    IN_MLPDOWN_SCALE,
    IN_MLPDOWN_OFFSET,
    IN_ATTENTIONMASK,
    IN_K_CACHE,
    IN_V_CACHE,
    IN_BLOCK_TABLES,
    IN_SLOTS,
    IN_SEQLEN,
    OUT_LAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDQKV,
    INTERMIDATE_MIXEDKV,
    INTERMIDATE_MIXEDQ,
    INTERMIDATE_MIXEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_ATTENTIONOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_SELFRESIDUALADDOUT, 
    INTERMIDATE_RSTD
};

atb::Status PagedAttentionLayer(const PagedAttentionLayerParam &param, atb::Operation **operation);

static atb::Operation *CreatePagedAttentionLayer(const nlohmann::json &paramJson)
{
    PagedAttentionLayerParam param;
    if (paramJson.find("rmsNormEps") != paramJson.end()) {
        param.rmsNormEps = paramJson["rmsNormEps"].get<float>();
    }
    if (paramJson.find("headNum") != paramJson.end()) {
        param.headNum = paramJson["headNum"].get<int>();
    }
    if (paramJson.find("dk") != paramJson.end()) {
        param.dk = paramJson["dk"].get<int>();
    }
    if (paramJson.find("rank") != paramJson.end()) {
        param.rank = paramJson["rank"].get<int>();
    }
    if (paramJson.find("rankSize") != paramJson.end()) {
        param.rankSize = paramJson["rankSize"].get<int>();
    }
    if (paramJson.find("backend") != paramJson.end()) {
        param.backend = paramJson["backend"].get<std::string>();
    }
    if (paramJson.find("model") != paramJson.end()) {
        param.model = paramJson["model"].get<std::string>();
    }
    if (paramJson.contains("numHeadsPerPartition")) {
        param.numHeadsPerPartition = paramJson["numHeadsPerPartition"].get<int>();
    }
    if (paramJson.contains("rotaryCoeff")) {
        param.rotaryCoeff = paramJson["rotaryCoeff"].get<int>();
    }

    ATB_LOG(INFO) << "PagedAttentionLayerParam params headNum:" << param.headNum << ", rmsNormEps:" << param.rmsNormEps
                  << ", dk:" << param.dk << ", model:" << param.model << ", rank:" << param.rank
                  << ", rankSize:" << param.rankSize << ", backend:" << param.backend
                  << ", numHeadsPerPartition:" << param.numHeadsPerPartition;
    atb::Operation *op;
    PagedAttentionLayer(param, &op);
    return op;
}

class PagedAttentionLayerBinder : public HostTensorBinder {
public:
    PagedAttentionLayerBinder();
    virtual ~PagedAttentionLayerBinder();
    void ParseParam(const nlohmann::json &paramJson) override;
    void BindTensor(atb::VariantPack &variantPack) override;

private:
    std::vector<int> tokenOffset_;
    std::vector<int> seqLen_;
    int32_t layerId_ = 0;
};

} // namespace contrib
} // namespace atb_speed
#endif


