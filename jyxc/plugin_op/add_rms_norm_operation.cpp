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
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include <syscall.h>
#include <unistd.h>

#include "acl/acl.h"
#include "aclnnop/aclnn_add_rms_norm.h"
#include "atb_speed/log.h"
#include "add_rms_norm_operation.h"

namespace atb_speed {
namespace common {
const int DIM0 = 0;
const int DIM1 = 1;
const int DIM2 = 2;
const int DIM3 = 3;
const int NUM1 = 1;
const int NUM2 = 2;
const int NUM3 = 3;
const int NUM4 = 4;

AddRmsNormOperation::AddRmsNormOperation(const std::string &name, float epsilon) : AclNnOperation(name)
{
    this->epsilon = epsilon;
}

AddRmsNormOperation::~AddRmsNormOperation() {}

atb::Status AddRmsNormOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                            atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_LOG(INFO) << opName_ << " infer shape start";
    for (size_t i = 0; i < outTensorDescs.size(); i++) {
        outTensorDescs.at(i).format = inTensorDescs.at(0).format;
        if (i == NUM1) {
            outTensorDescs.at(i).dtype = aclDataType::ACL_FLOAT;
        } else {
            outTensorDescs.at(i).dtype = inTensorDescs.at(0).dtype;
        }

        outTensorDescs.at(i).shape.dimNum = inTensorDescs.at(0).shape.dimNum;

        if (inTensorDescs.at(0).shape.dimNum == DIM3) {
            ATB_LOG(FATAL) << "[input0 dimNum = 3] CHECK W8A16_OP inputs shape: [input0]"
                           << inTensorDescs.at(0).shape.dims[DIM0] << ", " << inTensorDescs.at(0).shape.dims[DIM1]
                           << ", " << inTensorDescs.at(0).shape.dims[DIM2];
            outTensorDescs.at(i).shape.dims[DIM0] = inTensorDescs.at(0).shape.dims[DIM0];
            outTensorDescs.at(i).shape.dims[DIM1] = inTensorDescs.at(0).shape.dims[DIM1];
            outTensorDescs.at(i).shape.dims[DIM2] = inTensorDescs.at(0).shape.dims[DIM1];
        } else if (inTensorDescs.at(0).shape.dimNum == DIM2) {
            ATB_LOG(FATAL) << "[input0 dimNum = 2] CHECK W8A16_OP inputs shape: [input0]"
                           << inTensorDescs.at(0).shape.dims[DIM0] << ", "
                           << inTensorDescs.at(0).shape.dims[DIM1];
            if (i == NUM1) {
                outTensorDescs.at(i).shape.dims[DIM0] = inTensorDescs.at(0).shape.dims[DIM0];
                outTensorDescs.at(i).shape.dims[DIM1] = 1;
            } else {
                outTensorDescs.at(i).shape.dims[DIM0] = inTensorDescs.at(0).shape.dims[DIM0];
                outTensorDescs.at(i).shape.dims[DIM1] = inTensorDescs.at(0).shape.dims[DIM1];
            }
        } else {
            ATB_LOG(ERROR) << opName_ << " invalid dim num:" << inTensorDescs.at(DIM0).shape.dimNum;
        }
    }

    ATB_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t AddRmsNormOperation::GetInputNum() const { return NUM3; }

uint32_t AddRmsNormOperation::GetOutputNum() const { return NUM3; }

int AddRmsNormOperation::CreateAclTensors(const atb::VariantPack &variantPack, AclNnTask &task)
{
    ATB_LOG(INFO) << opName_ << " CreateAclTensor start";
    task.aclInTensors.resize(variantPack.inTensors.size());
    for (size_t i = 0; i < task.aclInTensors.size(); ++i) {
        task.aclInTensors[i] = CreateTensor(variantPack.inTensors.at(i));
    }

    ATB_LOG(INFO) << opName_ << " Create aclInTensor end";

    task.aclOutTensors.resize(variantPack.outTensors.size());
    for (size_t i = 0; i < task.aclOutTensors.size(); ++i) {
        task.aclOutTensors[i] = CreateTensor(variantPack.outTensors.at(i));
    }

    ATB_LOG(INFO) << opName_ << " Create aclOutTensor end";
    ATB_LOG(INFO) << opName_ << " CreateAclTensor end";
    return 0;
}

int AddRmsNormOperation::CallAclGetWorkspace(AclNnTask &task, uint64_t &workspaceSize)
{
    ATB_LOG(INFO) << opName_ << " aclnnAddRmsNormGetWorkspaceSize start";
    int ret = aclnnAddRmsNormGetWorkspaceSize(task.aclInTensors.at(0).tensor,
        task.aclInTensors.at(1).tensor,
        task.aclInTensors.at(2).tensor,
        this->epsilon,
        task.aclOutTensors.at(0).tensor,
        task.aclOutTensors.at(1).tensor,
        task.aclOutTensors.at(2).tensor,
        &workspaceSize,
        &task.aclExecutor);
    ATB_LOG(INFO) << opName_ << " aclnnAddRmsNormGetWorkspaceSize end, ret:" << ret
                  << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << task.aclExecutor;

    return ret;
}

int AddRmsNormOperation::CallAclExecute(uint8_t *workspace, uint64_t workspaceSize, aclOpExecutor *aclExecutor,
                                        aclrtStream stream)
{
    ATB_LOG(INFO) << opName_ << " aclnnAddRmsNorm start";
    int ret = aclnnAddRmsNorm(workspace, workspaceSize, aclExecutor, stream);
    ATB_LOG(INFO) << opName_ << " aclnnAddRmsNorm end, ret:" << ret;
    return ret;
}

AclNnTensor AddRmsNormOperation::CreateTensor(atb::Tensor atbTensor)
{
    AclNnTensor aclNnTensor;
    aclNnTensor.needUpdateTensorDataPtr = true;
    aclNnTensor.atbTensor = atbTensor;
    return aclNnTensor;
}
} // namespace common
} // namespace atb_speed

