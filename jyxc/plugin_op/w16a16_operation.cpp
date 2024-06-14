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
#include "w16a16_operation.h"
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include <syscall.h>
#include <unistd.h>
#include "acl/acl.h"
#include "atb_speed/log.h"
#include "aclnnop/level2/aclnn_mm.h"

namespace atb_speed {
namespace common {
const int DIM0 = 0;
const int DIM1 = 1;
const int DIM2 = 2;
const int DIM3 = 3;
const int DIM4 = 4;
const int NUM1 = 1;
const int NUM2 = 2;
const int NUM3 = 3;
const int NUM4 = 4;

W16A16Operation::W16A16Operation(const std::string &name, bool transB) : AclNnOperation(name)  {
    this->transB = transB;
}

W16A16Operation::~W16A16Operation() {}

atb::Status W16A16Operation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                       atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    // outTensorDescs.at(0).dtype = aclDataType::ACL_FLOAT16;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;

    if (inTensorDescs.at(0).shape.dimNum == DIM3) {
        ATB_LOG(FATAL) << "[input0 dimNum = 3] CHECK W16A16_OP inputs shape: [input0]"
                       << inTensorDescs.at(DIM0).shape.dims[DIM0] << ", "
                       << inTensorDescs.at(DIM0).shape.dims[DIM1] << ", " << inTensorDescs.at(DIM0).shape.dims[DIM2];
        ATB_LOG(FATAL) << "[input0 dimNum = 3] CHECK W16A16_OP inputs shape: [input1]"
                       << inTensorDescs.at(DIM1).shape.dims[DIM0] << ", " << inTensorDescs.at(DIM1).shape.dims[DIM1];
        outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
        outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM1];
        outTensorDescs.at(DIM0).shape.dims[DIM2] = inTensorDescs.at(DIM3).shape.dims[DIM1];
    } else if (inTensorDescs.at(0).shape.dimNum == DIM2) {
        ATB_LOG(FATAL) << "[input0 dimNum = 2] CHECK W16A16_OP inputs shape: [input0]"
                       << inTensorDescs.at(DIM0).shape.dims[DIM0] << ", " << inTensorDescs.at(DIM0).shape.dims[DIM1];
        ATB_LOG(FATAL) << "[input0 dimNum = 2] CHECK W16A16_OP inputs shape: [input1]"
                       << inTensorDescs.at(DIM1).shape.dims[DIM0] << ", " << inTensorDescs.at(DIM1).shape.dims[DIM1];
        ATB_LOG(FATAL) << "transB" << transB;
        
        outTensorDescs.at(0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
        outTensorDescs.at(0).shape.dims[DIM1] = inTensorDescs.at(DIM1).shape.dims[DIM1];

        ATB_LOG(FATAL) << "[output0 dimNum = 2] CHECK W16A16_OP inputs shape:"
                       << outTensorDescs.at(DIM0).shape.dims[DIM0] << ", " << outTensorDescs.at(DIM0).shape.dims[DIM1];
    } 

    ATB_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t W16A16Operation::GetInputNum() const { return NUM2; }

uint32_t W16A16Operation::GetOutputNum() const { return NUM1; }

int W16A16Operation::CreateAclTensors(const atb::VariantPack &variantPack, AclNnTask &task)
{
    ATB_LOG(INFO) << opName_ << " CreateAclTensor start";
    task.aclInTensors.resize(NUM2);
    for (size_t i = 0; i < task.aclInTensors.size(); ++i) {
        task.aclInTensors[i] = CreateTensor(variantPack.inTensors.at(i));
    }

    ATB_LOG(INFO) << opName_ << " Create aclInTensor end";

    task.aclOutTensors.resize(NUM1);
    for (size_t i = 0; i < task.aclOutTensors.size(); ++i) {
        task.aclOutTensors[i] = CreateTensor(variantPack.outTensors.at(i));
    }

    ATB_LOG(INFO) << opName_ << " Create aclOutTensor end";
    ATB_LOG(INFO) << opName_ << " CreateAclTensor end";
    return 0;
}

int W16A16Operation::CallAclGetWorkspace(AclNnTask &task,
                                        uint64_t &workspaceSize)
{
    ATB_LOG(INFO) << opName_ << " aclnnWeightQuantBatchMatmulV2GetWorkspaceSize start";
    int ret = aclnnMmGetWorkspaceSize(task.aclInTensors.at(0).tensor, task.aclInTensors.at(1).tensor, 
        task.aclOutTensors.at(0).tensor, 0, &workspaceSize,  &task.aclExecutor);
    ATB_LOG(INFO) << opName_ << " aclnnWeightQuantBatchMatmulV2GetWorkspaceSize end, ret:" << ret
                  << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << task.aclExecutor;

    return ret;
}

int W16A16Operation::CallAclExecute(uint8_t *workspace, uint64_t workspaceSize,
                                   aclOpExecutor *aclExecutor, aclrtStream stream)
{
    ATB_LOG(INFO) << opName_ << " aclnnWeightQuantBatchMatmulV2 start";
    int ret = aclnnMm(workspace, workspaceSize, aclExecutor, stream);
    ATB_LOG(INFO) << opName_ << " aclnnWeightQuantBatchMatmulV2 end, ret:" << ret;
    return ret;
}

AclNnTensor W16A16Operation::CreateTensor(atb::Tensor atbTensor)
{
    AclNnTensor aclNnTensor;
    aclNnTensor.needUpdateTensorDataPtr = true;
    aclNnTensor.atbTensor = atbTensor;
    if (aclNnTensor.atbTensor.desc.shape.dimNum == DIM3) {
        aclNnTensor.atbTensor.desc.shape.dimNum = DIM2;
        aclNnTensor.atbTensor.desc.shape.dims[DIM0] = atbTensor.desc.shape.dims[DIM0] * atbTensor.desc.shape.dims[DIM1];
        aclNnTensor.atbTensor.desc.shape.dims[DIM1] = atbTensor.desc.shape.dims[DIM2];
    }

    return aclNnTensor;
}
} // namespace common
} // namespace atb_speed

