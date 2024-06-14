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
#include "matmul_allreduce_operation.h"
#include <cstring>
#include <iostream>
#include <securec.h>
#include <sstream>
#include <syscall.h>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnnop/aclnn_matmul_all_reduce.h"
#include "atb_speed/log.h"

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
const int NUM5 = 5;

MatmulAllreduceOperation::MatmulAllreduceOperation(const std::string &name, 
                                    const std::string &hcomm_info) : AclNnOperation(name), hcomm_info_(hcomm_info) {}

MatmulAllreduceOperation::~MatmulAllreduceOperation() {}

atb::Status MatmulAllreduceOperation::InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                       atb::SVector<atb::TensorDesc> &outTensorDescs) const
{
    ATB_LOG(INFO) << opName_ << " infer shape start";
    outTensorDescs.at(0).format = inTensorDescs.at(0).format;
    outTensorDescs.at(0).dtype = inTensorDescs.at(0).dtype;
    outTensorDescs.at(0).shape.dimNum = inTensorDescs.at(0).shape.dimNum;

    if (inTensorDescs.at(0).shape.dimNum == DIM3) {
        ATB_LOG(FATAL) << "[input0 dimNum = 3] CHECK inputs shape: [input0]"
                       << inTensorDescs.at(DIM0).shape.dims[DIM0] << ", "
                       << inTensorDescs.at(DIM0).shape.dims[DIM1] << ", " << inTensorDescs.at(DIM0).shape.dims[DIM2];
        ATB_LOG(FATAL) << "[input0 dimNum = 3] CHECK inputs shape: [input1]"
                       << inTensorDescs.at(DIM1).shape.dims[DIM0] << ", " << inTensorDescs.at(DIM1).shape.dims[DIM1];
        outTensorDescs.at(DIM0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
        outTensorDescs.at(DIM0).shape.dims[DIM1] = inTensorDescs.at(DIM0).shape.dims[DIM1];
        outTensorDescs.at(DIM0).shape.dims[DIM2] = inTensorDescs.at(DIM1).shape.dims[DIM0];
    } else if (inTensorDescs.at(0).shape.dimNum == DIM2) {
        ATB_LOG(FATAL) << "[input0 dimNum = 2] CHECK inputs shape: [input0]"
                       << inTensorDescs.at(DIM0).shape.dims[DIM0] << ", " << inTensorDescs.at(DIM0).shape.dims[DIM1];
        ATB_LOG(FATAL) << "[input0 dimNum = 2] CHECK inputs shape: [input1]"
                       << inTensorDescs.at(DIM1).shape.dims[DIM0] << ", " << inTensorDescs.at(DIM1).shape.dims[DIM1];
        outTensorDescs.at(0).shape.dims[DIM0] = inTensorDescs.at(DIM0).shape.dims[DIM0];
        outTensorDescs.at(0).shape.dims[DIM1] = inTensorDescs.at(DIM1).shape.dims[DIM0];
    } else {
        ATB_LOG(ERROR) << opName_ << " invalid dim num:" << inTensorDescs.at(DIM0).shape.dimNum;
    }
    ATB_LOG(INFO) << opName_ << " infer shape end";
    return 0;
}

uint32_t MatmulAllreduceOperation::GetInputNum() const { return NUM2; }

uint32_t MatmulAllreduceOperation::GetOutputNum() const { return NUM1; }

int MatmulAllreduceOperation::CreateAclTensors(const atb::VariantPack &variantPack, AclNnTask &task)
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
    ATB_LOG(INFO) << opName_ << " CreateAclTensor end";
    return 0;
}

int MatmulAllreduceOperation::CallAclGetWorkspace(AclNnTask &task,
                                        uint64_t &workspaceSize)
{
    ATB_LOG(INFO) << opName_ << " aclnnMatmulAllReduceGetWorkspaceSize start";
    ATB_LOG(FATAL) << opName_ << " hcomm_info = " << hcomm_info_;
    int ret = aclnnMatmulAllReduceGetWorkspaceSize(
        task.aclInTensors.at(0).tensor, task.aclInTensors.at(1).tensor, nullptr,
        const_cast<char *>(hcomm_info_.c_str()), "sum", 0, 1,
        task.aclOutTensors.at(0).tensor, &workspaceSize, &task.aclExecutor);
    ATB_LOG(INFO) << opName_ << " aclnnMatmulAllReduceGetWorkspaceSize end, ret:"
                  << ret << ", workspaceSize:" << workspaceSize << ", aclExecutor:" << task.aclExecutor;
    return ret;
}

int MatmulAllreduceOperation::CallAclExecute(uint8_t *workspace, uint64_t workspaceSize,
                                   aclOpExecutor *aclExecutor, aclrtStream stream)
{
    ATB_LOG(INFO) << opName_ << " aclnnMatmulAllReduce start";
    int ret = aclnnMatmulAllReduce(workspace, workspaceSize, aclExecutor, stream);
    ATB_LOG(INFO) << opName_ << " aclnnMatmulAllReduce end, ret:" << ret;
    return ret;
}

AclNnTensor MatmulAllreduceOperation::CreateTensor(atb::Tensor atbTensor)
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

