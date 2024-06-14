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
#ifndef ATB_SPEED_PLUGIN_ACLNN_W16A16_OPERATION_V2_H
#define ATB_SPEED_PLUGIN_ACLNN_W16A16_OPERATION_V2_H
#include "acl_nn_operation.h"

namespace atb_speed {
namespace common {
class W16A16Operation : public AclNnOperation {
public:
    explicit W16A16Operation(const std::string &name, bool transB);
    ~W16A16Operation() override;
    atb::Status InferShape(const atb::SVector<atb::TensorDesc> &inTensorDescs,
                           atb::SVector<atb::TensorDesc> &outTensorDescs) const override;
    uint32_t GetInputNum() const override;
    uint32_t GetOutputNum() const override;

private:
    int CreateAclTensors(const atb::VariantPack &variantPack, AclNnTask &task) override;
    int CallAclGetWorkspace(AclNnTask &task, uint64_t &workspaceSize)override;
    int CallAclExecute(uint8_t *workspace, uint64_t workspaceSize, aclOpExecutor *aclExecutor,
                               aclrtStream stream) override;
    AclNnTensor CreateTensor(atb::Tensor atbTensor);
};
} // namespace common
} // namespace atb_speed
#endif

