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
#ifndef ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_H
#define ATB_SPEED_PLUGIN_ACLNN_NN_OPERATION_H
#include <string>
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include "atb/operation.h"

namespace atb_speed {
namespace common {
constexpr size_t SVECTOR_SIZE = 8;

struct AclNnTensor {
    atb::Tensor atbTensor;
    atb::SVector<int64_t> strides;
    aclTensor *tensor = nullptr;
    bool needUpdateTensorDataPtr = false;
    int CreateTensor(const std::string &opName);
    int CreateTensorTrans(const std::string &opName, bool destroy);
    int CreateTensorTransNZ(const std::string &opName, bool destroy);
    int InitTensor(void *executor, const std::string &opName, const size_t index, bool isInput);
    bool isDestroy = true;
};

struct AclNnTask {
    atb::SVector<AclNnTensor> aclInTensors;
    atb::SVector<aclIntArray *> aclInIntArrays;
    atb::SVector<AclNnTensor> aclOutTensors;
    aclOpExecutor *aclExecutor = nullptr;
    void Destory();
};

class AclNnOperation : public atb::Operation {
public:
    explicit AclNnOperation(const std::string &name);
    ~AclNnOperation() override;
    std::string GetName() const override;
    atb::Status Setup(const atb::VariantPack &variantPack, uint64_t &workspaceSize, atb::Context *context) override;
    atb::Status Execute(const atb::VariantPack &variantPack, uint8_t *workspace, uint64_t workspaceSize,
                        atb::Context *context) override;
    bool transB = false;

protected:
    aclTensor* CreateAclTensor(const AclNnTensor &aclNnTensor);
    atb::Status UpdateAclTensorDataPtr(const atb::VariantPack &variantPack);
    std::string opName_;
    AclNnTask aclNnTask_;

private:
    virtual int CreateAclTensors(const atb::VariantPack &variantPack, AclNnTask &task) = 0;
    virtual int CallAclGetWorkspace(AclNnTask &task, uint64_t &workspaceSize) = 0;
    virtual int CallAclExecute(uint8_t *workspace, uint64_t workspaceSize, aclOpExecutor *aclExecutor,
                               aclrtStream stream) = 0;
};
} // namespace common
} // namespace atb_speed
#endif

