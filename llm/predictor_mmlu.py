# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import json
import os
import sys
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from threading import Thread

import numpy as np
import paddle
import paddle.distributed.fleet.base.topology as tp
from paddle.distributed import fleet
# from paddlenlp_ops import reset_stop_value
from paddle_custom_device.npu import (
    reset_stop_value,
    atb_add,
)
from utils import (
    dybatch_preprocess,
    get_alibi_slopes,
    get_infer_model_path,
    get_prefix_tuning_params,
    init_chat_template,
    load_real_time_tokens,
)

from paddlenlp.generation import TextIteratorStreamer
from paddlenlp.peft import LoRAConfig, LoRAModel, PrefixConfig, PrefixModelForCausalLM
from paddlenlp.taskflow.utils import static_mode_guard
from paddlenlp.trainer import PdArgumentParser
from paddlenlp.transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    PretrainedModel,
    PretrainedTokenizer,
)
from paddlenlp.utils.import_utils import import_module, is_paddlenlp_ops_available
from paddlenlp.utils.log import logger
import paddle_custom_device.npu.passes as passes

from tqdm import tqdm
from typing import List
import pandas as pd
import numpy as np

TASK_NAME_MAPPING = {
    "stem": [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    ],
    "Humanities": [
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    ],
    "other": [
        "business_ethics",
        "college_medicine",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
        "global_facts",
        "clinical_knowledge",
    ],
    "social": [
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    ],
}
SUBJECTS = [v for vl in TASK_NAME_MAPPING.values() for v in vl]
choices = ["A", "B", "C", "D"]

def format_example(line, include_answer=True):
    example = "Question: " + line["question"]
    for choice in choices:
        example += f'\n{choice}. {line[f"{choice}"]}'

    if include_answer:
        example += "\nAnswer: " + line["answer"] + "\n\n"
    else:
        example += "\nAnswer:"
    return example

def generate_few_shot_prompt(k, subject, dev_df):
    def format_subject(subject):
        l = subject.split("_")
        s = ""
        for entry in l:
            s += " " + entry
        return s.strip()

    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )

    if k == -1:
        k = dev_df.shape[0]
    for i in range(k):
        prompt += format_example(
            dev_df.iloc[i, :],
            include_answer=True,
        )
    return prompt

def get_logits(predictor, inputs: List[str]):
    outputs = predictor.predict(inputs)
    out_numpy = outputs.copy_to_cpu()
    outputs_tensor = paddle.to_tensor(out_numpy, "float16")
    log_probs=[]
    for i in range(0, outputs_tensor.shape[0]):
        output = outputs_tensor[i,:]
        output_ = output.unsqueeze(0)
        log_probs.append(paddle.nn.functional.softmax(output, axis=-1))
    return log_probs


def eval_subject(
    predictor,
    subject_name,
    test_df,
    k=5,
    dev_df=None,
    few_shot=False,
    save_result_dir=None,
    **kwargs,
):
    result = []
    score = []

    few_shot_prompt = (
        generate_few_shot_prompt(k, subject_name, dev_df) if few_shot else []
    )
    all_probs = {"prob_A": [], "prob_B": [], "prob_C": [], "prob_D": []}


    # print(f"few_shot_prompt: {few_shot_prompt}")
    batch = 30
    count = 1
    full_prompt_multi_batch = []
    outputs = []
    row_input = []
    # question_count = 0
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        row_input.append(row)
        # question_count =question_count + 1
        question = format_example(row, include_answer=False)
        print(count, " ",  batch, " ", count%batch)
        if count%batch != 0:
            full_prompt = few_shot_prompt + question
            full_prompt_multi_batch.append(full_prompt)
            count = count + 1
        else:
            count =  count + 1
            full_prompt = few_shot_prompt + question
            full_prompt_multi_batch.append(full_prompt)
            output = get_logits(predictor, full_prompt_multi_batch)
            full_prompt_multi_batch = []
            for i in range(0, len(output)):
                outputs.append(output[i])
                
    left_question = len(full_prompt_multi_batch) 
    if left_question > 0:
        while len(full_prompt_multi_batch) != batch:
            full_prompt_multi_batch.append("add on")
        output = get_logits(predictor, full_prompt_multi_batch)
        full_prompt_multi_batch = []
        for i in range(0, left_question):
            outputs.append(output[i])       

    for i in range(0, len(outputs)):
        output = outputs[i]
        row = row_input[i]
        logits = output.flatten()
        # print(f"res: {output}")

        # print("-------logits:\n",logits)
        # print("-------logits.shape:\n",logits.shape)
        # print("-------tokenizer A:\n",tokenizer("A")["input_ids"][1:])
        # print("-------tokenizer A:\n",tokenizer("A")["input_ids"])
        # print("-------logits-tokenizer A:\n",logits[tokenizer("A")["input_ids"]])
        # print("-------tokenizer B:\n",tokenizer("B")["input_ids"][1:])
        # print("-------tokenizer B:\n",tokenizer("B")["input_ids"])
        # print("-------tokenizer C:\n",logits[tokenizer(" A")["input_ids"]])
        # print("-------tokenizer D:\n",logits[tokenizer(" A")["input_ids"]])
        # a=[logits[tokenizer("A")["input_ids"][1:]],logits[tokenizer("B")["input_ids"][1:]],logits[tokenizer("C")["input_ids"][1:]],logits[tokenizer("D")["input_ids"][1:]],]
        # logits[tokenizer(" A")["input_ids"]]
        # print('a\n',a)
        # print('row鈥斺€攕tack\n',a)
        # print(torch.row_stack([
        #                 logits[tokenizer('A')['input_ids'][1:]],
        #                 logits[tokenizer('B')['input_ids'][1:]],
        #                 logits[tokenizer('C')['input_ids'][1:]],
        #                 logits[tokenizer('D')['input_ids'][1:]],
        #             ]))
        # torch.tensor(
        #         torch.stack([
        #             logits[tokenizer(' A')['input_ids']],
        #             logits[tokenizer(' B')['input_ids']],
        #             logits[tokenizer(' C')['input_ids']],
        #             logits[tokenizer(' D')['input_ids']],
        #         ])
        #         # logits[tokenizer(" A")["input_ids"]],
        #         # logits[tokenizer(" B")["input_ids"]],
        #         # logits[tokenizer(" C")["input_ids"]],
        #         # logits[tokenizer(" D")["input_ids"]],
        # )
        print("A", logits[predictor.tokenizer('A')['input_ids'][1:]])
        print("B", logits[predictor.tokenizer('B')['input_ids'][1:]])
        print("C", logits[predictor.tokenizer('C')['input_ids'][1:]])
        print("D", logits[predictor.tokenizer('D')['input_ids'][1:]])
        concat_ = paddle.concat((
                logits[predictor.tokenizer('A')['input_ids'][1:]],
                logits[predictor.tokenizer('B')['input_ids'][1:]],
                logits[predictor.tokenizer('C')['input_ids'][1:]],
                logits[predictor.tokenizer('D')['input_ids'][1:]]))
        print("concat_", concat_)
        softval = paddle.nn.functional.softmax(
            concat_,
            axis=0,
        )
        print("softval1", softval)
        softval = paddle.cast(softval, dtype=paddle.float32)
        print("softval2", softval)
        probs = softval.numpy()
        print("probs", probs)
        for i, choice in enumerate(choices):
            all_probs[f"prob_{choice}"].append(probs[i])
        # print("probs\n",probs)
        # print("np.argmax(probs)\n",np.argmax(probs))
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
        # print("pred\n",pred)
        if "answer" in row:
            correct = 1 if pred == row["answer"] else 0
            score.append(correct)
            # if args.debug:
            print(f'{question} pred: {pred} ref: {row["answer"]}')
        result.append(pred)

    if save_result_dir:
        test_df["model_output"] = result
        for i, choice in enumerate(choices):
            test_df[f"prob_{choice}"] = all_probs[f"prob_{choice}"]
        if score:
            test_df["correctness"] = score
        os.makedirs(save_result_dir, exist_ok=True)
        test_df.to_csv(
            os.path.join(save_result_dir, f"{subject_name}_result.csv"),
            encoding="utf-8",
            index=False,
        )

    return score

def eval_mmlu(predictor):
    dev_result = {}
    for subject_name in tqdm(SUBJECTS):
        # val_file_path = os.path.join(args.eval_data_path, 'val', f'{subject_name}_val.csv')
        dev_file_path = os.path.join(
            "/home/yuanwei/eval_project/data/mmlu/data/", "dev", f"{subject_name}_dev.csv"
        )
        test_file_path = os.path.join(
            "/home/yuanwei/eval_project/data/mmlu/data/", "test", f"{subject_name}_test.csv"
        )
        # val_df = pd.read_csv(val_file_path, names=['question','A','B','C','D','answer'])
        dev_df = pd.read_csv(
            dev_file_path, names=["question", "A", "B", "C", "D", "answer"]
        )
        test_df = pd.read_csv(
            test_file_path, names=["question", "A", "B", "C", "D", "answer"]
        )

        score = eval_subject(
            predictor,
            subject_name,
            test_df,
            dev_df=dev_df,
            k=5,
            few_shot=True,
            save_result_dir=f"outs/mmlu_eval_result",
        )
        dev_result[subject_name] = score

@dataclass
class PredictorArgument:
    model_name_or_path: str = field(default=None, metadata={"help": "The directory of model."})
    model_prefix: str = field(default="model", metadata={"help": "the prefix name of static model"})
    src_length: int = field(default=1024, metadata={"help": "The max length of source text."})
    max_length: int = field(default=2048, metadata={"help": "the max length for decoding."})
    top_k: int = field(default=0, metadata={"help": "top_k parameter for generation"})
    top_p: float = field(default=0.7, metadata={"help": "top_p parameter for generation"})
    temperature: float = field(default=0.95, metadata={"help": "top_p parameter for generation"})
    repetition_penalty: float = field(default=1.0, metadata={"help": "repetition penalty parameter for generation"})
    device: str = field(default="npu", metadata={"help": "Device"})
    dtype: str = field(default=None, metadata={"help": "Model dtype"})
    lora_path: str = field(default=None, metadata={"help": "The directory of LoRA parameters. Default to None"})
    export_precache: bool = field(default=False, metadata={"help": "whether use prefix weight to do infer"})
    prefix_path: str = field(
        default=None, metadata={"help": "The directory of Prefix Tuning parameters. Default to None"}
    )
    decode_strategy: str = field(
        default="sampling",
        metadata={
            "help": "the decoding strategy of generation, which should be one of ['sampling', 'greedy_search', 'beam_search']. Default to sampling"
        },
    )

    mode: str = field(
        default="dynamic", metadata={"help": "the type of predictor, it should be one of [dynamic, static]"}
    )
    inference_model: bool = field(default=False, metadata={"help": "whether use InferenceModel to do generation"})
    quant_type: str = field(
        default=None,
        metadata={"help": "Quantization type. Supported values: a8w8, weight_only_int4, weight_only_int8"},
    )

    batch_size: int = field(default=1, metadata={"help": "The batch size of data."})
    benchmark: bool = field(
        default=False,
        metadata={
            "help": "If benchmark set as `True`, we will force model decode to max_length, which is helpful to compute throughput. "
        },
    )

    enable_memory_optim: bool = field(
        default=True,
        metadata={"help": "whether use `enable_memory_optim` in inference predictor"},
    )
    init_fleet_worker: bool = field(
        default=True,
        metadata={"help": "whether use `init_fleet_worker` in inference predictor"},
    )
    block_attn: bool = field(default=False, metadata={"help": "whether use block attention"})
    block_size: int = field(default=64, metadata={"help": "the block size for cache_kvs."})
    use_cachekv_int8: str = field(
        default="None",
        metadata={"help": "If use_cachekv_int8 set as `dynamic`, dynamic cache kv quantization will be applied; if set as `static`, static cache kv will be applied"},)
    
    chat_template: str = field(
        default=None,
        metadata={
            "help": "the path of `chat_template.json` file to handle multi-rounds conversation. If is None, it will not use `chat_template.json`; If is equal with `model_name_or_path`, it will use the default loading; If is directory, it will find the `chat_template.json` under the directory; If is file, it will load it."
        },
    )

    @property
    def total_max_length(self):
        return self.src_length + self.max_length


@dataclass
class ModelArgument:
    model_type: str = field(
        default=None,
        metadata={"help": "the type of the model, which can be one of ['gpt-3', 'ernie-3.5-se', 'llama-img2txt']"},
    )
    data_file: str = field(default=None, metadata={"help": "data file directory"})
    output_file: str = field(default="output.json", metadata={"help": "predict result file directory"})


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts


def init_dist_env():
    tensor_parallel_degree = paddle.distributed.get_world_size()
    tensor_parallel_rank = paddle.distributed.get_rank()

    if tensor_parallel_degree > 1:
        # refer to: https://github.com/PaddlePaddle/Paddle/blob/4abea956ee852ce52791a1e08fa92ed4d3be150d/python/paddle/distributed/fleet/fleet.py#L298C23-L298C45
        hcg = tp._HYBRID_PARALLEL_GROUP
        if hcg is None:
            strategy = fleet.DistributedStrategy()
            strategy.hybrid_configs = {
                "dp_degree": 1,
                "mp_degree": tensor_parallel_degree,
                "pp_degree": 1,
                "sharding_degree": 1,
            }
            fleet.init(is_collective=True, strategy=strategy)
            hcg = fleet.get_hybrid_communicate_group()

        tensor_parallel_rank = hcg.get_model_parallel_rank()
    return tensor_parallel_rank, tensor_parallel_degree


class BasePredictor:
    def __init__(self, config: PredictorArgument, tokenizer: PretrainedTokenizer = None):
        self.model_config = AutoConfig.from_pretrained(config.model_name_or_path)
        self.config: PredictorArgument = config
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, padding_side="left")

        self.tokenizer = tokenizer

        self.return_tensors = "pd"
        self.tensor_parallel_rank, self.tensor_parallel_degree = init_dist_env()
        self.model_config.tensor_parallel_rank, self.model_config.tensor_parallel_degree = init_dist_env()

    def _preprocess(self, source):
        if self.config.chat_template is not None:
            if self.tokenizer.chat_template is None:
                logger.warning(
                    f"Tokenizer<{self.tokenizer}> doesn't have chat_template field, so it will not use chat_template."
                    "Or you can customize your tokenizer, please refer to:"
                    "https://paddlenlp.readthedocs.io/zh/latest/get_started/chat_template.html"
                )
            else:
                source = [source] if isinstance(source, str) else source
                source = [self.tokenizer.apply_chat_template(sentence, tokenize=False) for sentence in source]

        tokenized_source = self.tokenizer(
            source,
            max_length=self.config.src_length,
            truncation=True,
            truncation_side="left",
            return_tensors=self.return_tensors,
            padding=True,
            # when use chat_template, it should not add special tokens
            add_special_tokens=self.config.chat_template is None,
        )
        return tokenized_source

    @abstractmethod
    def _infer(self, inputs):
        raise NotImplementedError

    def _postprocess(self, predictions):
        decoded_predictions = self.tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return decoded_predictions

    def predict(self, input_texts: str | list[str]):
        tokenized_source = self._preprocess(input_texts)
        print(f'tokenized_source={tokenized_source}')
        predictions = self._infer(tokenized_source)
        decoded_predictions = self._postprocess(predictions)
        return decoded_predictions

class StaticBlockInferencePredictor(BasePredictor):
    def __init__(
        self,
        config: PredictorArgument,
        cache_kvs_shape: list[list[int]],
        tokenizer: PretrainedTokenizer = None,
    ):
        self.cache_kvs_shape = cache_kvs_shape
        BasePredictor.__init__(self, config, tokenizer)

        self.inputs = {}
        self.num_attention_heads = self.cache_kvs_shape[0][-3]
        self.head_dim = self.cache_kvs_shape[0][-1]
        self.max_block_nums = self.cache_kvs_shape[0][0]
        self.block_size = config.block_size
        self.total_max_length = config.src_length + config.max_length
        self.num_layers = len(self.cache_kvs_shape) // 2
        pre_max_block_num = (self.total_max_length + config.block_size - 1) // config.block_size
        # not update
        self.pre_cache_length = 0
        if config.export_precache:
            pre_cache_npy = np.load(config.prefix_path)
            self.pre_cache_length = pre_cache_npy.shape[-2]
            config.max_length -= self.pre_cache_length
            for i in range(self.num_layers):
                self.inputs["pre_caches_{}".format(2 * i)] = paddle.to_tensor(pre_cache_npy[i][0], dtype=config.dtype).unsqueeze(0).broadcast_to([config.batch_size, self.num_attention_heads, self.pre_cache_length, self.head_dim])
                self.inputs["pre_caches_{}".format(2 * i + 1)] = paddle.to_tensor(pre_cache_npy[i][1], dtype=config.dtype).unsqueeze(0).broadcast_to([config.batch_size, self.num_attention_heads, self.pre_cache_length, self.head_dim])
            pre_cache_mask = paddle.zeros(shape=[config.batch_size, 1, config.src_length, config.src_length + self.pre_cache_length], dtype=config.dtype)
            pre_cache_mask[:, :, :, :self.pre_cache_length] = 1
            pre_cache_mask[:, :, :, self.pre_cache_length:] = paddle.tril(paddle.ones(shape=[config.batch_size, 1, config.src_length, config.src_length], dtype=config.dtype))
            self.inputs["src_mask"] = (pre_cache_mask - 1) * 1e4
        # else:
        #     self.attention_mask = np.zeros(shape=(config.batch_size, 1, config.total_max_length, config.total_max_length), dtype=config.dtype)
        #     self.inputs["src_mask"] = paddle.zeros(shape=(config.batch_size, 1, config.total_max_length, config.total_max_length), dtype=config.dtype)

        self.cache_kvs = {}
        if config.use_cachekv_int8 == "dynamic" or config.use_cachekv_int8 == "static":
            for i in range(len(self.cache_kvs_shape) // 2):
                self.cache_kvs["key_caches_{}".format(i)] = paddle.zeros(self.cache_kvs_shape[2 * i], dtype="uint8")
                self.cache_kvs["value_caches_{}".format(i)] = paddle.zeros(
                    self.cache_kvs_shape[2 * i + 1], dtype="uint8"
                )
        else:
            for i in range(len(self.cache_kvs_shape) // 2):
                self.cache_kvs["key_caches_{}".format(i)] = paddle.zeros(
                    self.cache_kvs_shape[2 * i], dtype=config.dtype
                )
                self.cache_kvs["value_caches_{}".format(i)] = paddle.zeros(
                    self.cache_kvs_shape[2 * i + 1], dtype=config.dtype
                )

        if config.use_cachekv_int8 == "dynamic":
            self.k_quant_scales = [
                paddle.zeros([self.num_attention_heads], dtype="float32") for _ in range(self.num_layers)
            ]
            self.v_quant_scales = [
                paddle.zeros([self.num_attention_heads], dtype="float32") for _ in range(self.num_layers)
            ]
            self.k_dequant_scales = [
                paddle.zeros([self.num_attention_heads], dtype="float32") for _ in range(self.num_layers)
            ]
            self.v_dequant_scales = [
                paddle.zeros([self.num_attention_heads], dtype="float32") for _ in range(self.num_layers)
            ]

        if config.benchmark:
            min_length = config.max_length
        else:
            min_length = 2
        self.inputs["min_length"] = paddle.full(shape=[config.batch_size, 1], fill_value=min_length, dtype="int64")
        self.inputs["max_length"] = paddle.full(
            shape=[config.batch_size, 1], fill_value=config.max_length, dtype="int64"
        )

        self.inputs["pre_ids"] = paddle.full([config.batch_size, self.total_max_length], -1, dtype="int64")
        self.inputs["bad_tokens"] = paddle.to_tensor([-1, ], dtype="int64")
        self.inputs["penalty_score"] = paddle.full(shape=[config.batch_size, 1], fill_value=1.0, dtype="float32")
        self.inputs["frequency_score"] = paddle.full(shape=[config.batch_size, 1], fill_value=0.0, dtype="float32")
        self.inputs["presence_score"] = paddle.full(shape=[config.batch_size, 1], fill_value=0.0, dtype="float32")

        self.inputs["stop_nums"] = paddle.full(shape=[1], fill_value=config.batch_size, dtype="int64")
        tmp_position_ids = paddle.arange(4096).reshape((1, -1))
        self.inputs["cos_tables"], self.inputs["sin_tables"], self.inputs["rope_emb"] = self._get_rotary_position_embedding(tmp_position_ids, self.head_dim)
        self.inputs["eos_token_id"] = paddle.to_tensor(
            [
                self.tokenizer.eos_token_id,
            ],
            "int64",
        )
        # need update
        self.inputs["block_tables"] = paddle.full(
            shape=[config.batch_size, pre_max_block_num], fill_value=-1, dtype="int32"
        )
        self.inputs["input_ids"] = paddle.full(
            shape=[config.batch_size, config.src_length], fill_value=-1, dtype="int64"
        )
        print("#########################src len:", config.src_length, " config bs: ", config.batch_size)
        print(f'top_p={config.top_p}')
        self.inputs["top_p"] = paddle.full(shape=[config.batch_size, 1], fill_value=config.top_p, dtype="float32")
        self.inputs["temperature"] = paddle.full(shape=[config.batch_size, 1], fill_value=1.0, dtype="float32")
        self.inputs["seq_lens_this_time"] = paddle.full(shape=[config.batch_size, 1], fill_value=0, dtype="int32")
        self.inputs["seq_lens_encoder"] = paddle.full(shape=[config.batch_size, 1], fill_value=0, dtype="int32")
        self.inputs["seq_lens_decoder"] = paddle.full(shape=[config.batch_size, 1], fill_value=0, dtype="int32")
        self.inputs["step_idx"] = paddle.full(shape=[config.batch_size, 1], fill_value=0, dtype="int64")
        self.inputs["not_need_stop"] = paddle.full(shape=[1], fill_value=False, dtype="bool").cpu()
        self.inputs["stop_flags"] = paddle.full(shape=[config.batch_size, 1], fill_value=True, dtype="bool")

        self.inputs['step_seq_lens_encoder'] = paddle.full(shape=[config.batch_size, 1], fill_value=0, dtype="int32")
        self.inputs['next_tokens'] = paddle.full(shape=[config.batch_size, 1], fill_value=-1, dtype="int64")
        self.inputs['is_block_step'] = paddle.full(shape=[config.batch_size], fill_value=False, dtype="bool")
        free_list = list(range(pre_max_block_num - 1, int(pre_max_block_num * 0.75) -1, -1))
        self.inputs['encoder_block_lens'] = paddle.full(shape=[config.batch_size], fill_value=0, dtype="int32")
        self.inputs['step_block_list'] = paddle.full(shape=[config.batch_size], fill_value=-1, dtype="int32")
        self.inputs['step_lens'] = paddle.full(shape=[1], fill_value=0, dtype="int32")
        self.inputs['recover_block_list'] = paddle.full(shape=[config.batch_size], fill_value=-1, dtype="int32")
        self.inputs['recover_lens'] = paddle.full(shape=[1], fill_value=0, dtype="int32")
        self.inputs['need_block_list'] = paddle.full(shape=[config.batch_size], fill_value=-1, dtype="int32")
        self.inputs['need_block_len'] = paddle.full(shape=[1], fill_value=0, dtype="int32")
        self.inputs['used_list_len'] = paddle.full(shape=[config.batch_size], fill_value=0, dtype="int32")
        self.inputs['free_list'] = paddle.to_tensor(free_list, dtype="int32")
        self.inputs['free_list_len'] = paddle.full(shape=[1], fill_value=pre_max_block_num * 0.25, dtype="int32")
        self.inputs['is_decoder'] = paddle.full(shape=[1], fill_value=False, dtype="bool")


        for i in range(self.num_layers):
            if self.config.use_cachekv_int8 == "dynamic":
                self.inputs["k_quant_scales_" + str(i)] = self.k_quant_scales[i]
                self.inputs["v_quant_scales_" + str(i)] = self.v_quant_scales[i]
                self.inputs["k_dequant_scales_" + str(i)] = self.k_dequant_scales[i]
                self.inputs["v_dequant_scales_" + str(i)] = self.v_dequant_scales[i]

        self.free_list = [i for i in range(self.max_block_nums)][::-1]
        self.used_list = [[] for _ in range(config.batch_size)]

        self._create_predictor(config)
        self.input_names = self.predictor.get_input_names()

        self.seq_lens_handle = self.predictor.get_input_handle("seq_lens_this_time")

    def _get_rotary_position_embedding(self, position_ids, head_dim):
        """
        Pre-calculate rotary position embedding for position_ids.

        Args:
            position_ids: [1, S]
            head_dim: D

        Returns:
            rot_emb: [2, 1, S, 1, D], cos + sin
        """
        bsz, max_seq_len = position_ids.shape[:2]
        rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, head_dim), dtype="float32")
        inv_freq = 1.0 / (10000 ** (paddle.arange(0, head_dim, 2, dtype="float32") / head_dim))
        t = paddle.arange(4096, dtype="float32")
        # shape: [B, S, D/2]
        freqs = paddle.einsum("i,j->ij", t, inv_freq)
        # shape: [B, S, 1, D]
        emb = paddle.concat([freqs, freqs], axis=-1).reshape((bsz, max_seq_len, 1, head_dim)).cast("float16")

        cos_table = paddle.cos(emb)
        sin_table = paddle.sin(emb)
        rot_emb[0] = cos_table
        rot_emb[1] = cos_table
        cos_table = paddle.squeeze(cos_table)
        sin_table = paddle.squeeze(sin_table)
        return cos_table, sin_table, rot_emb

    def _create_predictor(self, predictor_args: PredictorArgument):
        # if not is_paddlenlp_ops_available():
        #     raise ValueError(
        #         "you should install the paddlenlp ops to run inference predictor, "
        #         "https://github.com/PaddlePaddle/PaddleNLP/blob/develop/csrc/README.md"
        #     )

        infer_model_path = get_infer_model_path(predictor_args.model_name_or_path, predictor_args.model_prefix)

        config = paddle.inference.Config(infer_model_path + ".pdmodel", infer_model_path + ".pdiparams")

        config.switch_ir_optim(True)
        device_id = int(os.environ.get("FLAGS_selected_npus", 0))
        config.enable_custom_device("npu", device_id)
        # config.disable_glog_info()
        config.enable_memory_optim()

        config.set_optim_cache_dir("./optim_cache")
        pass_builder = config.pass_builder()
        # passes.addPasses(pass_builder, "llama65B_mp8") 
        passes.addPasses(pass_builder, "llama65B_mp8")
        # pass_builder.turn_on_debug()
        
        if self.tensor_parallel_degree >= 1:
            trainer_endpoints = fleet.worker_endpoints()
            current_endpoint = trainer_endpoints[self.tensor_parallel_rank]

            dist_config = config.dist_config()
            dist_config.set_ranks(self.tensor_parallel_degree, self.tensor_parallel_rank)
            dist_config.set_endpoints(trainer_endpoints, current_endpoint)
            dist_config.enable_dist_model(True)

            dist_config.set_comm_init_config(os.path.join(predictor_args.model_name_or_path, "rank_mapping.csv"))
            config.set_dist_config(dist_config)

        self.predictor = paddle.inference.create_predictor(config)

    def _share_data(self):
        """
        分享不拷贝数据
        """
        for name in self.input_names:
            if "pre_key_" in name or "pre_value_" in name:
                input_tensor = self.predictor.get_input_handle(name)
                input_tensor.share_external_data(self.inputs[name])
                continue
            if "caches" in name:
                input_tensor = self.predictor.get_input_handle(name)
                input_tensor.share_external_data(self.cache_kvs[name])
                continue
            if "seq_lens_this_time" in name:
                continue
            input_tensor = self.predictor.get_input_handle(name)
            input_tensor.share_external_data(self.inputs[name])

    def _infer(self):
        self.predictor.run()

    def predict(self, input_texts: str | list[str]):
        self._preprocess(input_texts)
        real_bsz = len(input_texts)
        print("################################ real_bsz: ", real_bsz)

        import copy
        seq_lens_this_time = copy.deepcopy(self.inputs["seq_lens_this_time"][:real_bsz])
        self.seq_lens_handle.share_external_data(seq_lens_this_time)

        self.predictor.run()
            # atb_add(self.inputs["tgt_pos"], paddle.to_tensor([1], dtype="int64"))
            # self.inputs["tgt_pos"] = self.inputs["tgt_pos"] + 3

        # reset free_list
        for i in range(self.config.batch_size):
            self.free_list.extend(self.used_list[i])
            self.used_list[i] = []
        reset_stop_value(self.inputs["not_need_stop"])
        false_decoder = paddle.full(shape=[1, 1], dtype="bool", fill_value=False)
        paddle.assign(false_decoder, self.inputs["is_decoder"])
        return self.predictor.get_output_handle("save_infer_model/scale_0.tmp_0")

    def _preprocess(self, source):
        seq_len = []
        max_len = 0
        for i, text in enumerate(source):
            # print("text: ", text)
            tokens = self.tokenizer(text, return_tensors="np", padding=False)
            input_ids = tokens["input_ids"][0]
            print("input_ids:", input_ids.shape)
            length = len(input_ids)
            if length > 2048:
                input_ids = input_ids[2048 :]
            length = len(input_ids)
            print("input_ids new:", length)
            seq_len.append(length)
            if (max_len < length):
                max_len = length
            # print("input_ids: ", input_ids)
            print("length: ", length)
            self.inputs["input_ids"][i : i + 1, :length] = input_ids
            self.inputs["penalty_score"][i : i + 1] = self.config.repetition_penalty
            self.inputs["frequency_score"][i : i + 1] = 0.0
            self.inputs["presence_score"][i : i + 1] = 0.0
            self.inputs['top_p'][i:i+1] = self.config.top_p
            self.inputs['temperature'][i:i+1] = self.config.temperature
            self.inputs["seq_lens_this_time"][i : i + 1] = length
            self.inputs['step_seq_lens_encoder'][i:i+1] = length
            self.inputs["seq_lens_encoder"][i : i + 1] = length
            self.inputs["seq_lens_decoder"][i : i + 1] = 0
            self.inputs["step_idx"][i : i + 1] = 0
            # self.inputs["stop_flags"][i : i + 1] = False
            reset_stop_value(self.inputs["not_need_stop"])
            need_block_nums = (length + self.config.max_length + self.pre_cache_length + self.block_size - 1) // self.block_size
            # print("self.free_list",  self.free_list)
            for bi in range(need_block_nums):
                bi_now = self.free_list.pop()
                self.used_list[i].append(bi_now)
                self.inputs["block_tables"][i : i + 1, bi] = bi_now

            # encoder_block_num = len(task['block_tables'])
            self.inputs['encoder_block_lens'][i:i+1] = need_block_nums
            # self.attention_mask[i, 0, :length, :length] = np.tril(np.ones(shape=(length, length), dtype=self.config.dtype))
        self.attention_mask = np.zeros(shape=(self.config.batch_size, max_len, max_len), dtype=self.config.dtype)
        for i in range(len(seq_len)):
            length = seq_len[i]
            self.attention_mask[i, :length, :length] = np.tril(np.ones(shape=(length, length), dtype=self.config.dtype))
        # self.inputs["src_mask"] = paddle.zeros(shape=(config.batch_size, 1, config.total_max_length, config.total_max_length), dtype=config.dtype)
        # self.inputs["src_mask"].get_tensor().set(self.attention_mask, paddle.base.framework._current_expected_place())
        self.inputs["src_mask"] = paddle.to_tensor(self.attention_mask)
        self.inputs["src_mask"] = (self.inputs["src_mask"] - 1) * 1e4
        position_ids = np.arange(sum(seq_len), dtype="int64")
        pre_len = seq_len[0]
        for length in seq_len[1:]:
            position_ids[pre_len : length + pre_len] = position_ids[pre_len : length + pre_len] - pre_len
            pre_len += length
        self.inputs["position_ids"] = paddle.to_tensor(position_ids)

        tgt_pos = []
        for i, valid_len in enumerate(seq_len):
            tgt_pos.append(valid_len - 1)
        self.inputs["tgt_pos"] = paddle.to_tensor(np.array(tgt_pos).astype("int64").reshape(-1, 1))

        self.inputs["stop_flags"] = paddle.full(shape=[self.config.batch_size, 1], fill_value=False, dtype="bool") # 规避setvalue不支持bool类型
        self._share_data() # TODO：如何init阶段完成

def get_ptq_multicards_num(directory):
    count = 0  
    prefix = "act_scales_"
    for filename in os.listdir(directory):  
        if filename.startswith(prefix):  
            count += 1 
    return count

def create_predictor(
    predictor_args: PredictorArgument,
    model_args: ModelArgument,
    tensor_parallel_degree: int = 1,
    tensor_parallel_rank: int = 0,
):
    tokenizer = AutoTokenizer.from_pretrained(predictor_args.model_name_or_path)
    init_chat_template(tokenizer, predictor_args.model_name_or_path, predictor_args.chat_template)
    # TODO(wj-Mcat): fix llama tokenzier pad_token bug
    if isinstance(tokenizer, LlamaTokenizer) and not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.unk_token

    # update config parameter for inference predictor
    if predictor_args.decode_strategy == "greedy_search":
        predictor_args.top_p = 0.0
        predictor_args.temperature = 1.0

    tensor_parallel_rank, tensor_parallel_degree = init_dist_env()
    if not predictor_args.inference_model:
        if predictor_args.mode == "dynamic":
            if model_args.model_type == "gpt-3":
                sys.path.append("./gpt-3")
                from modeling import GPTForCausalLM

                model = GPTForCausalLM.from_pretrained(
                    predictor_args.model_name_or_path,
                    dtype=predictor_args.dtype,
                    tensor_parallel_degree=tensor_parallel_degree,
                    tensor_parallel_rank=tensor_parallel_rank,
                )
            elif model_args.model_type == "ernie-3.5-se":
                sys.path.append("./ernie-3.5-se")
                from modeling import Ernie35ForCausalLM

                tensor_parallel_degree = paddle.distributed.get_world_size()
                tensor_parallel_rank = paddle.distributed.get_rank()
                model = Ernie35ForCausalLM.from_pretrained(
                    predictor_args.model_name_or_path,
                    dtype=predictor_args.dtype,
                    tensor_parallel_degree=tensor_parallel_degree,
                    tensor_parallel_rank=tensor_parallel_rank,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    predictor_args.model_name_or_path,
                    dtype=predictor_args.dtype,
                    tensor_parallel_degree=tensor_parallel_degree,
                    tensor_parallel_rank=tensor_parallel_rank,
                )

            predictor = DygraphPredictor(predictor_args, model=model, tokenizer=tokenizer)
        elif predictor_args.mode == "static":
            predictor = StaticGraphPredictor(predictor_args, tokenizer=tokenizer)
        else:
            raise ValueError("the `mode` should be one of [dynamic, static]")
    else:
        if predictor_args.mode == "dynamic":
            # TODO(wj-Mcat): complete AutoInferenceModel & AutoPredictor
            config = AutoConfig.from_pretrained(predictor_args.model_name_or_path)
            config.tensor_parallel_degree = tensor_parallel_degree
            config.tensor_parallel_rank = tensor_parallel_rank
            config.weight_only_quant_bits = -1
            config.quant_type = None
            config.model_name_or_path = ""
            config.use_cachekv_int8 = predictor_args.use_cachekv_int8
            config.single_card_ptq = True
            
            if predictor_args.quant_type is not None and predictor_args.quant_type.startswith("weight_only_int"):
                weight_only_quant_bits = int(predictor_args.quant_type[-1])
                config.weight_only_quant_bits = weight_only_quant_bits
                config.quant_type = predictor_args.quant_type

            if config.quantization_config.quant_type is not None and "a8w8" in config.quantization_config.quant_type:
                config.model_name_or_path = predictor_args.model_name_or_path
                config.quant_type = config.quantization_config.quant_type
                    
                ptq_multicards_num = get_ptq_multicards_num(config.model_name_or_path)
                logger.info(f"PTQ from {ptq_multicards_num} cards, so we will not split")
                if ptq_multicards_num > 1:
                    config.single_card_ptq = False

                # Turn on GEMM int8 kernel tuning
                paddle.base.core.enable_autotune()
                paddle.base.core.update_autotune_status()

            if "llama" in config.architectures[0].lower():
                if model_args.model_type == "llama-img2txt":
                    # we use llama for img2txt.
                    from paddlenlp.experimental.transformers import (
                        LlamaForMiniGPT4InferenceModel as LlamaInferenceModel,
                    )
                elif predictor_args.block_attn:
                    config.max_seq_len = predictor_args.src_length
                    config.block_size = predictor_args.block_size
                    from paddlenlp.experimental.transformers import (
                        LlamaForCausalLMBlockInferenceModel as LlamaInferenceModel,
                    )
                else:
                    from paddlenlp.experimental.transformers import (
                        LlamaForCausalLMInferenceModel as LlamaInferenceModel,
                    )
                model = LlamaInferenceModel.from_pretrained(
                    predictor_args.model_name_or_path, config=config, dtype=predictor_args.dtype, 
                    tensor_parallel_degree=tensor_parallel_degree,
                    tensor_parallel_rank=tensor_parallel_rank,
                )
                model.eval()

            elif "opt" in config.architectures[0].lower():
                if model_args.model_type == "opt-img2txt":
                    # we use opt for img2txt.
                    from paddlenlp.experimental.transformers import (
                        OPTForBlip2InferenceModel as OPTInferenceModel,
                    )
                else:
                    from paddlenlp.experimental.transformers import (
                        OPTForCausalLMInferenceModel as OPTInferenceModel,
                    )

                model = OPTInferenceModel.from_pretrained(
                    predictor_args.model_name_or_path, config=config, dtype=predictor_args.dtype
                )
                model.eval()

            elif "chatglmv2forcausallm" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    ChatGLMv2ForCausalLMInferenceModel as Model,
                )

                model = Model.from_pretrained(
                    predictor_args.model_name_or_path, config=config, dtype=predictor_args.dtype
                )
                model.eval()
            elif "chatglmforcausallm" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    ChatGLMForCausalLMInferenceModel,
                )

                model = ChatGLMForCausalLMInferenceModel.from_pretrained(
                    predictor_args.model_name_or_path,
                    config=config,
                    dtype=predictor_args.dtype,
                )
                model.eval()
            elif "bloom" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    BloomForCausalLMInferenceModel,
                )

                model = BloomForCausalLMInferenceModel.from_pretrained(
                    predictor_args.model_name_or_path,
                    config=config,
                    dtype=predictor_args.dtype,
                )
                cache_kvs_shape = BloomForCausalLMInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
                model.eval()
            elif "gpt" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    GPTForCausalLMInferenceModel,
                )

                model = GPTForCausalLMInferenceModel.from_pretrained(
                    predictor_args.model_name_or_path,
                    config=config,
                    dtype=predictor_args.dtype,
                )
                model.eval()
            else:
                raise ValueError("the `model type` should be one of [llama, chatglm, bloom, gpt]")
            if predictor_args.block_attn:
                predictor = DygraphBlockInferencePredictor(predictor_args, model=model, tokenizer=tokenizer)
            else:
                predictor = DygraphInferencePredictor(predictor_args, model=model, tokenizer=tokenizer)


        elif predictor_args.mode == "static":
            config = AutoConfig.from_pretrained(predictor_args.model_name_or_path)
            if "llama" in config.architectures[0].lower():
                if predictor_args.block_attn:
                    config.block_size = predictor_args.block_size
                    config.max_seq_len = predictor_args.src_length
                    config.use_dynamic_cachekv_quant = predictor_args.use_cachekv_int8 == "dynamic"
                    from paddlenlp.experimental.transformers import (
                        LlamaForCausalLMBlockInferenceModel as LlamaInferenceModel,
                    )
                else:
                    from paddlenlp.experimental.transformers import (
                        LlamaForCausalLMInferenceModel as LlamaInferenceModel,
                    )

                cache_kvs_shape = LlamaInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
            elif "chatglmv2forcausallm" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    ChatGLMv2ForCausalLMInferenceModel,
                )

                cache_kvs_shape = ChatGLMv2ForCausalLMInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
            elif "chatglmforcausallm" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    ChatGLMForCausalLMInferenceModel,
                )

                cache_kvs_shape = ChatGLMForCausalLMInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
            elif "bloom" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    BloomForCausalLMInferenceModel,
                )

                cache_kvs_shape = BloomForCausalLMInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
            elif "gpt" in config.architectures[0].lower():
                from paddlenlp.experimental.transformers import (
                    GPTForCausalLMInferenceModel,
                )

                cache_kvs_shape = GPTForCausalLMInferenceModel.get_cache_kvs_shape(
                    config, predictor_args.batch_size, predictor_args.total_max_length
                )
            else:
                raise ValueError("the `model type` should be one of [llama, chatglm, bloom, gpt]")
            if predictor_args.block_attn:
                predictor = StaticBlockInferencePredictor(predictor_args, cache_kvs_shape, tokenizer=tokenizer)
            else:
                predictor = StaticInferencePredictor(predictor_args, cache_kvs_shape, tokenizer=tokenizer)
        else:
            raise ValueError("the `mode` should be one of [dynamic, static]")
    return predictor


def predict():
    parser = PdArgumentParser((PredictorArgument, ModelArgument))
    predictor_args, model_args = parser.parse_args_into_dataclasses()

    paddle.set_device(predictor_args.device)
    paddle.set_default_dtype(predictor_args.dtype)

    tensor_parallel_degree = paddle.distributed.get_world_size()
    # Note(zhengzekang): force to use fleet executor.
    if predictor_args.init_fleet_worker or tensor_parallel_degree > 1:
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": tensor_parallel_degree,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

    predictor = create_predictor(predictor_args, model_args)
    source_texts = []
    target_texts = []
    if model_args.data_file:
        with open(model_args.data_file, "r", encoding="utf-8") as f:
            for line in f:
                example = json.loads(line)
                source_texts.append(example["src"])
                target_texts.append(example["tgt"])
    else:
        # source_texts = ["解释一下“温故而知新”", "你好，请问你是谁?"]
        source_texts = []

        data_file = open("humaneval_solution.json", 'r')
        
        dataset = []
        for line in data_file.readlines():
            dataset.append(json.loads(line))

        # for i in range(predictor_args.batch_size):
        #     data = dataset[i % 164]
            # source_texts.append(data["prompt"])
        source_texts.append("Summarize the main ideas of Jeff Walker's Product Launch Formula into bullet points as it pertains to a growth marketing agency implementing these strategies and tactics for their clients...")

    eval_mmlu(predictor)




def benchmark(predictor, predictor_args, model_args):
    # Just construct a simple benchmark input. We pad input to the src_length.
    test_texts = ""
    benchmark_texts = [test_texts + "<pad>" * 3072 for _ in range(predictor_args.batch_size)]

    batch_benchmark_texts = batchfy_text(benchmark_texts, predictor_args.batch_size)
    print("***********Start Benchmark**********")

    warmup_time = 1
    test_time = 1

    print("***********Start Warmup**********")
    for i in range(warmup_time):
        print("warm up ", i)
        for bs, batch_source_text in enumerate(batch_benchmark_texts):
            outputs = predictor.predict(batch_source_text)

    print("***********Start Speed Test**********")
    start = time.perf_counter()
    output_tokens = 0
    for i in range(test_time):
        print("test ", i)
        for bs, batch_source_text in enumerate(batch_benchmark_texts):
            outputs = predictor.predict(batch_source_text)
            output_tokens += predictor_args.max_length * predictor_args.batch_size
    end = time.perf_counter()
    print("Avg Elapse time is: ", (end - start) / test_time)
    print("Output tokens is: ", output_tokens)
    print(
        "Input length is: {}, Output length is: {}, bs is: {}, IPS: {:.3f} tokens/s, QPS: {:.3f} requests/s. ".format(
            predictor_args.src_length,
            predictor_args.max_length,
            predictor_args.batch_size,
            (output_tokens / (end - start)),
            (predictor_args.batch_size * test_time / (end - start)),
        )
    )


if __name__ == "__main__":
    predict()


