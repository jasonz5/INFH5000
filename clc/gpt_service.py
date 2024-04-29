#!/usr/bin/env python
# -*- coding:utf-8 _*-

import os

from typing import List
from peft import PeftModel
from langchain.llms.base import LLM
from typing import Dict, Union, Optional
from langchain.llms.utils import enforce_stop_tokens
from accelerate import load_checkpoint_and_dispatch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


class ChatQwenService(LLM): # 继承于langchain.llms.base
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    messages = []
    tokenizer: object = None
    model: object = None

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "Qwen2ForCausalLM"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        cur_usr = {"role":"user","content":prompt}
        self.messages += [cur_usr]
        text = self.tokenizer.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [ output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # response, _ = self.model.chat(
        #     self.tokenizer,
        #     prompt,
        #     history=self.history,
        #     max_length=self.max_token,
        #     temperature=self.temperature,
        # )

        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        cur_sys = {"role":"system","content":response}
        
        self.messages += [cur_sys]
        print("self.messages:",self.messages)
        return response

    def set_history(self, history):
        self.messages = history

    # 加载预训练的模型和对应的分词器,这里要改成自己的模型
    def load_model(self,
                   model_name_or_path: str,
                   use_lora: str = "Qwen1.5-0.5B-ChatMed",
                   ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).cuda()
        if use_lora == "Qwen1.5-0.5B-ChatMed":
            self.model = PeftModel.from_pretrained(self.model, model_id="/home/wangrui/LLM/infh5000/train/models/checkpoint-4000").cuda()
            print("*********PeftModel loaded successfully")
        else:
            print("*********PeftModel not loaded")
        self.model = self.model.eval()


# if __name__ == '__main__':
#     config=LangChainCFG()
#     chatLLM = ChatQwenService()
#     chatLLM.load_model(model_name_or_path=config.llm_model_name)
