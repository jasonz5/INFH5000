#!/usr/bin/env python
# -*- coding:utf-8 _*-

from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate

from clc.config import LangChainCFG
from clc.gpt_service import ChatQwenService


class LangChainApplication(object):
    def __init__(self, config):
        self.config = config
        self.llm_service = ChatQwenService()
        self.llm_service.load_model(model_name_or_path=self.config.llm_model_name)


    def get_llm_answer(self, query=''):
        prompt = query
        result = self.llm_service._call(prompt)
        print("************get_llm_answer***********result:",result)
        return result


# if __name__ == '__main__':
#     config = LangChainCFG()
#     application = LangChainApplication(config)
#     # result = application.get_knowledge_based_answer('香港科技大学成立背景是什么')
#     # print(result)
#     # application.source_service.add_document('/home/searchgpt/yq/Knowledge-ChatGLM/docs/港科广.txt')
#     # result = application.get_knowledge_based_answer('香港科技大学成立背景是什么')
#     # print(result)
#     result = application.get_llm_answer('香港科技大学成立背景是什么')
#     print(result)
