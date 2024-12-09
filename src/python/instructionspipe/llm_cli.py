# -*- coding: utf-8 -*-
# file: llm_cli.py
# date: 2024-12-09


from typing import Union, Optional, List, Dict, Coroutine, Callable, Any
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai import ChatCompletion


class LlmCli:
    def __init__(self):
        self.cli: Optional[OpenAI | AsyncAzureOpenAI] = None
        self.async_cli: Optional[AsyncOpenAI] = None
        self.model: Optional[str] = None
        self.seed: Optional[int] = None
        self.temperature: Optional[float] = None
        self.top_p: Optional[float] = None

    @classmethod
    def new(
        cls, 
        model: str, 
        api_key: str,
        api_url: str, 
        seed: int=2, 
        api_type: str="openai",
        api_version: str="",
        temperature: float=0.0, 
        top_p: float=0.01
    ):
        out = cls()
        if api_type == "openai":
            out.async_cli = AsyncOpenAI(api_key=api_key, base_url=api_url)
        # Refer to https://elanthirayan.medium.com/using-azure-openai-with-python-a-step-by-step-guide-415d5850169b
        elif api_type == "azure":
            out.async_cli = AsyncAzureOpenAI(
                azure_endpoint=api_url, 
                api_version=api_version,
                api_key=api_key
            )
        out.model = model
        out.seed = seed
        out.temperature = temperature
        out.top_p = top_p
        return out

    @classmethod
    def new_with_configs(cls, configs: Dict):
        return cls.new(
            model=configs["model"], 
            api_key=configs["api_key"],
            api_url=configs["api_url"],
            temperature=configs["temperature"],
            seed=configs["seed"],
            api_type=configs["api_type"],
            api_version=configs["api_version"]
        )

    async def async_run(
        self, 
        msg: Union[str, Dict],
        prefix: Union[Dict, List[Dict]]=None,
        json_schema: Optional[Dict]=None,
        temperature: Optional[float]=None, 
        max_tokens: int=320
    ):
        if isinstance(msg, str):
            msg = {"role": "user", "content": msg}
        if prefix is None:
            prefix = []
        if isinstance(prefix, dict):
            prefix = [prefix]
        return await self.async_cli.chat.completions.create(
            model=self.model, 
            messages=prefix + [msg],
            seed=self.seed,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens,
            top_p=self.top_p,
            response_format=json_schema
        )
