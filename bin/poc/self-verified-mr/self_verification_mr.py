# -*- coding: utf-8 -*-
# file: self_verification_mr.py
# date: 2024-11-09


import pdb
import asyncio
import sys
import os
import traceback
import copy
import json
from typing import Union, Optional, List, Dict, Coroutine, Callable, Any
from pydantic import BaseModel
from openai import AsyncOpenAI, AsyncAzureOpenAI
from openai import ChatCompletion


INIT_GEN_SCHEMA: Dict = \
{
    "type": "json_schema",
    "json_schema": {
        "name": "instruction_generated_elements_schema",
        "schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"}
            }, 
            "required": ["content"],
            "additionalProperties": False
        },
        "strict": True
    }
}


GENERAL_INSTRUCTION_TEMP: str = \
"""
## Your Role
__ROLE__

## Your Given Input
Your input: __INPUT_DESC__

## The Expected Output
Expected Output: __OUTPUT_DESC__

Based on above information, you need generate output based on given input.
"""


def llm_resp_json_clean(in_json: str) -> str:
    return in_json.replace("```json", "").replace("```", "")


def any_to_str(in_data: Any) -> str:
    if isinstance(in_data, str):
        return in_data
    elif isinstance(in_data, int) or isinstance(in_data, float):
        return str(in_data)
    elif isinstance(in_data, list) or isinstance(in_data, dict):
        return json.dumps(
            in_data, ensure_ascii=False, indent=2
        )
    else:
        raise Exception("Failed convert to string")


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
   

class Instruction(BaseModel):
    name: str
    input_desc: Optional[str] = None
    output_desc: Optional[str] = None
    content: Optional[str] = None
    scope: Optional[List[str]] = None
    msgs: Optional[List[Dict[str, str]]] = None
    finished: bool = False


class Instructions(BaseModel):
    instructions: List[Instruction]
    result: Optional[Dict[str, Dict | List | str]] = None
    finished: bool = False


def instructions_init_by_configs(
    conf: List[Dict]
) -> Instructions:
    return Instructions(
        instructions=[
            Instruction.parse_obj(x) for x in conf
        ],
        result=None,
        finished=False
    )


def instructions_to_output(
    instructions: Instructions
) -> Instructions:
    for instruction in instructions.instructions:
        if instruction.finished == False:
            instructions.result = None
            break
        name: str = instruction.name
        val: str = instruction.msgs[-1]["content"]
        """
        val: Dict | List | str = instruction.msgs[-1]["content"]
        try:
            val = json.loads(llm_resp_json_clean(val))
        except Exception as e:
            pass
        """
        if instructions.result is None:
            instructions.result = {}
        instructions.result[name] = val
    instructions.finished = True
    return instructions
        

class InstructionsMapper:
    def __init__(self):
        self.llm: Optional[LlmCli] = None
        self.instructions: Optional[Instructions] = None
        self.role: Optional[str] = None

    @classmethod
    def new_with_llm(
        cls,
        llm: LlmCli,
        role: str,
        instructions: Optional[Instructions]=None
    ):
        out = cls()
        out.llm = llm
        out.role = role
        out.instructions = instructions
        return out

    def inputs_proc(
        self, 
        input_data: Dict | List | str, 
        instruction: Instruction
    ) -> str:
        if isinstance(input_data, dict):
            if instruction.scope is not None:
                input_data = {k: v for k, v in input_data.items() if k in instruction.scope}
            return json.dumps(input_data, indent=2, ensure_ascii=False)
        elif isinstance(input_data, list):
            return json.dumps(input_data, indent=2, ensure_ascii=False)
        else:
            return input_data


class SelfVerifiedMapper(InstructionsMapper):
    def build_extraction_chatml(
        self,
        input_data: str,
        instruction: Instruction
    ) -> Instruction:
        try:
            input_json: Dict = json.loads(input_data)
            if instruction.scope is not None:
                proc_json: Dict = {}
                for k, v in input_json.items():
                    if k not in instruction.scope:
                        continue
                    if isinstance(v, str):
                        proc_json[k] = v
                    else:
                        proc_json[k] = json.dumps(
                            v, indent=2, ensure_ascii=False
                        )
                input_data = json.dumps(
                    proc_json, indent=2, ensure_ascii=False
                )
        except Exception as e:
            print("Input data is not able to serialize to JSON.")
            #print(traceback.format_exc())
            #pdb.set_trace()

        msgs: List[Dict] = [
            {
                "role": "system",
                "content": (
                    "__ROLE__\n"
                    "Your task is to extract elements from given text following the given instruction."
                    "The instruction you need to follow is: __INSTRUCTION__\n"
                    "Your output must be a JSON array of string, which contains the "
                    "elements you extracted from given text by following given instruction. \n"
                    "You have to only return the JSON array of string without anything else."
                )\
                    .replace("__ROLE__", self.role)\
                    .replace("__INSTRUCTION__", instruction.content)
            },
            {
                "role": "user",
                "content": input_data
            }
        ]
        instruction.msgs = msgs
        return instruction

    def build_omission_chatml(
        self,
        input_data: Optional[str],
        instruction: Instruction
    ) -> Instruction:
        prompt: Dict = {
            "role": "user",
            "content": (
                "Based on given text, "
                "check which information are missed in above result, "
                "do complementation if any missing information found.\n"
                "\n"
                "You should only return JSON array of string as above, "
                "with the complementation has been done."
            )
        }
        instruction.msgs.append(prompt)
        return instruction

    def build_evidence_chatml(
        self, 
        input_data: Optional[str],
        instruction: Instruction
    ) -> Instruction:
        prompt: Dict = {
            "role": "user",
            "content": (
                "Find the span of text which corresponds to each instruction response listed above. "
                "If no evidence is found, write \"No evidence can support this statement.\".\n"
                "\n"
                "Your output should be a JSON array of object, "
                "each object is a JSON contains 'content' and 'evidence', "
                "the value of 'content' must be one of the elements listed above, "
                "and the value of 'evidence' is the evidence you extracted from given text, "
                "which can support the value of 'content', "
                "which have to be a text span sourced from given text."
            )
        }
        instruction.msgs.append(prompt)
        return instruction

    def build_chatmls(
        self, 
        input_data: Dict | List | str,
        instructions: List[Instruction], 
        fn: Callable
    ) -> List[Instruction]:
        for instruction in instructions:
            input_text: str = self.inputs_proc(input_data, instruction)
            instruction = fn(input_text, instruction)
        return instructions
    
    async def async_run_extraction(
        self,
        input_data: str,
        instructions: List[Instruction]
    ) -> List[Instruction]:
        instructions = self.build_chatmls(
            input_data, instructions, self.build_extraction_chatml
        )
        tasks: List[Coroutine] = [
            self.llm.async_run(chatml[-1], chatml[:-1]) for chatml in 
            [x.msgs for x in instructions]
        ]
        resps: List[ChatCompletion] = await asyncio.gather(*tasks)
        for i in range(len(instructions)):
            instructions[i].msgs.append({
                "role": "assistant", 
                "content": resps[i].choices[0].message.content
            })
        return instructions

    async def async_run_omission(
        self, 
        instructions: List[Instruction]
    ) -> List[Instruction]:
        instructions = self.build_chatmls(
            None, instructions, self.build_omission_chatml
        )
        tasks: List[Coroutine] = [
            self.llm.async_run(chatml[-1], chatml[:-1]) for chatml in
            [x.msgs for x in instructions]
        ]
        resps: List[ChatCompletion] = await asyncio.gather(*tasks)
        for i in range(len(instructions)):
            instructions[i].msgs.append({
                "role": "assistant", 
                "content": resps[i].choices[0].message.content
            })
        return instructions

    async def async_run_evidence(
        self,
        instructions: List[Instruction]
    ) -> List[Instruction]:
        instructions = self.build_chatmls(
            None, instructions, self.build_evidence_chatml
        )
        tasks: List[Coroutine] = [
            self.llm.async_run(chatml[-1], chatml[:-1]) for chatml in
            [x.msgs for x in instructions]
        ]
        resps: List[ChatCompletion] = await asyncio.gather(*tasks)
        for i in range(len(instructions)):
            instructions[i].msgs.append({
                "role": "assistant", 
                "content": resps[i].choices[0].message.content
            })
        return instructions

    async def async_run_prune_rule_based(
        self,
        instructions: List[Instruction]
    ) -> List[Instruction]:
        for instruction in instructions:
            instruction.finished = True
        return instructions

    async def async_run(
        self,
        prev_instructions: Instructions,
        instructions: Optional[Instructions]=None
    ) -> List[Instruction]:
        if instructions is None:
            instructions = copy.deepcopy(self.instructions)
        assert(instructions is not None)
        assert(len(instructions.instructions) > 0)
        instructions.instructions = await self.async_run_extraction(
            prev_instructions.result, instructions.instructions
        )
        instructions.instructions = await self.async_run_omission(
            instructions.instructions
        )
        instructions.instructions = await self.async_run_evidence(
            instructions.instructions
        )
        instructions.instructions = await self.async_run_prune_rule_based(
            instructions.instructions
        )
        instructions_to_output(instructions)
        return instructions


class InstructionsReducer:
    def __init__(self):
        self.llm: Optional[LlmCli] = None
        self.role: Optional[str] = None
        self.instructions: Optional[Instructions]=None

    @classmethod
    def new_with_llm(
        cls,
        llm: LlmCli,
        role: str,
        instructions: Instructions=None
    ):
        out = cls()
        out.llm = llm
        out.role = role
        out.instructions = instructions
        return out


class RewritingReducer(InstructionsReducer):
    def build_chatml(
        self, 
        prev_instructions: Instructions,
        instruction: Instruction
    ) -> Instruction:
        instruction.msgs = []
        """
        inputs: Dict[str, Dict | List | str] = {
            k: v for k, v in prev_instructions.result.items() 
            if k in instruction.scope
        }
        """
        inputs: Dict[str, str] = {
            k: v for k, v in prev_instructions.result.items()
            if k in instruction.scope
        }
        target_data: str = ""
        for input_name, input_data in inputs.items():
            """
            map_output: str = json.dumps(
                [x["content"] for x in input_data], indent=2
            )
            """
            map_output: str = input_data
            target_data += (
                "<__NAME__>\n"
                "__CONTENT__\n"
                "</__NAME__>\n\n"
            )\
                .replace("__NAME__", input_name)\
                .replace("__CONTENT__", map_output)
   
        instruction.msgs = [
            {
                "role": "system",
                "content": GENERAL_INSTRUCTION_TEMP\
                    .replace("__ROLE__", self.role)\
                    .replace("__INPUT_DESC__", instruction.input_desc)\
                    .replace("__OUTPUT_DESC__", instruction.output_desc)
            },
            {
                "role": "user",
                "content": target_data
            }
        ]
        return instruction

    async def async_run_group(
        self, 
        prev_instructions: Instructions,
        instruction: Instruction
    ) -> Instruction:
        instruction = self.build_chatml(prev_instructions, instruction)
        resp: ChatCompletion = await self.llm.async_run(
            instruction.msgs[-1], instruction.msgs[:-1]
        ) 
        instruction.msgs.append(
            {
                "role": "assistant",
                "content": resp.choices[0].message.content
            }
        )
        instruction.finished = True
        return instruction

    async def async_run(
        self,
        prev_instructions: Instructions,
        instructions: Optional[Instructions]=None
    ) -> Instructions:
        if instructions is None:
            instructions = copy.deepcopy(self.instructions)
        assert(instructions is not None)
        assert(len(instructions.instructions) > 0)
        tasks: List[Coroutine] = [
            self.async_run_group(prev_instructions, x) 
            for x in instructions.instructions
        ]
        instructions.instructions = await asyncio.gather(*tasks)
        instructions_to_output(instructions)
        return instructions


class SelfVerifiedMR:
    def __init__(self):
        self.llm: Optional[LlmCli] = None
        self.map_role: Optional[str] = None
        self.instructions: Optional[List[Instruction]] = None
        self.reduce_role: Optional[str] = None
        self.groups: Optional[List[Instruction]] = None
        self.mapper: Optional[SelfVerifiedMappe] = None
        self.reducer: Optional[RewritingReducer] = None

    @classmethod
    def new_with_llm_and_mr_configs(
        cls,
        llm: LlmCli,
        map_conf: Dict, 
        reduce_conf: Dict
    ):
        out = cls()
        out.llm = llm
        out.mapper = SelfVerifiedMapper.new_with_llm(
            llm=llm,
            role=map_conf["role"],
            instructions=instructions_init_by_configs(
                map_conf["instructions"]
            )
        )
        out.reducer = RewritingReducer.new_with_llm(
            llm=llm,
            role=reduce_conf["role"],
            instructions=instructions_init_by_configs(
                reduce_conf["instructions"]
            )
        )
        return out

    async def async_run(self, init_instructions: Instructions) -> Dict:
        out: Dict = {}
        map_instructions: Instructions = (
            await self.mapper.async_run(init_instructions)
        )
        reduce_instructions: Instructions = (
            await self.reducer.async_run(map_instructions)
        )
        result: str = ""
        for instruction in reduce_instructions.instructions:
            name: str = instruction.name
            final_resp: str = instruction.msgs[-1]["content"]
            result += "# {}\n".format(name)
            result += "{}\n".format(final_resp)
            result += "\n"
        out["result"] = result
        return out


async def main() -> None:
    configs: Dict = json.loads(open(sys.argv[1], "r").read())
    print(configs)
    in_data_path: str = configs["in_data_path"]
    in_text_cols: str = configs["in_text_cols"]
    output_col: str = configs["output_col"]
    map_conf: Dict = configs["runner"]["map"]
    reduce_conf: Dict = configs["runner"]["reduce"]
 
    llm: LlmCli = LlmCli.new_with_configs(configs["llm"])
    runner: SelfVerifiedMR = SelfVerifiedMR.new_with_llm_and_mr_configs(
        llm, map_conf, reduce_conf
    ) 

    # Check
    print("Testing LLM's connection")
    test_resp: Coroutine = llm.async_run("Hi")
    print("Running 'Hi'")
    test_result: str = (await test_resp).choices[0].message.content
    print(test_result)
    print("Testing finished")

    in_samples: List[Dict] = [
        json.loads(x) for x in open(in_data_path, "r").read().split("\n")
        if x not in {""}
    ]
    for in_sample in in_samples:
        init_instructions: Instructions = Instructions(
            instructions=[], 
            result=in_sample, 
            finished=True
        )
        output: Dict = await runner.async_run(init_instructions)
        print(output["result"])
    return


if __name__ == "__main__":
    asyncio.run(main())
