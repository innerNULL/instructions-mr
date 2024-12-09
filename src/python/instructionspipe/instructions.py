# -*- coding: utf-8 -*-
# file: instructions.py.py
# date: 2024-12-09


from typing import Union, Optional, List, Dict, Coroutine, Callable, Any
from pydantic import BaseModel


class Instruction(BaseModel):
    name: str
    input_desc: Optional[str] = None
    output_desc: Optional[str] = None
    content: Optional[str] = None
    role: Optional[str] = None
    scope: Optional[List[str]] = None
    msgs: Optional[List[Dict[str, str]]] = None
    finished: bool = False


class Instructions(BaseModel):
    instructions: List[Instruction]
    result: Optional[Dict[str, str]] = None
    finished: bool = False


def instructions_init_by_configs(
    configs: List[Dict]
) -> Instructions:
    return Instructions(
        instructions=[
            Instruction.parse_obj(x) for x in configs
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


def instructions_to_md(instructions: Instructions) -> str:
    result: str = ""
    for instruction in instructions.instructions:
        name: str = instruction.name
        final_resp: str = instruction.msgs[-1]["content"]
        result += "# {}\n".format(name)
        result += "{}\n".format(final_resp)
        result += "\n"
    return result
    
