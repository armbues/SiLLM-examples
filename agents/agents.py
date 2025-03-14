import operator
import inspect
import typing
import re
import json

from RestrictedPython.compile import compile_restricted_exec
from RestrictedPython.PrintCollector import PrintCollector
from RestrictedPython import safe_globals

class Agent():
    """
    Base class for agents.
    """
    def __init__(self):
        raise NotImplementedError("Class agents.Agent is used for inheritance only")
    
    def handle_response(self, response):
        return None

def json_function_stub(name, func):
    """
    Generate a JSON stub from a function object.
    """
    stub = {
        "type": "function",
        "function": {
            "name": name,
            "description": "",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }

    sig = inspect.signature(func)
    for param in sig.parameters.values():
        param_entry = {}

        if typing.get_origin(param.annotation) is typing.Annotated:
            args = typing.get_args(param.annotation)
            param_entry["type"] = args[0].__name__
            param_entry["description"] = args[1]
        else:
            param_entry["type"] = param.annotation.__name__

        if param.default == inspect._empty:
            stub["function"]["parameters"]["required"].append(param.name)
        # else:
        #     param_entry["default"] = param.default

        stub["function"]["parameters"]["properties"][param.name] = param_entry

    docstring = inspect.getdoc(func)
    if docstring is not None:
        stub["function"]["description"] = docstring

    return stub

class ToolAgent(Agent):
    """
    Agent that handles JSON tool function calls.

    Use typing.Annotated to provide parameter descriptions.
    """
    def __init__(self, tool_functions: list|dict):
        if isinstance(tool_functions, list):
            tool_functions = {func.__name__:func for func in tool_functions}
        self.tool_functions = tool_functions
        self.tools = [json_function_stub(name, func) for name, func in tool_functions.items()]

    def handle_response(self, response):
        response_json = None
        if '<tool_call>' in response and '</tool_call>' in response:
            response_json = re.findall(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)[0]
        elif '```' in response:
            response_json = re.findall(r'```(.*?)```', response, re.DOTALL)[0]
            response_json = response_json.replace("'", '"')
        elif response.strip().startswith("{"):
            response_json = response.strip()
        else:
            return None
        
        try:
            function_calls = json.loads(response_json)
        except:
            return "Error: invalid JSON format."
        
        if not isinstance(function_calls, list):
            function_calls = [function_calls]

        tool_response = ""
        for function_call in function_calls:
            name = function_call.get("name", None)
            if name is None:
                return "Error: missing function name in function call."
            func = self.tool_functions.get(name, None)
            if func is None:
                return f"Error: unknown function {function_call['name']}."

            if "parameters" in function_call:
                params = function_call["parameters"]
            elif "arguments" in function_call:
                params = function_call["arguments"]
            else:
                return "Error: missing parameters in function call."
            
            try:
                result = func(**params)
            except Exception as e:
                return f"Error: function call failed"
            
            try:
                tool_response += json.dumps(result) + "\n"
            except:
                return "Error: function call did not return JSON serializable result."
            
        return tool_response

def code_function_stub(func):
    """
    Generate a function stub from a function object.
    """
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func)

    header = f"def {func.__name__}{signature}:\n"
    if docstring:
        header += '    """\n'
        for line in docstring.splitlines():
            header += "    " + line + "\n"
        header += '    """\n'
    header += "    pass"

    return header

default_code_system_prompt = """You are a Python coding expert and ReAct agent that acts by writing executable code.
At each step I will execute the code that you wrote and send you the execution result.
Then continue with the next step by reasoning and writing executable code until you have a final answer.
The final answer must be in plain text or markdown (exclude code and exclude latex).

You can use the following available functions:
{functions}

Think step by step and provide your reasoning, outside of the function calls.
You can write Python code but ONLY using the available functions. Use the print function to return the execution result for each step. You MUST NOT use imports or external libraries.

Provide all your python code in a SINGLE markdown code block like the following:
```python
var1 = example_function(arg1, "string")
result = example_function2(var1, arg2)
print(result)
```

Remember to only execute code for one step at a time and wait for the execution result to inspect the return values. All python markdown code you provide in your responses will be executed in order."""

class CodeAgent(Agent):
    """
    Agent that handles code execution in a Python sandbox.
    """
    def __init__(self, tool_functions: list|dict):
        if isinstance(tool_functions, list):
            tool_functions = {func.__name__:func for func in tool_functions}
        self.tool_functions = tool_functions

        self.sandbox_globals = safe_globals.copy()
        self.sandbox_globals.update(tool_functions)
        self.sandbox_globals["_print_"] = PrintCollector
        self.sandbox_globals["_getitem_"] = operator.getitem

        self.reset()

    def reset(self):
        """
        Reset the sandbox environment.
        """
        self.sandbox_locals = {}

    def format_system_prompt(self, prompt=None):
        """
        Format the system prompt with the available tool functions.

        Args:
            prompt (str, optional): System prompt template. Uses default template if None. Defaults to None.
        Returns:
            str: Formatted system prompt.
        """
        if prompt is None:
            prompt = default_code_system_prompt

        tool_block = "```python\n" + "\n\n".join([code_function_stub(func) for func in self.tool_functions.values()]) + "\n```"
        if "{functions}" not in prompt:
            raise ValueError("Prompt does not contain {functions} placeholder")

        return prompt.replace("{functions}", tool_block)
    
    def handle_response(self, response):
        """
        Handle the LLM response by extracting & executing code blocks and returning the result.

        Args:
            response (str): LLM response.
        Returns:
            str: Execution result, error message, or None in case of no code blocks.
        """
        # Extract code blocks
        code_blocks = []
        if '```' in response:
            if '```python' not in response:
                return "Error: code block must be formatted as ```python ... ```"

            code_blocks = re.findall(r'```python(.*?)```', response, re.DOTALL)

        # No code blocks found
        if len(code_blocks) == 0:
            return None
        
        # Multiple code blocks found
        if len(code_blocks) > 1:
            return "Error: multiple code blocks found"
        
        # Compile code block
        code_block = code_blocks[0]
        compiled = compile_restricted_exec(code_block, filename="<agent>")

        # Handle compilation errors
        if compiled.code is None:
            return "Compilation errors:\n" + "\n".join(compiled.errors)
        
        # Execute code block
        try:
            exec(compiled.code, self.sandbox_globals, self.sandbox_locals)
        except Exception as e:
            return f"Error: {e}"
        
        if "_print" in self.sandbox_locals:
            printed = self.sandbox_locals["_print"]().strip()
            del self.sandbox_locals["_print"]

            if len(printed) > 0:
                return printed
            else:
                return "Error: result string is empty."
        else:
            return "Error: No print statement executed in code block."