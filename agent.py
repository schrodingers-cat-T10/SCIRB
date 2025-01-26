from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_cohere.chat_models import ChatCohere
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langchain.agents import Tool
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import os
import pandas as pd

os.environ['COHERE_API_KEY'] ="yTmKdlP6vaGOZ91YAlPCKqMUpvmD2rgSoZqZJRHS"

chat = ChatCohere(model="command-r-plus", temperature=0.3)
prompt = ChatPromptTemplate.from_template("{input}")

python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="Executes python code and returns the result. The code runs in a static sandbox without interactive mode, so print output or save output to a file.",
    func=python_repl.run,
)

# from langchain_core.pydantic_v1 import BaseModel, Field
class ToolInput(BaseModel):
    code: str = Field(description="Python code to execute.")
repl_tool.args_schema = ToolInput


class get_csv:
    def get_csv():
        pass

        
    
class get_csv(BaseModel):
    code: str = Field(description="you can access the database and answer for the quires like student tracking")

get_csver=Tool(
    name="tracker_tool",
    description="you can access the database and answer for the quires like student tracking",
    func=get_csv.tracking,
    arg_schema=get_csv
)



agent = create_cohere_react_agent(
    llm=chat,
    tools=[repl_tool,get_csver],
    prompt=prompt,
)

agent_executor = AgentExecutor(agent=agent, tools=[repl_tool,get_csver], verbose=True)



result = agent_executor.invoke({"input": "print('Hello, World!')"})
print("response",result)