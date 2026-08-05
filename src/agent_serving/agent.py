# Databricks notebook source

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit
)
from langchain_core.tools import tool

from langchain.agents import create_agent


# COMMAND ----------
# SECTION 1: MLflow setup
mlflow.langchain.autolog()

CATALOG = "main"
SCHEMA = "default"
table_prefix = "traces"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"


# COMMAND ----------
# SECTION 2: UC Function tools
# DatabricksFunctionClient defaults to serverless execution

toolkit = UCFunctionToolkit(function_names=[
    f"{CATALOG}.{SCHEMA}.get_employee_leave_balance",
    f"{CATALOG}.{SCHEMA}.get_department_headcount",
])
tools = toolkit.tools


@tool
def query_hr_genie(question: str) -> str:
    """ ALWAYS use this tool when the user asks any question about:
    - salary (average, distribution, by department, by role)
    - headcount or number of employees across departments
    - who has the most or least leave remaining
    - trends or comparisons across multiple employees
    - any aggregate or analytical question about the HR database
    This tool queries the HR employee database using natural language.
    Do NOT use for looking up one specific employee by ID. """

    from databricks.sdk import WorkspaceClient  # import inside too
    w = WorkspaceClient()   

    response = w.genie.start_conversation_and_wait(
    space_id="01f18deae2f61879ab492990908453df",
    content=question)

    result = ""
    for attachment in response.attachments:
        if attachment.text and attachment.text.content:
            result += attachment.text.content + "\n"
    return result

tools = tools + [query_hr_genie]
# COMMAND ----------
# SECTION 3: Agent
# create_agent replaces create_react_agent
# system_prompt replaces state_modifier
llm = ChatDatabricks(endpoint=LLM_ENDPOINT, temperature=0.1)

graph = create_agent(
    llm,
    tools,
    system_prompt=(
    "You are an HR Operations assistant. "
    "Use get_employee_leave_balance for one employee's leave balance. "
    "Use get_department_headcount for team size and open roles. "
    "Use query_hr_genie for ANY analytical question across many employees — "
    "salary averages, headcount trends, top earners, leave statistics. "
    "Be concise. Never invent data."
    ),
)

# COMMAND ----------
# SECTION 4: ResponsesAgent wrapper
import uuid

class HROpsAgent(ResponsesAgent):

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:

        # convert input messages to dicts for LangGraph
        input_messages = [
            {"role": message.role, "content": message.content}
            for message in request.input
        ]

        # run the agent
        result = graph.invoke({"messages": input_messages})
        final  = result["messages"][-1].content

        # count tool calls
        tool_call_count = 0
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_call_count += 1

        # use the built-in helper — handles id, type, status, content format for you
        return ResponsesAgentResponse(
            output=[
                self.create_text_output_item(
                    text=final,
                    id=str(uuid.uuid4()),
                )
            ]
        )

agent = HROpsAgent()
mlflow.models.set_model(agent)

