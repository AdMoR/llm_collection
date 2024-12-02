import os
import asyncio
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step
from llama_index.core import set_global_handler, global_handler

os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-cff8c32a-05d7-4ca7-924b-592ab992b8ab"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-b97a846a-eb65-450a-8c14-bb95070676e8"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
set_global_handler("langfuse")

class EchoWorkflow(Workflow):
    """A dummy workflow with only one step sending back the input given."""

    @step()
    async def run_step(self, ev: StartEvent) -> StopEvent:
        message = str(ev.get("message", ""))
        print("Coucouc")
        return StopEvent(result=f"Message received: {message}")


# `echo_workflow` will be imported by Llama Deploy
echo_workflow = EchoWorkflow()


async def main():
    print(await echo_workflow.run(message="Hello!"))


# Make this script runnable from the shell so we can test the workflow execution
if __name__ == "__main__":
    asyncio.run(main())