import os
import asyncio
from langfuse.decorators import observe
from llama_index.core import set_global_handler, global_handler
import os
from pydantic import BaseModel, Field
from llama_index.core.workflow import (
    Workflow,
    step,
    Event,
    Context,
    StartEvent,
    StopEvent
)
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core import (
    SimpleDirectoryReader,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer


os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf--05d7--924b-592ab992b8ab"
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf--eb65--8c14-bb95070676e8"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"
set_global_handler("langfuse")

class EchoWorkflow(Workflow):
    """A dummy workflow with only one step sending back the input given."""

    @observe()
    @step()
    async def run_step(self, ev: StartEvent) -> StopEvent:
        message = str(ev.get("message", ""))
        documents = SimpleDirectoryReader(
            input_files=["../notebooks/paul_graham_essay.txt"],
        ).load_data()
        splitter = SentenceSplitter(chunk_size=256)
        nodes = splitter.get_nodes_from_documents(documents)
        retriever_top_5 = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=5,
            stemmer=Stemmer.Stemmer("english"),
            language="english",
        )
        nodes = retriever_top_5.retrieve(message)
        return StopEvent(result=f"Message received: {nodes[0].text}")


# `echo_workflow` will be imported by Llama Deploy
echo_workflow = EchoWorkflow()


async def main():
    print(await echo_workflow.run(message="Hello!"))


# Make this script runnable from the shell so we can test the workflow execution
if __name__ == "__main__":
    asyncio.run(main())