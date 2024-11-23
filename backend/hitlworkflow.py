from typing import Union
import os
from pydantic import BaseModel, Field
from llama_index.core.workflow import (
    Workflow,
    step,
    Event,
    Context
)
from llama_index.core.workflow.events import (
    StartEvent,
    StopEvent,
    InputRequiredEvent,
    HumanResponseEvent
)
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
import txtai
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer


os.environ["OPENAI_API_KEY"] = ""
llm = OpenAI("gpt-4o-mini")


class RetryEvent(Event):
    pass


# if the user says the research is good enough, we generate a report
class ReportEvent(Event):
    pass


# we emit progress events to the frontend so the user knows what's happening
class ProgressEvent(Event):
    pass


class OutLineEvent(Event):
    pass


class ResetEvent(Event):
    pass


class Paragraph(BaseModel):
    """A line item in an invoice."""
    subtitle: str = Field(description="The title of this paragraph")
    content: str = Field(description="The content of the paragraph")
    source: str = Field(description="Exact quote from the source document")

    def render(self):
        return f"##{self.subtitle} \n>> {self.content} \n[*] {self.source} \n"


class Article(BaseModel):
    title: str = Field(description="Title of the article")
    paragraphs: list[Paragraph] = Field(description="Paragraphs of the article")

    def render(self):
        paragraphs = '\n'.join(x.render() for x in self.paragraphs)
        return f"""
            {self.title}
            {paragraphs}
        """

class Outline(BaseModel):
    title: str = Field(description="The title of the future article. The outline is directly related to it.")
    section_subtitles: list[str] = Field(description="The subtitle of each section, "
                                                     "they represent what will be discussed in the article")



sllm_article = llm.as_structured_llm(Article)
sllm_outline = llm.as_structured_llm(Outline)
documents = SimpleDirectoryReader(
    input_files=["/home/amor/Documents/code_dw/explorations/on-train-llm-agents-rag/data/paul_graham/paul_graham_essay.txt"],
).load_data()
splitter = SentenceSplitter(chunk_size=512)
nodes = splitter.get_nodes_from_documents(documents)
retriever = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=2,
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)



# this is a dummy workflow to show how to do human in the loop workflows
# the purpose of the flow is to research a topic, get human review, and then write a report
class HITLWorkflow(Workflow):

    """
    @step
    async def retrieve(
        self, ctx: Context, ev: StartEvent | RetryEvent
    ) -> Union[RetrieverEvent, None]:
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")
        if not query:
            return None

        print(f"Query the database with: {query}")

        # store the query in the global context
        await ctx.set("query", query)

        if ev.index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = ev.index.as_retriever(similarity_top_k=2)
        nodes = retriever.retrieve(query)
        print(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)
    """

    @step
    async def outline_gen(self, ctx: Context, ev: StartEvent | ResetEvent) -> InputRequiredEvent:
        ctx.write_event_to_stream(ProgressEvent(msg=f"I am doing the outline of  '{ev.query}'"))
        await ctx.set("original_query", ev.query)

        nodes = retriever.retrieve(ev.query)
        references = '\n'.join(n.text for n in nodes)

        ctx.write_event_to_stream(ProgressEvent(msg=f"Found the following docs : {references}"))

        query = f"""
            Generate an outline of an article about {ev.query}
            Given the following extract of documents.
            {references}
        """
        rez = sllm_outline.complete(query)

        await ctx.set("outline", rez.raw)

        return InputRequiredEvent(prefix="validation", query=ev.query, payload=rez.text)

    # this does the "research", which might involve searching the web or
    # looking up data in a database or our vector store.
    @step
    async def research_query(self, ctx: Context, ev: OutLineEvent | RetryEvent) -> InputRequiredEvent:

        outline: Outline = await ctx.get("outline")

        feedback = ev.get("feedback")
        previous = ev.get("previous")

        nodes = list()
        for x in outline.section_subtitles:
            nodes += retriever.retrieve(x)

        references = '\n'.join(n.text for n in nodes)
        ctx.write_event_to_stream(ProgressEvent(msg=f"Found the following docs : {references}"))

        query = f"""
            Build an article on the following subject {ev.query} based on the source documents below
            ```
            {references}
            ```
            Think step by step by : 
            - Generating an outline first
            - Generating one paragraph per element of the outline. Each paragraph must be sourced from the source documents
            """
        if feedback is not None:
            query += f"""
                The user is not satisfied with the preciseness of the previous answers : 
                {previous}
                Correct the answer based on the following feedback :
                {feedback}
            """
        resp = sllm_article.complete(query).raw.render()
        return InputRequiredEvent(prefix="verification", query=ev.query, payload=resp)
    
    # this accepts the HumanResponseEvent, which is either approval or rejection
    # if it's approval, we write the report, otherwise we do more research
    @step
    async def human_review(self, ctx: Context, ev: HumanResponseEvent) -> ReportEvent | RetryEvent | ResetEvent | OutLineEvent:
        ctx.write_event_to_stream(ProgressEvent(msg=f"The human has responded: {ev.response}"))
        previous = ev.get("payload")
        if (ev.response == "yes"):
            if ev.get("validation") is False:
                return ReportEvent(result=f"Here is the research on {await ctx.get('original_query')}")
            else:
                return OutLineEvent()
        else:
            ctx.write_event_to_stream(ProgressEvent(msg=f"The human has rejected the research, retrying"))
            if ev.get("validation") is False:
                return ResetEvent(query=await ctx.get("original_query"),)
            return RetryEvent(query=await ctx.get("original_query"), feedback=ev.feedback, previous=previous)
        
    # this writes the report, which would be an LLM operation with a bunch of context.
    @step
    async def write_report(self, ctx: Context, ev: ReportEvent) -> StopEvent:
        ctx.write_event_to_stream(ProgressEvent(msg=f"The human has approved the research, generating final report"))
        # generate a report here
        return StopEvent(result=f"This is a report on {await ctx.get('original_query')}")
