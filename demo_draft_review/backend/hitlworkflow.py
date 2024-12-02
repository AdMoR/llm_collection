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

    def render(self, index=None):
        if index:
            return f"{index} - {self.subtitle} \n{self.content} \n"
        return f"##{self.subtitle} \n>> {self.content} \n"


class Article(BaseModel):
    title: str = Field(description="Title of the article")
    paragraphs: list[Paragraph] = Field(description="Paragraphs of the article")

    def render(self):
        paragraphs = '\n'.join(x.render(i) for i, x in enumerate(self.paragraphs))
        return f"""
            Title: {self.title}
            {paragraphs}
        """

class Outline(BaseModel):
    title: str = Field(description="The title of the future article. The outline is directly related to it.")
    section_subtitles: list[str] = Field(description="The subtitle of each section, "
                                                     "they represent what will be discussed in the article")


class FactCheck(BaseModel):
    grounded: bool = Field(description="If the extract is grounded on one of the sources provided")
    source: str = Field(description="The corresponding source on which extract is based upon")

    def render(self):
        if self.grounded:
            return f"Source is {self.source}"
        else:
            return "Not sourced"


sllm_article = llm.as_structured_llm(Article)
sllm_outline = llm.as_structured_llm(Outline)
sllm_fc = llm.as_structured_llm(FactCheck)
documents = SimpleDirectoryReader(
    input_files=["../paul_graham_essay.txt"],
).load_data()
splitter = SentenceSplitter(chunk_size=256)
nodes = splitter.get_nodes_from_documents(documents)
retriever_top_5 = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=5,
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)
retriever_top_1 = BM25Retriever.from_defaults(
    nodes=nodes,
    similarity_top_k=1,
    stemmer=Stemmer.Stemmer("english"),
    language="english",
)


class HITLWorkflow(Workflow):

    @step
    async def outline_gen(self, ctx: Context, ev: StartEvent | ResetEvent) -> InputRequiredEvent:
        ctx.write_event_to_stream(ProgressEvent(msg=f"I am doing the outline of  '{ev.query}'"))
        await ctx.set("original_query", ev.query)

        nodes = retriever_top_5.retrieve(ev.query)
        references = '\n'.join(n.text for n in nodes)

        ctx.write_event_to_stream(ProgressEvent(msg=f"Found the following docs : {references}"))

        query = f"""
            Generate an outline of an article about {ev.query}
            Given the following extract of documents.
            {references}
        """
        rez = sllm_outline.complete(query)

        await ctx.set("outline", rez.raw)

        return InputRequiredEvent(prefix="outline", query=ev.query, payload=rez.text)

    # this does the "research", which might involve searching the web or
    # looking up data in a database or our vector store.
    @step
    async def research_query(self, ctx: Context, ev: OutLineEvent | RetryEvent) -> InputRequiredEvent:

        outline: Outline = await ctx.get("outline")

        feedback = ev.get("feedback")
        previous = ev.get("previous")

        nodes = list()
        for x in outline.section_subtitles:
            nodes += retriever_top_1.retrieve(x)

        references = '\n'.join(n.text[:100] for n in nodes)
        ctx.write_event_to_stream(ProgressEvent(msg=f"Found the following docs : {references}"))

        query = f"""
            Build an article on the following subject {outline.title} based on the source documents below
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
        article: Article = sllm_article.complete(query).raw
        references = '\n-'.join(n.text for n in nodes)
        fact_checks = []
        for i, x in enumerate(article.paragraphs):
            query = f"""
                Given this paragraph and sources, define if the paragraph is source from one of the sources.
                Example 1 : 
                Paragraph: I live for 10 years in London where I got to learn about the intricate details of the financial system. There I specialized in crypto.
                source : 
                - My favorite color is red.
                - London is always rainy
                - I work for Morgan Stanley in the City as a data analyst.
                answer: 
                grounded : True; quote: I work for Morgan Stanley in the City as a data analyst.
                 Given this paragraph and sources, define if the paragraph is source from one of the sources.
                Example 2 : 
                paragraph: i wrote a bout about medical knowledge 
                Sources : 
                - I wrote a book named "1000 recipes about pork"
                - I am a professional surgeon
                answer : 
                grounded: False, quote: None
                Now do it for the following : 
                Paragraph : {x.content}
                Sources : 
                {references}
            """
            fact_checks.append(sllm_fc.complete(query).raw)
            ctx.write_event_to_stream(ProgressEvent(msg=query))
            ctx.write_event_to_stream(ProgressEvent(msg=f"Checked paragraph {i} : {fact_checks[-1].render()}"))

        return InputRequiredEvent(prefix="article", query=outline.title, payload=article.render())
    
    # this accepts the HumanResponseEvent, which is either approval or rejection
    # if it's approval, we write the report, otherwise we do more research
    @step
    async def human_review(self, ctx: Context, ev: HumanResponseEvent) -> ReportEvent | RetryEvent | ResetEvent | OutLineEvent:
        ctx.write_event_to_stream(ProgressEvent(msg=f"The human has responded: {ev.response}"))
        previous = ev.response
        if ev.validation == "yes":
            if ev.step == "article_review":
                return ReportEvent(result=f"Here is the research on {await ctx.get('original_query')}")
            else:
                return OutLineEvent()
        else:
            ctx.write_event_to_stream(ProgressEvent(msg=f"The human has rejected the research, retrying"))
            if ev.step == "outline_review":
                return ResetEvent(query=await ctx.get("original_query"),)
            return RetryEvent(query=await ctx.get("original_query"), feedback=ev.feedback, previous=previous)
        
    # this writes the report, which would be an LLM operation with a bunch of context.
    @step
    async def write_report(self, ctx: Context, ev: ReportEvent) -> StopEvent:
        ctx.write_event_to_stream(ProgressEvent(msg=f"The human has approved the research, generating final report"))
        # generate a report here
        return StopEvent(result=f"This is a report on {await ctx.get('original_query')}")
