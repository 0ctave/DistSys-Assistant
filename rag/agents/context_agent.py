from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag.utils.llm import llm


def get_document_evaluator():
    class DocumentEvaluation(BaseModel):
        """Relevance of the document to the task."""

        relevance: str = Field(
            description="Document relevance score 'yes' or 'no'"
        )

    structured_llm_evaluator = llm().with_structured_output(DocumentEvaluation)

    # Prompt
    system = """You are a grader assessing whether a given document is useful for a task. \n 
            Give a binary score 'yes' or 'no' score to indicate whether the document is useful to solve the task. \n
            Context : \n{context}"""
    plan_completion_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Task : {task} \n\n\n Document: \n {document}"),
        ]
    )

    return plan_completion_prompt | structured_llm_evaluator


def get_summary_generator():
    class Summary(BaseModel):
        """Summary of the document."""

        summary: str = Field(
            description="Summary of a document summary"
        )

    structured_llm_summarizer = llm().with_structured_output(Summary)

    # Prompt
    system = """You are an assistant summarising a document. \n 
        Give a summary of the document to extract only the useful information. \n
        The summary should be a concise but precise summary of the document. \n
        Make sure the summary is accurate and is based on document. \n
        If you don't have any information relevant to the task answer "Nothing useful". \n
        Don't answer with a solution to the task just give the context information for it. \n
        Stay grounded to the document. \n"""
    generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Task: \n {task} \n\n\n Document: \n {document} \n\n\n"),
        ]
    )

    return generator_prompt | structured_llm_summarizer


def get_context_generator():
    class Context(BaseModel):
        """Generate context information for the task."""

        context: str = Field(
            description="The generated context"
        )

    structured_llm_context_generator = llm().with_structured_output(Context)

    # Prompt
    system = """You are in assistant generating context information for a task with summaries of document. \n
                You have access to a list of summaries, and the task to address, generate the context information that will be used to address the task. \n
                The context information should be a concise concatenation of the summaries. \n
                Make sure the context information is accurate and relevant to the task. \n
                Combine the generated context information with the given previous context. \n
                Don't answer with a solution to the task just give the context information. \n
                Stay grounded to the task and the summaries. \n"""
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Task: {task} \n\n\n Summaries: {summaries} \n\n\n Previous Context: {context}"),
        ]
    )

    return planner_prompt | structured_llm_context_generator
