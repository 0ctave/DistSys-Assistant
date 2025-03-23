import time
from typing import TypedDict, List, Tuple

from langgraph.constants import START, END

from rag.agents.context_agent import *
from rag.graphs.graph_base import GraphBase


class ContextGraph(GraphBase):
    class InputState(TypedDict):
        context: str
        task: str

    class OutputState(TypedDict):
        result: str

    class ContextGraphState(TypedDict):
        context: str
        task: str
        documents: List[str]
        summaries: List[str]

    def __init__(self, assistant):
        super().__init__(assistant, self.ContextGraphState, input_state=self.InputState, output_state=self.OutputState)

    def _load_agents(self) -> None:
        """
        Loads and initializes agent instances required for executing various tasks.

        This private method establishes a dictionary of agents with specific purposes
        such as description generation, command execution, correctness grading, and
        other operations. Each agent is instantiated and assigned to its corresponding
        key in the `_agents` attribute, ensuring centralized agent management and
        streamlined access throughout the application.

        """
        self._agents = {
            "document_retriever": self._assistant.get_vector_store().retriever(),
            "document_evaluator": get_document_evaluator(),
            "summary_generator": get_summary_generator(),
            "context_generator": get_context_generator(),
        }

    def _load_nodes(self, builder) -> None:
        builder.add_node("document_retriever", self.document_retriever)
        builder.add_node("document_evaluator", self.document_evaluator)
        builder.add_node("summary_generator", self.summary_generator)
        builder.add_node("context_generator", self.context_generator)

    def _load_edges(self, builder) -> None:
        builder.add_edge(START, "document_retriever")
        builder.add_edge("document_retriever", "document_evaluator")
        builder.add_edge("document_evaluator", "summary_generator")
        builder.add_edge("summary_generator", "context_generator")
        builder.add_edge("context_generator", END)

    def document_retriever(self, state: ContextGraphState):
        print("     ---RETRIEVING DOCUMENTS---")
        context = state["context"]
        task = state["task"]
        start_time = time.time()
        documents = self._agents["document_retriever"].invoke(task)
        total_time = time.time() - start_time

        print("     Total time:", total_time)
        return {"context": context, "task": task, "documents": documents}

    def document_evaluator(self, state: ContextGraphState):
        print("     ---EVALUATING DOCUMENTS---")
        context = state["context"]
        task = state["task"]
        documents = state["documents"]

        relevant_documents = []

        start_time = time.time()
        for document in documents:
            evaluation = None
            while not evaluation or not evaluation.relevance:
                if evaluation:
                    print("Failed generation. Retrying...")
                evaluation = self._agents["document_evaluator"].invoke({"context": context, "task": task, "document": document})
            if evaluation.relevance == 'yes':
                relevant_documents.append(document)
        total_time = time.time() - start_time

        print("     Total time:", total_time)
        return {"documents": relevant_documents}

    def summary_generator(self, state: ContextGraphState):
        print("     ---SUMMARIZING DOCUMENTS---")
        task = state["task"]
        documents = state["documents"]

        start_time = time.time()
        summaries = []
        for document in documents:
            generation = None
            while not generation or not generation.summary:
                if generation:
                    print("Failed generation. Retrying...")
                generation = self._agents["summary_generator"].invoke(
                    {"task": task, "document": document})
            summaries.append(generation.summary)
        total_time = time.time() - start_time

        print("     Total time:", total_time)
        return {"summaries": summaries}

    def context_generator(self, state: ContextGraphState):
        print("     ---GENERATING CONTEXT---")
        context = state["context"]
        task = state["task"]
        summaries = state["summaries"]

        start_time = time.time()
        generation = None
        while not generation or not generation.context:
            if generation:
                print("Failed generation. Retrying...")
            generation = self._agents["context_generator"].invoke(
                {"task": task, "summaries": summaries, "context": context})
        total_time = time.time() - start_time

        print("     Total time:", total_time)
        return {"result": generation.context}
