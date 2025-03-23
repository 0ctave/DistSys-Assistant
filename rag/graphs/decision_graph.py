import time
from typing import TypedDict, Tuple

from langgraph.constants import START, END

from rag.agents.decision_agent import *
from rag.graphs.graph_base import GraphBase


class DecisionGraph(GraphBase):
    class InputState(TypedDict):
        query: str

    class OutputState(TypedDict):
        answer: str

    class DecisionGraphState(TypedDict):
        query: str

        task: Tuple[str, str, str]  # task, task completion, task completion score
        subtask: Tuple[str, str]  # subtask, subtask completion, subtask completion score
        plan: Tuple[str, str, str, str]  # plan, last step, plan completion, plan completion score

        action: Tuple[str, str, str]

        data: str

        context: str

    def __init__(self, assistant):
        super().__init__(assistant, self.DecisionGraphState, input_state=self.InputState, output_state=self.OutputState)

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
            "task_generator": get_task_generator(),
            "system_context_query_generator": get_system_context_query_generator(),
            "task_evaluator": get_task_evaluator(),
            "task_grader": get_task_grader(),
            "subtask_generator": get_subtask_generator(),
            "plan_generator": get_plan_generator(),
            "step_generator": get_step_generator(),
            "step_evaluator": get_step_evaluator(),
            "action_executor": self._assistant.get_command_graph(),
            "context_getter": self._assistant.get_context_graph(),
            "content_generator": get_content_generator(),
            "data_generator": get_data_generator(),
            "plan_evaluator": get_plan_evaluator(),
            "plan_grader": get_plan_grader(),
            "answer_generator": get_answer_generator(),
        }

    def _load_nodes(self, builder) -> None:
        builder.add_node("task_generator", self.task_generator)
        builder.add_node("system_context_query_generator", self.system_context_query_generator)
        builder.add_node("system_context_getter", self.system_context_getter)

        builder.add_node("task_context_getter", self.task_context_getter)
        builder.add_node("task_evaluator", self.task_evaluator)

        builder.add_node("subtask_generator", self.subtask_generator)

        builder.add_node("plan_generator", self.plan_generator)

        builder.add_node("step_generator", self.step_generator)

        builder.add_node("action_executor", self.action_executor)
        builder.add_node("context_getter", self.context_getter)
        builder.add_node("content_generator", self.content_generator)
        builder.add_node("data_generator", self.data_generator)

        builder.add_node("plan_evaluator", self.plan_evaluator)
        builder.add_node("subtask_evaluator", self.subtask_evaluator)

        builder.add_node("answer_generator", self.answer_generator)

    def _load_edges(self, builder) -> None:
        builder.add_edge(START, "task_generator")
        builder.add_edge("task_generator", "system_context_query_generator")
        builder.add_edge("system_context_query_generator", "system_context_getter")
        builder.add_edge("system_context_getter", "task_context_getter")

        builder.add_edge("task_context_getter", "task_evaluator")

        builder.add_conditional_edges(
            "task_evaluator",
            self.task_evaluation,
            {
                "incomplete": "subtask_generator",
                "complete": "answer_generator",
                "abort": "answer_generator",
            },
        )

        builder.add_edge("subtask_generator", "plan_generator")
        builder.add_edge("plan_generator", "step_generator")

        builder.add_conditional_edges(
            "step_generator",
            self.step_evaluation,
            {
                "action": "action_executor",
                "context": "context_getter",
                "generation": "action_executor",
            },
        )

        builder.add_edge("action_executor", "data_generator")
        builder.add_edge("context_getter", "data_generator")
        builder.add_edge("content_generator", "data_generator")
        builder.add_edge("data_generator", "plan_evaluator")

        builder.add_conditional_edges(
            "plan_evaluator",
            self.plan_evaluation,
            {
                "incomplete": "plan_generator",
                "complete": "subtask_evaluator",
                "abort": "subtask_evaluator",
            },
        )

        builder.add_edge("subtask_evaluator", "task_evaluator")

        builder.add_edge("answer_generator", END)

    def task_generator(self, state: DecisionGraphState):
        print("---GENERATING TASK---")
        query = state["query"]

        start_time = time.time()
        generation = None
        while not generation or not generation.task:
            if generation:
                print("Failed generation. Retrying...")
            generation = self._agents["task_generator"].invoke(
                {"query": query})
        total_time = time.time() - start_time

        print("Total time:", total_time)

        return {"task": (generation.task, "Nothing has been done yet.", "no"), "data": "No data."}

    def system_context_query_generator(self, state: DecisionGraphState):
        print("---GENERATING SYSTEM CONTEXT QUERY---")
        task = state["task"]

        start_time = time.time()
        generation = None
        while not generation or not generation.context_query:
            if generation:
                print("Failed generation. Retrying...")
            generation = self._agents["system_context_query_generator"].invoke(
                {"task": task})
        total_time = time.time() - start_time

        print("Total time:", total_time)

        return {"context": generation.context_query}

    def system_context_getter(self, state: DecisionGraphState):
        print("---GETTING SYSTEM CONTEXT---")
        (task, _, _) = state["task"]
        context = state["context"]

        start_time = time.time()
        context = self._agents["context_getter"].invoke(
            {"task": context, "context": ""})
        total_time = time.time() - start_time

        print("Total time:", total_time)

        return {"context": context["result"]}

    def task_context_getter(self, state: DecisionGraphState):
        print("---GETTING TASK CONTEXT---")
        (task, _, _) = state["task"]
        context = state["context"]

        start_time = time.time()
        getter = self._agents["context_getter"].invoke({"context": context, "task": task})
        total_time = time.time() - start_time

        print("Total time:", total_time)

        return {"context": getter["result"]}

    def task_evaluator(self, state: DecisionGraphState):
        print("---EVALUATING TASK---")
        context = state["context"]
        data = state["data"]
        (task, task_completion, _) = state["task"]
        (_, subtask_completion) = state.get("subtask", ("", "Nothing has been done yet."))

        start_time = time.time()
        evaluator = None
        while not evaluator or not evaluator.completion:
            if evaluator:
                print("Failed generation. Retrying...")
            evaluator = self._agents["task_evaluator"].invoke(
                {"context": context, "task": task, "completion": task_completion,
                 "progress": subtask_completion, "data": data})
        print("---GRADING TASK---")
        grader = None
        while not grader or not grader.score:
            if grader:
                print("Failed generation. Retrying...")
            grader = self._agents["task_grader"].invoke(
                {"context": context, "task": task, "completion": evaluator.completion})
        total_time = time.time() - start_time

        print("Total time:", total_time)

        return {"task": (task, evaluator.completion, grader.score)}

    def subtask_generator(self, state: DecisionGraphState):
        print("---GENERATING SUBTASK---")
        context = state["context"]
        (task, completion, score) = state["task"]
        (subtask, _) = state.get("subtask", ("No subtask have been generated yet.", "Nothing has been done yet."))

        start_time = time.time()
        generation = None
        while not generation or not generation.task:
            if generation:
                print("Failed generation. Retrying...")
            generation = self._agents["subtask_generator"].invoke(
                {"context": context, "task": task, "subtask": subtask, "completion": completion})
        total_time = time.time() - start_time

        print("Total time:", total_time)

        return {"subtask": (generation.task, "Nothing has been done yet.")}

    def plan_generator(self, state: DecisionGraphState):
        print("---GENERATING PLAN---")
        context = state["context"]
        data = state["data"]
        (task, _) = state["subtask"]
        (_, step, completion, score) = state.get("plan", ("", "No step has been generated yet.", "Nothing has been done yet.", "no"))
        start_time = time.time()
        generation = None
        while not generation or not generation.plan:
            if generation:
                print("Failed generation. Retrying...")
            generation = self._agents["plan_generator"].invoke(
                {"context": context, "data": data, "task": task})
        total_time = time.time() - start_time

        print("Total time:", total_time)

        return {"plan": (generation.plan, step, completion, score)}

    def step_generator(self, state: DecisionGraphState):
        print("---GENERATING NEXT STEP---")
        context = state["context"]
        data = state["data"]
        (task, _) = state["subtask"]
        (plan, last_step, completion, score) = state["plan"]

        start_time = time.time()
        generation = None
        while not generation or not generation.step:
            if generation:
                print("Failed generation. Retrying...")
            generation = self._agents["step_generator"].invoke(
                {"context": context, "task": task, "plan": plan, "completion": completion,
                 "step": last_step, "path": self._assistant.get_path(), "data": data})
        total_time = time.time() - start_time

        print("Total time:", total_time)

        return {"plan": (plan, generation.step, completion, score)}

    def action_executor(self, state: DecisionGraphState):
        print("---EXECUTION ACTION---")
        context = state["context"]
        (_, step, _, _) = state["plan"]

        start_time = time.time()
        execution = None
        while not execution or not execution["action"]:
            if execution:
                print("Failed generation. Retrying...")
            execution = self._agents["action_executor"].invoke({"context": context, "task": step})
        total_time = time.time() - start_time

        print("Total agent time:", total_time)

        return {"action": (execution["action"], execution["description"], execution["result"])}

    def context_getter(self, state: DecisionGraphState):
        print("---GETTING CONTEXT---")
        context = state["context"]
        (_, step, _, _) = state["plan"]

        start_time = time.time()
        getter = None
        while not getter or not getter["result"]:
            if getter:
                print("Failed generation. Retrying...")
            getter = self._agents["context_getter"].invoke({"context": context, "task": step})
        total_time = time.time() - start_time

        print("Total agent time:", total_time)

        return {"action": ("context_getter", step, getter["result"])}


    def content_generator(self, state: DecisionGraphState):
        print("---GENERATING CONTENT---")
        context = state["context"]
        (_, step, _, _) = state["plan"]

        start_time = time.time()
        generation = None
        while not generation or not generation.content:
            if generation:
                print("Failed generation. Retrying...")
            generation = self._agents["content_generator"].invoke(
                {"context": context, "task": step})
        total_time = time.time() - start_time

        print("Total time:", total_time)

        return {"action": ("content_generator", step, generation.content)}


    def data_generator(self, state: DecisionGraphState):
        print("---DATA GENERATOR---")
        data = state["data"]
        (task, _, _) = state["task"]
        (action, description, result) = state["action"]

        start_time = time.time()
        generation = None
        while not generation or not generation.data:
            if generation:
                print("Failed generation. Retrying...")
            generation = self._agents["data_generator"].invoke(
                {"data": data, "task": task, "action": action, "description": description, "result": result})
        total_time = time.time() - start_time

        print("Total time:", total_time)

        return {"data": generation.data}

    def plan_evaluator(self, state: DecisionGraphState):
        print("---EVALUATING PLAN---")
        context = state["context"]
        data = state["data"]
        (plan, step, completion, _) = state["plan"]
        (action, _, result) = state["action"]

        start_time = time.time()
        evaluator = None
        while not evaluator or not evaluator.completion:
            if evaluator:
                print("Failed generation. Retrying...")
            evaluator = self._agents["plan_evaluator"].invoke(
                {"context": context, "data": data, "plan": plan, "step": step, "result": result, "completion": completion})
        print("---GRADING PLAN---")
        grader = None
        while not grader or not grader.score:
            if grader:
                print("Failed generation. Retrying...")
            grader = self._agents["plan_grader"].invoke(
                {"plan": plan, "completion": evaluator.completion})
        total_time = time.time() - start_time

        print("Total time:", total_time)

        return {"plan": (plan, step, evaluator.completion, grader.score)}

    def subtask_evaluator(self, state: DecisionGraphState):
        print("---EVALUATING SUBTASK---")
        context = state["context"]
        data = state["data"]
        (task, task_completion) = state["subtask"]
        (plan, _, plan_completion, _) = state["plan"]

        start_time = time.time()
        evaluator = None
        while not evaluator or not evaluator.completion:
            if evaluator:
                print("Failed generation. Retrying...")
            evaluator = self._agents["task_evaluator"].invoke(
                {"data": data, "context": context, "task": task, "completion": task_completion, "progress": plan_completion})
        total_time = time.time() - start_time

        print("Total time:", total_time)

        return {"subtask": (task, evaluator.completion)}

    def answer_generator(self, state: DecisionGraphState):
        print("---GENERATING ANSWER---")
        query = state["query"]
        data = state["data"]
        (task, completion, _) = state["task"]

        start_time = time.time()
        generator = None
        while not generator or not generator.answer:
            if generator:
                print("Failed generation. Retrying...")
            generator = self._agents["answer_generator"].invoke({"query": query, "task": task, "completion": completion, "data": data})
        total_time = time.time() - start_time

        print("Total time:", total_time)

        return {"answer": generator.answer}

    @staticmethod
    def task_evaluation(state: DecisionGraphState):
        print("---CHECKING TASK COMPLETION---")
        (_, _, score) = state["task"]

        #if score < 0:
        #    return "abort"

        if score == "yes":
            return "complete"
        return "incomplete"

    @staticmethod
    def plan_evaluation(state: DecisionGraphState):
        print("---CHECKING PLAN COMPLETION---")
        (_, _, _, score) = state["plan"]

        #if score < 0:
        #    return "abort"

        if score == "yes":
            return "complete"
        return "incomplete"

    def step_evaluation(self, state: DecisionGraphState):
        print("---EVALUATING STEP---")
        context = state["context"]
        (_, step, _, _) = state["plan"]

        start_time = time.time()
        generator = None
        while not generator or not generator.tool or generator.tool not in ("action", "context", "generation"):
            if generator:
                print("Failed generation. Retrying...")
            generator = self._agents["step_evaluator"].invoke({"context": context, "task": step})
        total_time = time.time() - start_time

        print("Total time:", total_time)

        return generator.tool
