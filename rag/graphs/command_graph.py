import sys
import time
from typing import TypedDict, Tuple, List

from langgraph.constants import START, END

from rag.agents.command_agent import *
from rag.graphs.graph_base import GraphBase

class CommandGraph(GraphBase):
    class InputState(TypedDict):
        context: str
        task: str

    class OutputState(TypedDict):
        action: str
        description: str
        result: str

    class CommandGraphState(TypedDict):
        context: str
        task: str

        command: str
        description: str

        correctness: Tuple[str, str]
        security: Tuple[str, str]

        approved: int

        chunks: List[str]

    def __init__(self, assistant):
        super().__init__(assistant, self.CommandGraphState, input_state=self.InputState, output_state=self.OutputState)
        self._command_executor = assistant.get_command_executor()
        self._path = self._command_executor.run_command("pwd")

    def get_path(self):
        return self._path

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
            "description_generator": get_description_generator(),
            "command_generator": get_command_generator(),
            "correctness_evaluator": get_correctness_evaluator(),
            "correctness_grader": get_correctness_grader(),
            "security_evaluator": get_security_evaluator(),
            "security_grader": get_security_grader(),

            "executor": self._assistant.get_command_executor(),
            "result_analyser": get_result_analyser(),
        }

    def _load_nodes(self, builder) -> None:
        """
        Adds nodes to the given builder in a specific sequence to set up the necessary
        pipeline for executing tasks. Each node represents a different functionality
        essential for the process, such as generating descriptions and commands,
        grading correctness and security, managing approvals, execution, and analyzing
        results.

        Args:
            builder: The object that manages and constructs the pipeline of nodes.
        """
        builder.add_node("description_generator", self.description_generator)
        builder.add_node("command_generator", self.command_generator)

        builder.add_node("correctness_evaluator", self.correctness_evaluator)
        builder.add_node("security_evaluator", self.security_evaluator)
        builder.add_node("abort", self.abort)

        builder.add_node("approval", self.approval)

        builder.add_node("executor", self.executor)

        builder.add_node("result_analyser", self.result_analyser)

    def _load_edges(self, builder) -> None:
        """
        Builds the directed graph by defining edges and conditional edges between various nodes
        to represent a workflow. The graph follows a specific path determined by conditional
        evaluations at certain stages. It ensures logical flow through nodes like generators,
        graders, approval, and execution, culminating in final nodes like result analysis or END.

        Args:
            builder: An object responsible for constructing the graph by adding edges and
                conditional edges between nodes, representing states and their transitions.
        """
        builder.add_edge(START, "description_generator")
        builder.add_edge("description_generator", "command_generator")
        builder.add_edge("command_generator", "correctness_evaluator")

        builder.add_conditional_edges(
            "correctness_evaluator",
            self.correctness_evaluation,
            {
                "incorrect": "command_generator",
                "correct": "security_evaluator",
            })

        builder.add_conditional_edges(
            "security_evaluator",
            self.security_evaluation,
            {
                "abort": "abort",
                "approve": "approval",
                "execute": "executor",
            })

        builder.add_conditional_edges(
            "approval",
            self.approval_evaluation,
            {
                "abort": "abort",
                "execute": "executor",
            })
        builder.add_edge("abort", END)

        builder.add_edge("executor", "result_analyser")
        builder.add_edge("result_analyser", END)

    def description_generator(self, state: CommandGraphState):
        """
        Generates a command description based on the given state using an agent.

        This function utilizes a description generation agent to produce a description
        for a specific task extracted from the given state. It calculates and logs
        the time taken for the generation process.

        Args:
            state (CommandGraphState): The state containing the task and context
                information required by the description generator agent.

        Returns:
            dict: A dictionary containing the generated description under the
                key 'description'.
        """
        print("     ---GENERATING COMMAND DESCRIPTION---")
        task = state["task"]
        context = state["context"]

        start_time = time.time()
        generation = None
        while not generation or not generation.description:
            if generation:
                print("Failed generation. Retrying...")
            generation = self._agents["description_generator"].invoke({"context": context, "task": task, "path": self._path})
        total_time = time.time() - start_time

        print("     Total time:", total_time)

        return {"task": task, "description": generation.description}

    def command_generator(self, state: CommandGraphState):

        """
        Generates the next command based on the provided state. Uses an external
        agent specified by the key "command_generator" to generate the command. The
        context, task, and description from the state are passed to the agent to
        produce a relevant command. Additionally, measures the time taken to
        generate the command and outputs relevant logging information.

        Args:
            state (CommandGraphState): A dictionary containing the current state,
                including "task", "context", and "description" keys required for
                command generation.

        Returns:
            dict: A dictionary containing the generated command under the key
                "command".
        """
        print("     ---GENERATING NEXT COMMAND---")
        task = state["task"]
        context = state["context"]
        description = state["description"]
        (correctness, _) = state.get("correctness", ("None", "no"))

        start_time = time.time()
        generation = None
        while not generation or not generation.command:
            if generation:
                print("Failed generation. Retrying...")
            generation = self._agents["command_generator"].invoke(
                {"context": context, "task": task, "description": description, "correctness": correctness, "path": self._path})
        total_time = time.time() - start_time

        print("     Total time:", total_time)
        print("     The found command:", generation.command)

        return {"command": generation.command}

    def correctness_evaluator(self, state: CommandGraphState):
        """
        Evaluates the correctness of a given task based on the provided task description, command, and
        context within the state.
g
        The method interacts with an agent designated as the correctness grader to analyze and grade
        the correctness of a task's components. This process measures the command's compliance with the
        task requirements and provides a correctness level score.

        Args:
            state (CommandGraphState): A dictionary-like object that includes:
                - task: The task to be assessed.
                - context: The contextual information surrounding the task.
                - command: The command to evaluate for correctness.
                - description: An explanation or additional details about the task.

        Returns:
            dict: A dictionary containing:
                - correctness (float): A score indicating the overall correctness of the evaluated task.
                - correctness_level (str): A qualitative level of correctness determined by the agent.
        """
        print("     ---EVALUATING THE CORRECTNESS---")
        task = state["task"]
        context = state["context"]
        command = state["command"]

        start_time = time.time()
        evaluator = None
        while not evaluator or not evaluator.comment:
            if evaluator:
                print("Failed generation. Retrying...")
            evaluator = self._agents["correctness_evaluator"].invoke(
                {"context": context, "task": task, "command": command})
        print("     ---GRADING THE CORRECTNESS---")
        grader = None
        while not grader or not grader.score:
            if grader:
                print("Failed generation. Retrying...")
            grader = self._agents["correctness_grader"].invoke(
                {"context": context, "command": command, "correctness": evaluator.comment})
        total_time = time.time() - start_time

        print("     Total time:", total_time)
        print("     Correctness:", grader.score)

        return {"correctness": (evaluator.comment, grader.score)}

    def security_evaluator(self, state: CommandGraphState):
        """
        Evaluates the security impact of a command using information from the given
        state. This function utilizes the `security_evaluator` agent to assess the
        security level of the provided command and its context.

        Args:
            state (CommandGraphState): The state containing the task, context,
                command, and description necessary for security grading.

        Returns:
            dict: A dictionary containing the evaluated security information:
                - "security": The overall security grade.
                - "security_level": The assessed security level.

        """
        print("     ---EVALUATING THE SECURITY---")
        context = state["context"]
        command = state["command"]

        start_time = time.time()
        evaluator = None
        while not evaluator or not evaluator.security:
            if evaluator:
                print("Failed generation. Retrying...")
            evaluator = self._agents["security_evaluator"].invoke(
                {"context": context, "command": command})
        print("     ---GRADING THE SECURITY---")
        grader = None
        while not grader or not grader.score:
            if grader:
                print("Failed generation. Retrying...")
            grader = self._agents["security_grader"].invoke(
                {"context": context, "command": command, "security": evaluator.security})
        total_time = time.time() - start_time

        print("     Total time:", total_time)
        print("     Security:", grader.score)

        return {"security": (evaluator.security, grader.score)}

    @staticmethod
    def abort(state: CommandGraphState):
        command = state["command"]
        description = state["description"]

        return {"action": command, "description": description, "result": "Unsafe command aborted execution !"}



    @staticmethod
    def approval(state: CommandGraphState):
        """
        Requests user approval for executing a command.
        
        This function prompts the user to approve or reject the execution of the command by
        displaying relevant information from the state. The user's decision is then returned.
        
        Args:
            state (CommandGraphState): The current state containing details such as task, 
            context, command, and description.
        
        Returns:
            str: The user's decision:
                - "executor" if the user approves the execution.
                - "abort" if the user rejects the execution.
        """
        print("--- USER APPROVAL NEEDED ---")
        print(f"Task: {state['task']}")
        print(f"Context: {state['context']}")
        print(f"Command: {state['command']}")
        print(f"Description: {state['description']}")

        while True:
            decision = input("Approve execution? (yes/no): ").strip().lower()
            if decision in {"yes", "y"}:
                return {"approved": 1}
            elif decision in {"no", "n"}:
                return {"approved": 0}
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

    def executor(self, state: CommandGraphState):
        """
        Executes a command using the provided execution agent and returns the result.

        This method is responsible for executing a command encapsulated in the
        `CommandGraphState`. It measures the time taken to execute the command by
        retrieving and invoking the respective executor from the predefined set of
        agents. The execution metrics, including the total elapsed time, are printed
        for transparency.

        Args:
            state (CommandGraphState): The state containing the command to be executed.
                This should be a dictionary-like object wherein the key "command"
                maps to the command to be run.

        Returns:
            dict: A dictionary containing the result of the execution, under the key
                "result".
        """
        print("     ---EXECUTING THE COMMAND---")
        command = state["command"]

        start_time = time.time()
        result = self._agents["executor"].run_command(command)
        chunks = self.chunk_output(result)

        self._path = self._agents["executor"].run_command("pwd")
        total_time = time.time() - start_time
        print("     Total time:", total_time)

        return {"chunks": chunks}

    def result_analyser(self, state: CommandGraphState):
        """
        Analyzes the result of a command execution within a specified context.

        This method uses the `result_analyser` agent to process the provided state information
        and computes the completion level of the task. It measures and displays the total
        execution time for the analysis.

        Args:
            state (CommandGraphState): A dictionary-like state object containing the execution
                context, the command itself, a description of the command, and its result.

        Returns:
            dict: A dictionary containing the command analysis result generated by the
                `result_analyser` agent.
        """
        print("     ---ANALYSING THE RESULT---")
        context = state["context"]
        task = state["task"]
        command = state["command"]
        description = state["description"]
        chunks = state["chunks"]

        analysis_parts = []

        analysis_start_time = time.time()

        for i, chunk in enumerate(chunks):
            if i > 20:
                print(f"     Logs are too long for complete analysis.")
                print(f"     Stopping logs analysis at 20 chunks.")

                analysis_parts.append("The analysis stopped. \n "
                                      "The data is too long for complete analysis.\n"
                                      "Analysed {}% of the result.\n".format((i + 1) / len(chunks) * 100))
                break
            print(f"     Analyzing chunk {i + 1}/{len(chunks)}...")
            start_time = time.time()
            analyser = None
            while not analyser or not analyser.analysis:
                if analyser:
                    print("Failed generation. Retrying...")
                analyser = self._agents["result_analyser"].invoke({
                    "task": task,
                    "context": context,
                    "command": command,
                    "description": description,
                    "result": chunk
                })
            total_time = time.time() - start_time
            print(f"     Chunk {i + 1} analyzed in {total_time:.2f}s")
            analysis_parts.append(analyser.analysis)

        # Merge or summarize the full analysis
        full_analysis = "\n\n".join(analysis_parts)
        print("     Total time:", time.time() - analysis_start_time)
        return {"action": command, "description": description, "result": full_analysis}

    @staticmethod
    def correctness_evaluation(state: CommandGraphState):
        """
        Evaluates the correctness level of a given state and determines if regeneration
        is required based on the evaluation level.

        Args:
            state (CommandGraphState): The state containing the correctness value to
            be evaluated.

        Returns:
            str: Returns "regenerate" if the correctness level is not equal to 10;
            otherwise, returns "correct".
        """
        (_, score) = state["correctness"]

        if score == "no":
            return "incorrect"
        return "correct"

    @staticmethod
    def security_evaluation(state: CommandGraphState):
        """
        Evaluates the security level of a given command execution state and determines the
        appropriate execution role based on the security level. This function assesses the
        security of requested commands, comparing it against a defined threshold, and returns
        an execution role based on the evaluation.

        Args:
            state (CommandGraphState): A dictionary-like object containing the current execution
                state. It includes keys such as "task", "command", and "description", which provide
                details about the task being executed, the command to evaluate, and an optional
                description of the command.

        Returns:
            str: A string indicating the execution role to be assigned based on the evaluated
                security level. Possible values are:
                - "generator": For commands with a security level below the defined threshold.
                - "approval": If the security level is inadequate but not extremely low.
                - "executor": For commands that meet or exceed the security threshold.
        """
        (_, score) = state["security"]

        if score == "no":
                return "abort"
        if score == "approval":
            return "approve"
        return "execute"

    @staticmethod
    def approval_evaluation(state: CommandGraphState):
        approval = state["approved"]

        if approval:
            return "execute"
        return "abort"

    @staticmethod
    def chunk_output(output: str, max_chars: int = 3000) -> list[str]:
        """Splits the output string into chunks of max_chars."""
        lines = output.splitlines()
        chunks = []
        current_chunk = []
        current_len = 0

        for line in lines:
            if current_len + len(line) + 1 > max_chars:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            current_chunk.append(line)
            current_len += len(line) + 1  # +1 for newline

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return chunks
