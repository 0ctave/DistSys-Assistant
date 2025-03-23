from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag.utils.llm import llm


def get_description_generator():
    class CommandDescription(BaseModel):
        """A description of a command to be executed."""

        description: str = Field(
            description="The generated command description"
        )

    structured_llm_command_generator = llm().with_structured_output(CommandDescription)
    # Prompt
    system = """You are a command description generator, you have to describe the command that should be executed to answer a specific task. \n 
        Only answer with a command description and not a command to be executed or the exact content of the command, just describe the command that should be run. \n
        Your description will be used to generate the most appropriate command or combination of commands to answer the task in the specified context. \n
        Ensure the description you generate is precise, grounded to the facts given the context that has been given. \n
        Never use a text editor (nano, vi etc..) or a command line util that need further input (except passwords). \n
        To edit a file pipe the content directly inside. \n
        Don't give command examples just the description. \n
        You have the current execution path the command will be executed in. \n
        
        
        The command execution context {context}"""
    generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "The task you have to answer is : \n {task} \n\n\n Current execution path : {path}"),
        ]
    )

    return generator_prompt | structured_llm_command_generator


def get_command_generator():
    class Command(BaseModel):
        """A command to be executed."""

        command: str = Field(
            description="The generated command"
        )

    structured_llm_command_generator = llm().with_structured_output(Command)
    # Prompt
    system = """You are a command generator, your only task is to generate a command to address a specific task. \n 
        Give a syntactically correct command that fit the given description. \n
        Make sure that the command can run and is useful in advancing the task. \n
        Make sure your command includes all the necessary command arguments to do your task properly. \n
        You can use any commands you know you have access to. \n
        Make sure that the command you generate doesn't block the execution flow. \n
        Except for password inputs you have to make sure the command doesn't need an external intervention to continue or exit. \n
        If you have correctness comments, take them into account to generate a correct command. \n
        Some commands return nothing, try to avoid them as the output of the command will be analysed. \n
        Make sure the command works in the current path of execution. \n
        Use command that return an output if possible. \n
        Use commands in detached mode if possible. \n

        System context: {context}"""
    generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "The task : \n {task} \n\n Description : {description} \n Current path of execution : {path}\n\n\n Correctness comments : {correctness}"),
        ]
    )

    return generator_prompt | structured_llm_command_generator


def get_correctness_evaluator():
    class Correctness(BaseModel):
        """A description of a command to be executed."""

        comment: str = Field(
            description="The generated comments on the correctness of the command"
        )

    structured_llm_grader = llm().with_structured_output(Correctness)

    # Prompt
    system = """You are a evaluator commenting whether a command is syntactically correct on the system configuration\n 
         Give a comment on the correctness of this command of maximum 2 lines. \n
         You have to find if the given command it is syntactically correct, if the command works in this context and has good chances to run. \n
         If the command will require an input to quit or execute, it's not correct. \n
         Always answer with your comments on about the correctness of the command. \n
         
         Is the command syntactically correct? \n
         Is the command correct for the given task? \n
         You can add ideas in the comment the correct command if the command is incorrect. \n

         System context: \n{context}"""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Task to solve : {task} \n\n\n Generated command : \n {command}"),
        ]
    )

    return answer_prompt | structured_llm_grader


def get_correctness_grader():
    class Correctness(BaseModel):
        """A description of a command to be executed."""

        score: str = Field(
            description="Correctness score of the command 'yes' or 'no'"
        )

    structured_llm_grader = llm().with_structured_output(Correctness)

    # Prompt
    system = """You are a grader assessing whether a command is syntactically correct on the system configuration\n 
         Give a binary "score" 'yes' or 'no' to indicate whether the given command is syntactically correct. \n
         You have access comments on the correctness of the command. \n
         
         System context: \n{context}"""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Generated command : \n {command} \n\n\n Correctness comments : \n {correctness}"),
        ]
    )

    return answer_prompt | structured_llm_grader


def get_security_evaluator():
    class Security(BaseModel):
        """A description of a command to be executed."""

        security: str = Field(
            description="The generated comments on the correctness of the command"
        )

    structured_llm_grader = llm().with_structured_output(Security)

    # Prompt
    system = """You are a evaluator commenting whether a command is safe to run on the system configuration\n 
             Give a comment on the security of this command of maximum 3 lines. \n
             If you really need to use a command that is not fully safe, explain why you need it and the reason. \n
             Using "super user" privileges in your command doesn't necessarily mean that the command is unsafe. \n
             Unsafe commands are those that modify crucial files or directories, or that might break the system. \n
             You can still do filesystem operations in the home directory or the user. \n
             Checking the existence of a file or folder as well as reading system configuration files are considered safe operations. \n

         System context: \n{context}"""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Generated command : \n {command} \n\n\n"),
        ]
    )

    return answer_prompt | structured_llm_grader


def get_security_grader():
    class Security(BaseModel):
        """A description of a command to be executed."""

        score: str = Field(
            description="Security score of the command 'yes', 'no' or 'approval"
        )

    structured_llm_grader = llm().with_structured_output(Security)

    system = """You are a grader assessing whether a given task is fully complete or not. \n 
            Give a ternary score 'yes', 'no' or 'approval' to indicate whether the given command is safe to run. \n

            Context : \n{context}"""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Generated command : \n {command} \n\n\n Security concerns : \n {security}"),
        ]
    )

    return answer_prompt | structured_llm_grader


def get_result_analyser():
    class AnalyseCommandResult(BaseModel):
        """Analysis of the command result"""

        analysis: str = Field(
            description="The command result analysis"
        )

    structured_llm_analyser = llm().with_structured_output(AnalyseCommandResult)

    system = """You are an evaluator analysing what as happened during a command execution. \n 
         Give an analysis of the command output that is useful to solve the given task. \n
         You have access to the commands, its description and the task the command is trying to solve. \n
         Stay grounded to the output and extract all the relevant information. \n
         If the command ran properly, give only the information that can be useful for the task. \n
         If the command didn't ran properly, give an explanation and your analysis to prevent this error from happening. \n
         Your command analysis must be as concise as possible. \n
         You might not have the full command output, just analyse the command output you have an extract all the relevant data. \n
         Don't talk about the command, just answer a precise and short analysis. \n
         If there was no output says so, but it doesn't necessarily mean that the command didn't worked. \n
         A lot of commands don't give output, but usually it give error outputs. \n
         
         Context : \n{context}"""

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Task: {task} \n\n\n Command : \n {command} \n\n description: \n {description} \n\n Command output: \n {result}"),
        ]
    )

    return answer_prompt | structured_llm_analyser