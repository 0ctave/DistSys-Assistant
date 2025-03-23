from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from rag.utils.llm import llm


def get_task_generator():
    class Task(BaseModel):
        """A generated task to answer a question."""

        task: str = Field(
            description="The generated task"
        )

    structured_llm_task_generator = llm().with_structured_output(Task)

    # Prompt
    system = """You are an assistant helping defining a task to answer a question. Use the given question to formulate a task to execute. \n 
        The task can be in multiple part and must address the entire question. \n
        The generated task will be used by an IA assistant to answer the question and execute the necessary actions, make sure it is accurate and require as least external intervention as possible. \n
        Don't give commands to execute or code, just a general task that addresses the query. \n
        Context information can be retrieved using system information or commands on the host.
        You have to give the task that will be solved by other agent, just reformulate so it's understandable and clear for further agents.\n"""
    generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Question: \n {query} \n\n\n"),
        ]
    )

    return generator_prompt | structured_llm_task_generator


def get_system_context_query_generator():
    class ContextQuery(BaseModel):
        """A generated task to answer a question."""

        context_query: str = Field(
            description="The generated context query"
        )

    structured_llm_task_generator = llm().with_structured_output(ContextQuery)

    # Prompt
    system = """You are an assistant generating a list of key words, a "context query", this list will be used to query system information that would be useful to solve a given task. \n
        You can request to access data like folder structure, specific files existence, os information, installed programs, system configuration, etc. \n 
        For exemple if you need a specific program to execute the task, you can answer you need the installed programs and version. \n
        The generated context query will be used by a vector database for Retrieval Augmented Generation purpose to gather context information to solve the task. \n
        Always ask for all the basic system information, and add more precise request if needed. \n"""
    generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Task: \n {task} \n\n\n"),
        ]
    )

    return generator_prompt | structured_llm_task_generator

def get_subtask_generator():
    class Task(BaseModel):
        """A refined task."""

        task: str = Field(
            description="The generated subtask"
        )

    structured_llm_task_refiner = llm().with_structured_output(Task)

    # Prompt
    system = """You are an assistant generating a subtask that address a part of a general task. \n 
        You have access to the task, the last subtask that you generated and how it went. \n
        The score is the level of completion of the task, use it to refine the subtask, it's an integer from 0 to 10, 0 nothing has been done to solve the task, 10 the task has been fully addressed. \n
        The subtask that you generate will be used to make and action plan to solve this subtask and advance solving the general task. \n
        If something is already done or existing, do not repeat it or recreate it. \n
        Make sure the subtask is grounded to the information you have.\n

        Context : {context}"""
    generator_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Task: \n {task} \n\n\n SubTask: \n {subtask} \n\n\n Completion: \n {completion}"),
        ]
    )

    return generator_prompt | structured_llm_task_refiner


def get_plan_generator():
    class Plan(BaseModel):
        """A generated plan to address a given task."""

        plan: str = Field(
            description="The generated plan"
        )

    structured_llm_plan_generator = llm().with_structured_output(Plan)

    # Prompt
    system = """You are an action planner, you have to define the action to do or the information to retrieve to answer a given task. \n 
        Give the plan to solve this task in the given context, make sure it is accurate and grounded to the information you have access to. \n
    
        Each generated step must be tested : \n
            - Check the presence of a program before using them. \n
            - Check the presence of a file/folder before creating it (use it if it already exists). \n
            - Check the presence of a file/folder after creating it. \n
            - Check the content of a file before editing it. \n
            - Check the manual of commands before use with "man {{command}}". \n
        Do it for every action where it is necessary. \n
        Analyse the file structure if you need to. \n
        Don't generate steps that need external intervention to be executed, never use a text editor like nano or any program that might need external intervention to proceed. \n
        Don't generate steps if the context and retrieved data indicates that the goal of the steps is covered. \n 
        Don't give the commands to execute, just give a detailed plan in one or multiple steps to answer the task. \n
        Make sure every step is numbered, and contain a short explanation of the step's goal. \n
        If you need to move in a folder, say that you need to. \n
        Never include opening a terminal or a shell, there is already one open. \n
                
        Answer the detailed plan. \n
        
        Context: \n {context}"""
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Task: {task} \n\n\n Retrieved data : \n {data} \n"),
        ]
    )

    return planner_prompt | structured_llm_plan_generator


def get_step_generator():
    class Step(BaseModel):
        """A generated step to follow a plan."""

        step: str = Field(
            description="The next step to address the plan"
        )

    structured_llm_step_generator = llm().with_structured_output(Step)

    # Prompt
    system = """You are a step planner, you have to define the next step of the plan to do. \n 
        Give the next step to solve the task. \n
        Use the current plan, it's completion and the given data and context to find the next most logical step to do. \n
        Make sure your step is basic and simple, don't give any explanation. \n
        The step must only be a unique simple action. \n
        Don't include anything else than the most appropriate action to do. \n
        If the last step was unsuccessful, try to find the best step to correct it and solve the task. \n
        If the step you plan to do is already covered in the completion or the data, skip it and generate a new one following the plan. \n
        Don't give any commands to execute, just describe the next step in 2 line maximum. \n
        Always prefer using file or directory that already exist instead of recreating them. \n
        
        For example, don't delete a folder if it already exists. \n

        Current execution path : {path}\n
        Context: \n{context}"""
    step_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Task : \n {task} \n\n\n Plan : \n {plan} \n\n\n Plan Completion : \n {completion} \n\n\n Last generated step: \n {step} \n\n\n Data : {data}"),
        ]
    )

    return step_prompt | structured_llm_step_generator


def get_step_evaluator():
    class TaskEvaluation(BaseModel):
        """Completion of the task."""

        tool: str = Field(
            description="The step tool, 'action', 'context' or 'generation'"
        )

    structured_llm_evaluator = llm().with_structured_output(TaskEvaluation)

    # Prompt
    system = """You are an evaluator assessing what is the best tool to use for the task. \n 
            Give the correct tool to use either : \n
             - 'action' if the step requires interaction with the system, such as modifying files or executing commands to solve the given task. \n
             - 'context' if additional information is required from the available context information, the context data already summarized the available context data, use it if you need precisions. \n
             - 'generation' if the step requires generating a text to be used later, never use it to execute an action. \n 
            Only answer with one of the 3 possibles tools 'action', 'context' or 'generation', answer only the tool name with any formatting or additional text \n 

            Context : \n{context}"""
    plan_completion_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Task : {task}"),
        ]
    )

    return plan_completion_prompt | structured_llm_evaluator

def get_task_evaluator():
    class TaskEvaluation(BaseModel):
        """Completion of the task."""

        completion: str = Field(
            description="The summary of what has been completed of the task"
        )

    structured_llm_evaluator = llm().with_structured_output(TaskEvaluation)

    # Prompt
    system = """You are an evaluator assessing what has been completed in a task. \n 
         Give a summary of what has been completed in the task. \n
         Your summary should detail everything that has been done previously whether it has worked out or not. \n
         You have the task you are evaluation, as well as the prior completion status and the progress that has been made. \n
         You also have access to datas that have been fetched to help solve the task. \n 
         Add comments in your summary to help direct the next step of the task. \n
         If the task has not been addressed at all, answer "Nothing has been done yet". \n

         Context : \n{context}"""
    plan_completion_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Task : {task} \n\n\n Prior completion status : \n {completion} \n\n\n Completion Progress : \n {progress} \n\n\n Useful data : {data}"),
        ]
    )

    return plan_completion_prompt | structured_llm_evaluator


def get_task_grader():
    class TaskGrade(BaseModel):
        """Completion score to assess the task level of completion."""

        score: str = Field(description="Task completion score 'yes' or 'no'")


    structured_llm_evaluator = llm().with_structured_output(TaskGrade)

    # Prompt
    system = """You are a grader assessing whether a given task is fully complete or not. \n 
            Give a binary score 'yes' or 'no' score to indicate whether the task has been fully completed according to the task completion summary. \n
            Context : \n{context}"""
    plan_completion_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Task : \n {task} \n\n\n Task completion summary : \n {completion}"),
        ]
    )

    return plan_completion_prompt | structured_llm_evaluator


def get_content_generator():
    class ContentGeneration(BaseModel):
        """Completion score to assess the task level of completion."""

        content: str = Field(description="The generated content")

    structured_llm_content = llm().with_structured_output(ContentGeneration)

    # Prompt
    system = """You are an assistant generating content. \n 
            Give generated content by following the task and using the context information. \n
            Make sure the length of your generation is not too long, and is not too short. \n

            Context : \n{context}"""
    content_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Task : \n {task}"),
        ]
    )

    return content_prompt | structured_llm_content


def get_data_generator():
    class Data(BaseModel):
        """Completion score to assess the task level of completion."""

        data: str = Field(description="The extracted data")


    structured_llm_context = llm().with_structured_output(Data)

    # Prompt
    system = """You are an assistant extracting data from the result of an action. \n 
            Give a summary of the useful data to solve the given task, combining previous data with your new information. \n
            You have the action that has just been performed and the result of it. \n
            Don't directly add the action result to the context or don't describe the command, just add the new pieces of information useful for the task. \n
            Use every piece of information that help solving the task from the action result, extract all information that can be useful. \n
            Only extract the data from the 'Action result', the rest is only to make you understand the context. \n
            Give for each information you extract, a few words on the action. \n
            For exemple "moved in {{folder}}", "created {{thing}}", "checked {{something}} and {{result}}". \n
            Never give instructions, or the next thing to do. \n
            Also tell shortly if something wrong happened but remove any error that are now fixed. \n
            If the action is a 'content_generator', include it's content in the data. \n
            Make sure you are precise and concise. \n
            
            Current data : \n{data}"""
    context_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Task : \n {task} \n\n\n Action : \n {action} \n\n\n Description : \n {description} \n\n\n Action result : \n {result}"),
        ]
    )

    return context_prompt | structured_llm_context

def get_plan_evaluator():
    class PlanEvaluation(BaseModel):
        """Completion score to assess if the plan level of completion."""

        completion: str = Field(
            description="The summary of what has been completed in the plan"
        )

    structured_llm_evaluator = llm().with_structured_output(PlanEvaluation)

    # Prompt
    system = """You are an evaluator assessing what as been completed in the plan. \n 
         You have access to the plan, what has been completed in it as well as the current step of the plan we are addressing. \n
         From the current step that was just addressed and its action result, generate a new summary of completion. \n
         The generated plan completion summary should detail everything that has been done previously whether it has worked out or not. \n
         But if the status of a step of the plan has changed, update it in your summary. \n
         The summary must be an explanation of what has been done in the plan, with commentaries on it. \n
         Combine the last summary of what has been completed with the new summary of completion. \n
         If the task has not been addressed at all, answer "Nothing has been done yet".\n
         if a step is not necessary anymore (because it's already covered by the previous action), specify it. \n
         
         Plan data : \n{data}\n\n
         Context : \n{context}"""

    plan_completion_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Plan : \n {plan} \n\n\n Plan Completion: \n {completion} \n\n\n Current Step: {step} \n\n\n Action result: \n {result}"),
        ]
    )

    return plan_completion_prompt | structured_llm_evaluator


def get_plan_grader():
    class PlanGrade(BaseModel):
        """Completion score to assess if the plan level of completion."""

        score: str = Field(description="Plan completion score 'yes' or 'no'")


    structured_llm_evaluator = llm().with_structured_output(PlanGrade)

    # Prompt
    system = """You are a grader assessing whether a given plan is fully complete or not. \n 
            Give a binary score 'yes' or 'no' score to indicate whether the plan has been fully completed according to the plan completion summary. \n"""
    plan_completion_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Plan : \n {plan} \n\n\n Plan completion summary: \n {completion}"),
        ]
    )

    return plan_completion_prompt | structured_llm_evaluator


def get_answer_generator():
    class Answer(BaseModel):
        """Analysis of the command result"""

        answer: str = Field(
            description="The answer to the initial query"
        )

    structured_llm_answer = llm().with_structured_output(Answer)

    # Prompt
    system = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."""
    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human",
             "Query: {query} \n\n\n Task: {task} \n\n\n Completion: {completion} \n\n\n Data : {data}"),
        ]
    )

    return answer_prompt | structured_llm_answer
