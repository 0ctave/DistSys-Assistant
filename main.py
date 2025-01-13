import timeit
import argparse
from enum import Enum

from llama_index import PromptTemplate

from rag.pipeline import build_rag_pipeline

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


class Role(Enum):
    CMD = """
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, \
    provide only {shell} commands for {os} without any description.
    If there is a lack of details, provide most logical solution.
    Ensure that all solution can be achieved only with commands, \
    if you need to edit a file pipe content in it \
    and make sure to not open a text editor just create the file and folders if you need it.
    Ensure the output is a valid shell command.
    If multiple steps required try to combine them together using &&.
    Provide only plain text without Markdown formatting.
    Do not provide markdown formatting such as ```.
    Please provide your answer in the form of a structured JSON format containing \
    an array named commands with the list of commands to execute.
    
    Query: {query_str}
    Answer: \
    """
    TASKER = """
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, \
    provide the list of tasks to execute and their respective queries.
    A task is an ACTION that takes a query as input \
    and outputs data in the form of a structured JSON format containing.
    Tasks are chained together to really answer the query of the user you received.
    If there is a lack of details, provide most logical solution.
    The list of the 3 ACTIONs is :
      - CMD : this action takes a simple QUERY as input and can execute commands \
      on the user machine to answer the task and provide more contextual information.
      - THINK : this action takes a complex QUERY as input returns the list of ACTIONs \
      to answer the query and provide more contextual information. \
      It can be used to answer questions on the context information we have, \
      this is useful to know if you have to do a CMD action \
      to retrieve more contextual information or if it is already there
      - ANSWER : this action takes the complete context necessary to answer the user query \
      and returns the most coherent textual answer according to the query.

    Ensure your answer is a valid structured JSON.
    If multiple steps are required try use the ACTIONs \
    in hand to plan the most logical chain of thoughts.
    Do not provide markdown formatting such as ```.
    Please provide your answer in the form of a structured JSON format containing \
    an array of json components with the one word name of the task as key \
    and containing for each the ACTION type, and the QUERY to input to this action.
    This is an example of how you would solve the query 'What is my username ?'
    
      '1' 
        'type' 'CMD',
        'query' "Provide the command to get my username.'
      ,
      '2' 
        'type' 'ANSWER',
        'query' 'Give an answer to the question what is my username with the given context.'
      
    
    
    Query: {query_str}
    Answer: \
    """
    INFO = "INFO"

def get_os(**kwargs):
    return "debian"

def get_shell(**kwargs):
    return "bash"

def load_role(role: Role, rag_chain):
    rag_prompt_tmpl = PromptTemplate(
        role.value,
        function_mappings={"shell": get_shell,
                           "os": get_os,},
    )

    rag_chain.update_prompts(
        {"response_synthesizer:text_qa_template": rag_prompt_tmpl}
    )
    return rag_chain


def get_rag_response(query, rag_chain):
    result = rag_chain.query(query)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input',
                        type=str,
                        default='What is the invoice number value?',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()

    start = timeit.default_timer()

    rag_chain = build_rag_pipeline()

    rag_chain = load_role(Role.CMD, rag_chain)

    print('Retrieving answer...')
    answer = get_rag_response(args.input, rag_chain)
    answer = str(answer).strip()

    end = timeit.default_timer()

    print(f'\nAnswer:\n{answer}')
    print('=' * 50)

    print(f"Time to retrieve answer: {end - start}")
