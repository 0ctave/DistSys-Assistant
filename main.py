import timeit
import argparse

from llama_index import PromptTemplate

from rag.pipeline import build_rag_pipeline

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def get_rag_response(query, rag_chain):
    new_summary_tmpl_str = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Provide only Bash commands for Debian without any description."
        "If there is a lack of details, provide most logical solution.\n"
        "Ensure the output is a valid shell command.\n"
        "If multiple steps required try to combine them together using &&.\n"
        "Provide only plain text without Markdown formatting.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    new_summary_tmpl = PromptTemplate(new_summary_tmpl_str)
    rag_chain.update_prompts(
        {"response_synthesizer:summary_template": new_summary_tmpl}
    )

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

    print('Retrieving answer...')
    answer = get_rag_response(args.input, rag_chain)
    answer = str(answer).strip()

    end = timeit.default_timer()

    print(f'\nAnswer:\n{answer}')
    print('=' * 50)

    print(f"Time to retrieve answer: {end - start}")