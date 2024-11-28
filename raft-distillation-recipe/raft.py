import argparse
import json
import logging
import random
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import ceil
from pathlib import Path
from threading import Event
from typing import Literal, get_args

import datasets
import pyarrow as pa
import PyPDF2
from datasets import Dataset
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from mdc import MDC
from openai import BadRequestError
from openai.types.chat.chat_completion import ChatCompletion
from tqdm import tqdm

from checkpointing import Checkpointing, checkpointed
from client_utils import (
    ChatCompleter,
    UsageStats,
    build_langchain_embeddings,
    build_openai_client,
)
from format import DatasetConverter, dataset_formats, output_dataset_types
from logconf import log_setup

log_setup()

load_dotenv()

logger = logging.getLogger("raft")

DocType = Literal["api", "pdf", "json", "txt"]
doc_types = list(get_args(DocType))

SystemPromptKey = Literal["gpt", "llama"]
system_prompt_keys = list(get_args(SystemPromptKey))


def parse_args() -> argparse.Namespace:
    """
    Parses and returns the arguments specified by the user's command.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--datapath",
        type=Path,
        default="",
        help="If a file, this is the path at which the document is located. If a folder, this is the path at which to load all documents",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./",
        help="The path at which to save the dataset to",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default="hf",
        help="The format of the output dataset",
        choices=dataset_formats,
    )
    parser.add_argument(
        "--output-type",
        type=str,
        default="jsonl",
        help="The output type to export the dataset to",
        choices=output_dataset_types,
    )
    parser.add_argument(
        "--output-chat-system-prompt",
        type=str,
        help="The system prompt to use when the output format is chat",
    )
    parser.add_argument(
        "--output-completion-prompt-column",
        type=str,
        default="prompt",
        help="The prompt column name to use for the completion format",
    )
    parser.add_argument(
        "--output-completion-completion-column",
        type=str,
        default="completion",
        help="The completion column name to use for the completion format",
    )
    parser.add_argument(
        "--distractors",
        type=int,
        default=3,
        help="The number of distractor documents to include per data point / triplet",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=1.0,
        help="The probability that the oracle document is included in the context",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=5,
        help="The number of data points / triplets to generate per chunk",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="The size of each chunk in terms of the number of tokens",
    )
    parser.add_argument(
        "--doc-type",
        type=str,
        default="pdf",
        help="The type of the document, must be one of the accepted document types",
        choices=doc_types,
    )
    parser.add_argument(
        "--openai-key",
        type=str,
        default=None,
        help="Your OpenAI API key to make queries to GPT-3.5 or GPT-4",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-ada-002",
        help="The embedding model to use to encode the documents chunks (e.g. text-embedding-ada-002, ...)",
    )
    parser.add_argument(
        "--completion-model",
        type=str,
        default="gpt-4",
        help="The model to use to generate questions and answers (e.g. gpt-3.5, gpt-4, ...)",
    )
    parser.add_argument(
        "--system-prompt-key",
        type=str,
        default="gpt",
        help="The system prompt to use to generate the dataset",
        choices=system_prompt_keys,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="The number of worker threads to use to generate the dataset",
    )
    parser.add_argument(
        "--auto-clean-checkpoints",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to auto clean the checkpoints after the dataset is generated",
    )
    parser.add_argument(
        "--qa-threshold",
        type=int,
        default=None,
        help="The number of QA samples to generate after which to stop the generation process. Defaults to None, which means generating Q/A samples for all documents",
    )

    # Additonal arguments
    parser.add_argument(
        "--embedding-env-prefix",
        type=str,
        default="EMBEDDING",
        help="The prefix for the OpenAI environment variables. Defaults to EMBEDDING for EMBEDDING_OPENAI_BASE_URL and EMBEDDING_OPENAI_API_KEY",
    )
    parser.add_argument(
        "--completion-env-prefix",
        type=str,
        default="COMPLETION",
        help="The prefix for the OpenAI environment variables. Defaults to COMPLETION for COMPLETION_OPENAI_BASE_URL and COMPLETION_OPENAI_API_KEY",
    )

    args = parser.parse_args()
    return args


def get_chunks(
    data_path: Path,
    doc_type: DocType = "pdf",
    chunk_size: int = 512,
    prefix: str = "EMBEDDING",
    openai_key: str | None = None,
    model: str | None = None,
) -> list[str]:
    """
    Get chunks from a document or a folder of documents.

    Args:
        data_path (Path): The path to the document or path to a folder of documents.
        doc_type (DocType, optional): The type of the document. Defaults to "pdf".
        chunk_size (int, optional): The maximum number of tokens in a chunk. Defaults to 512.
        prefix (str, optional): The prefix for the OpenAI environment variables. Defaults to "EMBEDDING".
        openai_key (str | None, optional): The OpenAI API key. Defaults to None.
        model (str | None, optional): The OpenAI embedding model. Defaults to None.

    Returns:
        list[str]: The list of chunks.
    """
    chunks = []

    logger.info(
        f"Retrieving chunks from {data_path} of type {doc_type} using the {model} model."
    )

    if doc_type == "api":
        with open(data_path, "r") as f:
            api_docs_json = json.load(f)
        chunks = list(api_docs_json)
        chunks = [str(api_doc_json) for api_doc_json in api_docs_json]

        for field in [
            "user_name",
            "api_name",
            "api_call",
            "api_version",
            "api_arguments",
            "functionality",
        ]:
            if field not in chunks[0]:
                raise TypeError(
                    f"API documentation is not in the format specified by the Gorilla API Store: Missing field `{field}`"
                )

    else:
        embeddings = build_langchain_embeddings(
            prefix=prefix, api_key=openai_key, model=model
        )
        chunks = []
        file_paths = [data_path]
        if data_path.is_dir():
            file_paths = list(data_path.rglob("**/*." + doc_type))

        futures = []
        with tqdm(total=len(file_paths), desc="Chunking", unit="file") as pbar:
            with ThreadPoolExecutor(max_workers=2) as executor:
                for file_path in file_paths:
                    futures.append(
                        executor.submit(
                            get_doc_chunks, embeddings, file_path, doc_type, chunk_size
                        )
                    )
                for future in as_completed(futures):
                    doc_chunks = future.result()
                    chunks.extend(doc_chunks)
                    pbar.set_postfix({"chunks": len(chunks)})
                    pbar.update(1)

    return chunks


def get_doc_chunks(
    embeddings: OpenAIEmbeddings | AzureOpenAIEmbeddings,
    file_path: Path,
    doc_type: DocType = "pdf",
    chunk_size: int = 512,
) -> list[str]:
    """
    Get chunks from a document based on the document type.

    Args:
        embeddings (OpenAIEmbeddings | AzureOpenAIEmbeddings): The embeddings model client.
        file_path (Path): The path to the document.
        doc_type (DocType, optional): The type of the document. Defaults to "pdf".
        chunk_size (int, optional): The maximum number of tokens in a chunk. Defaults to 512.

    Raises:
        TypeError: The document is not one of the accepted types: 'api', 'pdf', 'json', 'txt'.

    Returns:
        list[str]: The list of chunks for a document
    """
    if doc_type == "json":
        with open(file_path, "r") as f:
            data = json.load(f)
        text = data["text"]

    elif doc_type == "pdf":
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text += page.extract_text()

    elif doc_type == "txt":
        with open(file_path, "r") as f:
            data = f.read()
        text = str(data)

    else:
        raise TypeError(
            "Document is not one of the accepted types: 'api', 'pdf', 'json', 'txt'."
        )

    num_chunks = ceil(len(text) / chunk_size)
    logger.debug(f"Splitting text into {num_chunks} chunks.")

    text_splitter = SemanticChunker(embeddings, number_of_chunks=num_chunks)
    chunks = text_splitter.create_documents([text])
    chunks = [chunk.page_content for chunk in chunks]
    return chunks


def generate_api_questions(
    chat_completer: ChatCompleter, chunk: str, x: int = 5, model: str | None = None
) -> list[str]:
    """
    Generates `x` questions / use cases for `api_call`. Used when the input document is of type `api`.

    Args:
        chat_completer (ChatCompleter): The chat completer client.
        chunk (str): The chunk to generate instructions for.
        x (int, optional): The number of instructions to generate. Defaults to 5.
        model (str | None, optional): The completions model to use. Defaults to None.

    Returns:
        list[str]: The list of generated questions / instructions.
    """
    response: ChatCompletion = chat_completer(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a synthetic instruction-api pair generator. Given an API endpoint in the form of a JSON object, generate %s example queries of instructions a user could ask and would be answered by invoking the API call. For example, if the given API call is the `service.users().getProfile(userId='me').execute()` call from the Gmail API, an example query could be 'How can I fetch my Gmail account's email address?'"
                % (x),
            },
            {
                "role": "system",
                "content": "The API endpoint is a JSON object with required params: user_name, api_name, api_call, api_version, api_arguments, functionality, and optional params: env_requirements, example_code, meta_data, Questions",
            },
            {
                "role": "system",
                "content": "For instance, if the api call contains: {'user_name': 'felixzhu555', 'api_name': 'Google Maps - Address Validation', 'api_call': 'Client.addressvalidation(addressLines, regionCode=region_code, locality=locality, enableUspsCass=boolean)', 'api_version': '4.10.0', 'api_arguments': {}, 'functionality': 'Validate an address and its components, standardize the address for mailing, and determine the best known geocode for it.', 'env_requirements': ['googlemaps'], 'example_code': 'client = googlemaps.Client(key='YOUR_API_KEY')\nresponse = client.addressvalidation('1600 Amphitheatre Pk', regionCode='US', locality='Mountain View', enableUspsCass=True)', 'meta_data': {'description': 'The googlemaps python client is an abstraction for the Google Maps API that requires python 3.5+. Each Google Maps web service request requires an API key or client ID. API keys are generated in the 'Credentials' page of the 'APIs & Services' tab of Google Cloud console. This key should be kept secret on your server.'}, 'questions': []}, an example instruction would be 'Validate the following address: University Avenue and, Oxford St, Berkeley, CA 94720.'",
            },
            {
                "role": "system",
                "content": "Don't mention 'API' or use any hints or the name of the API. In one-third of the queries, make sure to include a specific example, like 'Validate this address: 123 Harrison St, Oakland CA'. Include ONLY the queries in your response.",
            },
            {"role": "user", "content": str(chunk)},
        ],
    )

    content = response.choices[0].message.content
    queries = content.split("\n")
    # Helper function for helping format strings returned by GPT-4
    queries = [strip_str(q) for q in queries]
    queries = [q for q in queries if any(c.isalpha() for c in q)]

    return queries


build_qa_messages = {
    "gpt": lambda chunk, x: [
        {
            "role": "system",
            "content": """You are a synthetic question-answer pair generator. Given a chunk of context about
             some topic(s), generate %s example questions a user could ask and would be answered using information from the chunk.
             For example, if the given context was a Wikipedia paragraph about the United States, an example question could be
             'How many states are in the United States?'"""
            % (x),
        },
        {
            "role": "system",
            "content": "The questions should be able to be answered in a few words or less. Include only the questions in your response.",
        },
        {"role": "user", "content": str(chunk)},
    ],
    "llama": lambda chunk, x: [
        {
            "role": "system",
            "content": """You are a synthetic question generator.

                Instructions:
                - Given a chunk of context about some topic(s), generate %s example questions a user could ask
                - Questions should be answerable using only information from the chunk.
                - Generate one question per line
                - Generate only questions
                - Questions should be succinct

                Here are some samples:
                Context: A Wikipedia paragraph about the United States,
                Question: How many states are in the United States?

                Context: A Wikipedia paragraph about vampire bats,
                Question: What are the different species of vampire bats?
                """
            % (x),
        },
        {
            "role": "system",
            "content": "The questions should be able to be answered in a few words or less. Include only the questions in your response.",
        },
        {"role": "user", "content": str(chunk)},
    ],
}


def generate_general_questions(
    chat_completer: ChatCompleter,
    chunk: str,
    x: int,
    model: str | None = None,
    system_prompt_key: str = "gpt",
) -> list[str]:
    """
    Generates `x` questions / use cases for `chunk`. Used when the input document is of general types
    `pdf`, `json`, or `txt`.

    Args:
        chat_completer (ChatCompleter): The chat completer client.
        chunk (str): The chunk to generate instructions for.
        x (int): The number of instructions to generate. Defaults to 5.
        model (str | None, optional): The completions model to use. Defaults to None.
        system_prompt_key (str, optional): The system prompt key. Either "gpt" or "llama". Defaults to "gpt".

    Raises:
        e: BadRequestError if there is a content filter error.

    Returns:
        list[str]: The list of generated questions / instructions.
    """
    try:
        response: ChatCompletion = chat_completer(
            model=model,
            messages=build_qa_messages[system_prompt_key](chunk, x),
            max_tokens=min(25 * x, 512),  # 25 tokens per question
        )
    except BadRequestError as e:
        if e.code == "content_filter":
            logger.warning(f"Got content filter error, skipping chunk: {e.message}.")
            return []
        raise e

    content = response.choices[0].message.content
    queries = content.split("\n") if content else []
    # queries = [strip_str(q) for q in queries]
    queries = [q for q in queries if any(c.isalpha() for c in q)]

    return queries


def strip_str(s: str) -> str:
    """
    Helper function for helping format strings returned by GPT-4.

    Args:
        s (str): The string to format.

    Returns:
        str: The formatted string.
    """
    l, r = 0, len(s) - 1
    beg_found = False
    for i in range(len(s)):
        if s[i].isalpha():
            if not beg_found:
                l = i
                beg_found = True
            else:
                r = i
    r += 2
    return s[l : min(r, len(s))]


def encode_api_question(question: str, chunk: str) -> list[str]:
    """
    Encodes multiple prompt instructions into a single string for the `api` case.

    Args:
        question (str): The question.
        chunk (str): The chunk containing the definition of the API.

    Returns:
        list[str]: The list of prompts (system and user).
    """
    prompts = []

    prompt = (
        question
        + "\nWrite a python program to call API in "
        + str(chunk)
        + ".\n\nThe answer should follow the format: <<<domain>>> $DOMAIN \n, <<<api_call>>>: $API_CALL \n, <<<api_provider>>>: $API_PROVIDER \n, <<<explanation>>>: $EXPLANATION \n, <<<code>>>: $CODE}. Here are the requirements:\n \n2. The $DOMAIN should be the domain of the API ('N/A' if unknown). The $API_CALL should have only 1 line of code that calls api.\n3. The $API_PROVIDER should be the programming framework used.\n4. $EXPLANATION should be a numbered, step-by-step explanation.\n5. The $CODE is the python code.\n6. Do not repeat the format in your answer."
    )
    prompts.append(
        {
            "role": "system",
            "content": "You are a helpful API writer who can write APIs based on requirements.",
        }
    )
    prompts.append({"role": "user", "content": prompt})
    return prompts


prompt_templates = {
    "gpt": """
        Question: {question}\nContext: {context}\n
        Answer this question using the information given in the context above. Here is things to pay attention to:
        - First provide step-by-step reasoning on how to answer the question.
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context.
        - End your response with final answer in the form <ANSWER>: $answer, the answer should be succinct.
        You MUST begin your final answer with the tag "<ANSWER>:".
    """,
    "llama": """
        Question: {question}
        Context: {context}

        Answer this question using the information given in the context above.

        Instructions:
        - Provide step-by-step reasoning on how to answer the question.
        - Explain which parts of the context are meaningful and why.
        - Copy paste the relevant sentences from the context in ##begin_quote## and ##end_quote##.
        - Provide a summary of how you reached your answer.
        - End your response with the final answer in the form <ANSWER>: $answer, the answer should be succinct.
        - You MUST begin your final answer with the tag "<ANSWER>:".

        Here are some samples:

        Example question: What movement did the arrest of Jack Weinberg in Sproul Plaza give rise to?
        Example answer: To answer the question, we need to identify the movement that was sparked by the arrest of Jack Weinberg in Sproul Plaza.
        The context provided gives us the necessary information to determine this.
        First, we look for the part of the context that directly mentions Jack Weinberg's arrest.
        We find it in the sentence: ##begin_quote##The arrest in Sproul Plaza of Jack Weinberg, a recent Berkeley alumnus and chair of Campus CORE,
        prompted a series of student-led acts of formal remonstrance and civil disobedience that ultimately gave rise to the Free Speech Movement##end_quote##.
        From this sentence, we understand that the arrest of Jack Weinberg led to student-led acts which then gave rise to a specific movement.
        The name of the movement is explicitly mentioned in the same sentence as the "Free Speech Movement."
        Therefore, based on the context provided, we can conclude that the arrest of Jack Weinberg in Sproul Plaza gave rise to the Free Speech Movement.
        <ANSWER>: Free Speech Movement
    """,
}


def encode_general_question(
    question: str, chunk: str, system_prompt_key: str = "gpt"
) -> list[str]:
    """
    Encodse multiple prompt instructions into a single string for the general case (`pdf`, `json`, or `txt`).

    Args:
        question (str): The question.
        chunk (str): The chunk.
        system_prompt_key (str, optional): The system prompt key. Either "gpt" or "llama". Defaults to "gpt".

    Returns:
        list[str]: The list of prompts (system and user).
    """
    prompts = []

    prompt = prompt_templates[system_prompt_key].format(
        question=question, context=chunk
    )
    prompts.append(
        {
            "role": "system",
            "content": "You are a helpful question answerer who can provide an answer given a question and relevant context.",
        }
    )
    prompts.append({"role": "user", "content": prompt})
    return prompts


def generate_answer(
    chat_completer: ChatCompleter,
    question: str,
    context: str,
    doc_type: DocType = "pdf",
    model: str | None = None,
    system_prompt_key: str = "gpt",
) -> str | None:
    """
    Generates the label / answer to `question` using `context` and GPT-4.

    Args:
        chat_completer (ChatCompleter): The chat completer client.
        question (str): The question.
        context (str): The chunk for context.
        doc_type (DocType, optional): The type of the document. Defaults to "pdf".
        model (str | None, optional): The completions model to use. Defaults to None.
        system_prompt_key (str, optional): The system prompt key. Either "gpt" or "llama". Defaults to "gpt".

    Returns:
        str | None: The generated label / answer.
    """
    question = (
        encode_api_question(question, chunk=context)
        if doc_type == "api"
        else encode_general_question(
            question, chunk=context, system_prompt_key=system_prompt_key
        )
    )
    response: ChatCompletion = chat_completer(
        model=model,
        messages=question,
        n=1,
        temperature=0,
        max_tokens=512,
    )
    response = response.choices[0].message.content
    return response


def generate_question_cot_answer(
    chat_completer: ChatCompleter,
    chunks: list[str],
    chunk: str,
    chunk_id: int,
    question: str,
    doc_type: DocType = "pdf",
    num_distract: int = 3,
    p: float = 1.0,
    model: str | None = None,
    system_prompt_key: str = "gpt",
) -> dict:
    """
    Generates a data point / triplet with a question, context, and chain-of-thought answer.

    This function constructs a data point / triplet containing a question, context, oracle context,
    chain-of-thought answer, and instruction based on the provided chunks and question.
    It includes option for adding distractor documents.

    Args:
        chat_completer (ChatCompleter): The chat completer client.
        chunks (list[str]): List of document chunks.
        chunk (str): The oracle chunk or document.
        chunk_id (int): The index of the oracle chunk.
        question (str): The question related to the context, if oracle document is present. Otherwise, unrelated.
        doc_type (DocType, optional): The type of the document. Defaults to "pdf".
        num_distract (int, optional): The number of distractor documents to add. Defaults to 3.
        p (float, optional): Probability of including the oracle document in the context. Defaults to 1.0.
        model (str | None, optional): The completions model to use. Defaults to None.
        system_prompt_key (str, optional): The system prompt key. Either "gpt" or "llama". Defaults to "gpt".

    Returns:
        dict: A dictionary containing the data point with keys for id, type, question,
            context, oracle_context, chain-of-thought answer, and instruction.
    """
    datapt = {
        "id": None,
        "type": None,
        "question": None,
        "context": None,
        "oracle_context": None,
        "cot_answer": None,
        "instruction": None,
    }

    datapt["id"] = str(uuid.uuid4())
    datapt["type"] = "api_call" if doc_type == "api" else "general"
    datapt["question"] = question

    # The first chunk is the oracle document
    docs = [chunk]
    indices = list(range(0, len(chunks)))
    # Remove the index of the oracle document so we don't accidentally sample it
    indices.remove(chunk_id)
    # Randomly samply `num_distract` distractor documents
    for i in random.sample(indices, num_distract):
        # Add distractor document to documents
        docs.append(chunks[i])
    # Decides whether to add oracle document
    to_add = random.uniform(0, 1) < p
    if not to_add:
        # If don't add oracle document, we remove the first document in docs
        # (which is the oracle document) and replace it with a randomly sampled
        # distractor document.
        docs[0] = chunks[random.sample(indices, 1)[0]]
    # Shuffle the documents
    random.shuffle(docs)

    d = {"title": [], "sentences": []}

    d["title"].append(
        ["placeholder_title"] * (num_distract + 1)
    )  # additonal 1 for the oracle document or distractor if not `to_add`
    d["sentences"].append(docs)

    datapt["context"] = d
    datapt["oracle_context"] = chunk
    # Add answer to question
    datapt["cot_answer"] = generate_answer(
        chat_completer,
        question,
        chunk,
        doc_type,
        model=model,
        system_prompt_key=system_prompt_key,
    )
    # Construct model instruction
    context = ""
    for doc in docs:
        context += "<DOCUMENT>" + str(doc) + "</DOCUMENT>\n"
    context += question
    datapt["instruction"] = context

    return datapt


def build_or_load_chunks(
    data_path: Path,
    checkpoints_dir: Path,
    doc_type: DocType = "pdf",
    chunk_size: int = 512,
    embedding_prefix: str = "EMBEDDING",
    openai_api_key: str | None = None,
    embedding_model: str | None = None,
) -> list[str]:
    """
    Builds chunks and checkpoints them if needed. Otherwise, loads them.

    Args:
        data_path (Path): The path to the document or path to a folder of documents.
        checkpoints_dir (Path): The path to the checkpoints directory.
        doc_type (DocType, optional): The type of the document. Defaults to "pdf".
        chunk_size (int, optional): The maximum number of tokens in a chunk. Defaults to 512.
        embedding_prefix (str, optional): The prefix for the embedding. Defaults to "EMBEDDING".
        openai_api_key (str | None, optional): The OpenAI API key. Defaults to None.
        embedding_model (str | None, optional): The embedding model. Defaults to None.

    Returns:
        list[str]: _description_
    """
    chunks_ds: Dataset = None
    chunks = None
    checkpoints_chunks_path = checkpoints_dir / "chunks"
    if checkpoints_chunks_path.exists():
        logger.info(f"Using checkpoint chunks {checkpoints_chunks_path}.")
        # Load dataset from checkpoint if the checkpoint path exists
        chunks_ds = Dataset.load_from_disk(checkpoints_chunks_path)
        chunks = chunks_ds["chunk"]

    # If checkpoint path does not exist, build chunks
    if not chunks:
        logger.info(f"Checkpoint does not exist. Building chunks.")
        chunks = get_chunks(
            data_path,
            doc_type,
            chunk_size,
            embedding_prefix,
            openai_api_key,
            model=embedding_model,
        )

    # If checkpoint path does not exist, build chunks dataset and save to checkpoint path
    if not chunks_ds:
        chunks_table = pa.table({"chunk": chunks})
        chunks_ds = Dataset(chunks_table)
        chunks_ds.save_to_disk(checkpoints_chunks_path)

    return chunks


class StoppingException(Exception):
    """
    Raised by worker threads when the process is stopping early.
    """

    pass


def stage_generate(
    chat_completer: ChatCompleter,
    checkpoints_dir: Path,
    chunks: list[str],
    num_questions: int = 5,
    max_workers: int = 2,
    doc_type: DocType = "pdf",
    completion_model: str = "gpt-4",
    system_prompt_key: str = "gpt",
    num_distract: int = 3,
    p: float = 1.0,
    qa_threshold: int | None = None,
):
    questions_checkpointing = Checkpointing(checkpoints_dir / "questions")
    answers_checkpointing = Checkpointing(checkpoints_dir / "answers")

    # Tracking when the process is stopping, so we can stop the generation process early.
    # Initial value is False.
    is_stopping = Event()

    @checkpointed(questions_checkpointing)
    def generate_chunk_questions_ds(
        chunk: str, chunk_id: int, doc_type: DocType, *args, **kwargs
    ) -> Dataset:
        """
        Generates a dataset of questions / instructions for a given chunk.

        Args:
            chunk (str): The chunk to generate questions / instructions for.
            chunk_id (int): The id of the chunk.
            doc_type (DocType): The type of the document.

        Returns:
            Dataset: The dataset of questions / instructions.
        """
        # The decorator will first check if the checkpoint exists and loads it if it is
        # and we don't have to run this logic as it can be computationally expensive.
        questions = (
            generate_api_questions(chunk=chunk, *args, **kwargs)
            if doc_type == "api"
            else generate_general_questions(chunk=chunk, *args, **kwargs)
        )
        chunk_question_pairs = [
            {"chunk": chunk, "chunk_id": chunk_id, "question": question}
            for question in questions
        ]
        questions_ds = Dataset.from_list(chunk_question_pairs)
        return questions_ds

    @checkpointed(answers_checkpointing)
    def generate_question_cot_answers(
        questions_ds: Dataset,
        chunk_id: int,
        *args,
        **kwargs,
    ) -> Dataset:
        """
        Generates a dataset of chain-of-thought answers for a given dataset of questions / instructions.

        Args:
            questions_ds (Dataset): The dataset of questions / instructions.
            chunk_id (int): The id of the oracle document.

        Returns:
            Dataset: The dataset of chain-of-thought answers.
        """

        def process_example(chunk: str, question: str) -> dict | None:
            """
            Processes the oracle document and question to generate a chain-of-thought
            answer for a given question.

            Args:
                chunk (str): The oracle document.
                question (str): The question.

            Returns:
                dict | None: A dictionary containing the data point with keys for id, type, question,
                    context, oracle_context, chain-of-thought answer, and instruction.
            """
            try:
                cot_answer = generate_question_cot_answer(
                    chunks=chunks,
                    chunk=chunk,
                    chunk_id=chunk_id,
                    question=question,
                    *args,
                    **kwargs,
                )
            except BadRequestError as e:
                if e.code == "content_filter":
                    logger.warning(
                        f"Got content filter error, skipping question '{question}': {e.message}."
                    )
                    return None
            return cot_answer

        # The decorator will first check if the checkpoint exists and loads it if it is
        # and we don't have to run this logic as it can be computationally expensive.
        results = (
            [
                process_example(chunk, question)
                for chunk, question in zip(
                    questions_ds["chunk"], questions_ds["question"]
                )
            ]
            if len(questions_ds) > 0
            else []
        )
        # Exclude None values
        results = [r for r in results if r is not None]
        table = pa.Table.from_pylist(results)
        ds = Dataset(table)
        return ds

    def process_chunk(index: int) -> Dataset:
        if is_stopping.is_set():
            raise StoppingException()
        chunk = chunks[index]
        questions_ds = generate_chunk_questions_ds(
            chunk=chunk,
            chunk_id=index,
            doc_type=doc_type,
            chat_completer=chat_completer,
            x=num_questions,
            model=completion_model,
            system_prompt_key=system_prompt_key,
        )
        answers_ds = generate_question_cot_answers(
            questions_ds=questions_ds,
            chunk=chunk,
            chunk_id=index,
            chat_completer=chat_completer,
            model=completion_model,
            doc_type=doc_type,
            system_prompt_key=system_prompt_key,
            num_distract=num_distract,
            p=p,
        )
        return answers_ds

    # We use the checkpointing to keep track of the chunks that have already been processed.
    # The answers are generated after the questions so the process might have been stopped in
    # between a batch of answers and matching questions. So we need to use the answers checkpointing
    # to keep track of which chunks we need to process if the questions for a given chunk have already
    # been checkpointed, they will just be loaded from the checkpoint we set the tqdm's initial position
    # to avoid having cached data skew the stats.
    num_chunks = len(chunks)
    missing_chunks = answers_checkpointing.missing_checkpoints(num_chunks)

    gen_questions_count = 0
    if answers_checkpointing.has_checkpoints():
        ds = answers_checkpointing.collect_checkpoints()
        gen_questions_count = len(ds)

    done_chunks = num_chunks - len(missing_chunks)
    if done_chunks > 0 or gen_questions_count > 0:
        logger.info(
            f"Resuming generation from chunk {done_chunks}/{num_chunks} and {gen_questions_count} questions."
        )

    # If we have a QA threshold, it makes more sense to keep track of the number of questions generated.
    # Otherwise, track chunks instead.
    track_questions = qa_threshold is not None

    if qa_threshold:
        logger.info(
            f"Will stop early as soon as the QA threshold is met: `qa_threshold` set to {qa_threshold} questions."
        )
    # if True means qa_thresold is set and we track the questions
    if track_questions:
        tqdm_args = {
            "total": qa_threshold,
            "unit": "qa",
            "initial": gen_questions_count,
        }
    # If not, we track the chunks
    else:
        tqdm_args = {
            "total": num_chunks,
            "unit": "chunk",
            "initial": done_chunks,
        }

    usage_stats = UsageStats()

    tps = 0
    futures = []
    answers_ds_list = []
    with tqdm(**tqdm_args, desc="Generating") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in missing_chunks:
                futures.append(executor.submit(process_chunk, i))
            for future in as_completed(futures):
                if qa_threshold and gen_questions_count >= qa_threshold:
                    logger.info(
                        f"Met QA threshold {gen_questions_count} >= {qa_threshold} questions Stopping generation."
                    )
                    is_stopping.set()
                    break
                answers_ds = future.result()
                answers_ds_list.append(answers_ds)
                increment = (
                    min(len(answers_ds), qa_threshold - gen_questions_count)
                    if track_questions
                    else 1
                )
                gen_questions_count += len(answers_ds)
                done_chunks += 1
                stats = chat_completer.get_stats_and_reset()
                if stats:
                    tps = stats.total_tokens / stats.duration
                    usage_stats += stats
                postfix = {
                    "last tok/s": tps,
                    "avg tok/s": (
                        usage_stats.total_tokens / usage_stats.duration
                        if usage_stats.duration > 0
                        else 0
                    ),
                }
                if track_questions:
                    postfix["chunks"] = done_chunks
                else:
                    postfix["qa"] = gen_questions_count
                pbar.set_postfix(postfix)
                pbar.update(increment)

    ds = answers_checkpointing.collect_checkpoints()
    ds = ds.select(range(qa_threshold)) if qa_threshold else ds
    logger.info(
        f"Consumed {usage_stats.prompt_tokens} prompt tokens, {usage_stats.completion_tokens} completion tokens, {usage_stats.total_tokens} total tokens."
    )

    return ds


def main():
    main_start = time.time()

    # Get arguments
    args = parse_args()

    # Validate arguments
    if args.output_chat_system_prompt and args.output_format != "chat":
        raise Exception(
            "Parameter `--output-chat-system-prompt` can only be used with `--output-format chat`."
        )

    OPENAPI_API_KEY: str = args.openai_key
    CHUNK_SIZE: int = args.chunk_size
    NUM_DISTRACT_DOCS: int = args.distractors

    client = build_openai_client(
        prefix=args.completion_env_prefix,
        api_key=OPENAPI_API_KEY,
    )
    chat_completer = ChatCompleter(client)

    # Absolute path returns the full path
    output_path = Path(args.output).absolute()
    checkpoints_dir = Path(str(output_path) + "-checkpoints").absolute()
    auto_clean_checkpoints = args.auto_clean_checkpoints
    if auto_clean_checkpoints:
        logger.info(
            f"Checkpoints will be automatically deleted after dataset generation. Remove `--auto-clean-checkpoints` argument to deactivate."
        )

    datapath: Path = args.datapath
    datasets.disable_progress_bars()

    # Build or load chunks
    chunks = build_or_load_chunks(
        datapath,
        args.doc_type,
        CHUNK_SIZE,
        args.embedding_env_prefix,
        OPENAPI_API_KEY,
        args.embedding_model,
        checkpoints_dir,
    )

    cot_answers_ds = None

    num_questions: int = args.questions
    max_workers: int = args.workers
    doc_type: str = args.doc_type
    completion_model: str = args.completion_model
    system_prompt_key: str = args.system_prompt_key

    logger.info(f"Using system prompt key: {system_prompt_key}.")
    logger.info(f"Using {max_workers} worker threads.")

    cot_answers_ds = stage_generate(
        chat_completer,
        checkpoints_dir,
        chunks,
        num_questions,
        max_workers,
        doc_type,
        completion_model,
        system_prompt_key,
        num_distract=NUM_DISTRACT_DOCS,
        p=args.p,
        qa_threshold=args.qa_threshold,
    )

    # Save as .arrow format
    datasets.enable_progress_bars()
    cot_answers_ds.save_to_disk(str(output_path))

    # Save as .jsonl format
    formatter = DatasetConverter()

    # Extract format specific parameters
    format_params = {}
    if args.output_chat_system_prompt:
        format_params["system_prompt"] = args.output_chat_system_prompt

    if args.output_format == "completion":
        format_params["prompt_column"] = args.output_completion_prompt_column
        format_params["completion_column"] = args.output_completion_completion_column

    formatter.convert(
        cot_answers_ds,
        args.output_format,
        str(output_path),
        args.output_type,
        **format_params,
    )

    # WARNING: This deletes all intermediate checkpoint files
    if auto_clean_checkpoints:
        shutil.rmtree(checkpoints_dir)

    logger.info(f"Generated {len(cot_answers_ds)} QA/CoT/documents samples.")
    logger.info(f"Dataset saved to {output_path}.")
    logger.info(f"Done in {time.time() - main_start:.2f}s.")


if __name__ == "__main__":
    with MDC(progress="0%"):
        main()
