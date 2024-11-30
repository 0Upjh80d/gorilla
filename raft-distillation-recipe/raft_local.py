import argparse
import json
import logging
import os
import random
import shutil
from math import ceil
from pathlib import Path

import PyPDF2
import torch
from datasets import Dataset, concatenate_datasets
from format import dataset_formats, output_dataset_types
from logconf import log_setup
from raft import DocType, doc_types
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

log_setup()

logger = logging.getLogger("huggingface_script")

# Every N chunks, save a checkpoint
N = 15


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
        "--fast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the script in fast mode (no recovery implemented)",
    )

    args = parser.parse_args()
    return args


def get_chunks(
    data_path: Path,
    doc_type: DocType = "pdf",
    chunk_size: int = 512,
) -> list[str]:
    """
    Get chunks from a document or a folder of documents.

    Args:
        data_path (Path): The path to the document or path to a folder of documents.
        doc_type (DocType, optional): The type of the document. Defaults to "pdf".
        chunk_size (int, optional): The maximum number of tokens in a chunk. Defaults to 512.

    Returns:
        list[str]: The list of chunks.
    """
    chunks = []

    logger.info(f"Retrieving chunks from {data_path} of type {doc_type}.")

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
        if doc_type == "json":
            with open(data_path, "r") as f:
                data = json.load(f)
            text = data["text"]

        elif doc_type == "pdf":
            text = ""
            with open(data_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                for page_num in range(num_pages):
                    page = reader.pages[page_num]
                    text += page.extract_text()

        elif doc_type == "txt":
            with open(data_path, "r") as f:
                data = f.read()
            text = str(data)

        else:
            raise TypeError(
                "Document is not one of the accepted types: 'api', 'pdf', 'json', 'txt'."
            )

        num_chunks = ceil(len(text) / chunk_size)
        logger.debug(f"Splitting text into {num_chunks} chunks.")
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i : i + chunk_size])
        return chunks


def generate_questions_hf(
    chunk: str, x: int = 5, model_name: str = "t5-small"
) -> list[str]:
    """
    Uses a HuggingFace model to generate `x` questions / instructions based on the given text chunk,
    utilizing the GPU if available.

    Args:
        chunk (str): The text chunk.
        x (int, optional): The number of questions / instructions to generate. Defaults to 5.
        model_name (str, optional): The name of the HuggingFace model. Defaults to "t5-small".

    Returns:
        list[str]: The list of generated questions / instructions.
    """
    # Load the HuggingFace model and tokenizer for question generation
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_text = f"Generate questions based on the following text: {chunk}."
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, padding="longest"
    ).to(device)

    outputs = model.generate(
        inputs.input_ids,
        max_length=64,
        num_beams=x,  # using beam search with `x` beams
        num_return_sequences=x,  # returns `x` sequences
    )

    questions = [
        tokenizer.decode(output, skip_special_tokens=True) for output in outputs
    ]

    return questions


def generate_answer_hf(
    question: str, context: str, model_name: str = "deepset/roberta-base-squad2"
) -> str:
    """
    Uses a HuggingFace model to generate an answer / label to the given question based on the context,
    utilizing the GPU if available.

    Args:
        question (str): The question.
        context (str): The context.
        model_name (str, optional): The name of the HuggingFace model. Defaults to "deepset/roberta-base-squad2".

    Returns:
        str: The generated answer / label.
    """
    # Load the HuggingFace model and tokenizer for question-answering
    question_answering_pipeline = pipeline(
        "question-answering",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1,
    )
    result = question_answering_pipeline(question=question, context=context)
    return result["answer"]


def add_chunk_to_dataset(
    chunks: list[str],
    chunk: str,
    doc_type: DocType = "pdf",
    x: int = 5,
    num_distract: int = 3,
    p: float = 1.0,
    model_name_qg: str = "t5-small",
    model_name_qa: str = "deepset/roberta-base-squad2",
):
    """
    Given a chunk, create {Q, A, D} triplets and add them to the dataset using HuggingFace models.

    Args:
        chunks (list[str]): List of document chunks.
        chunk (str): The oracle chunk or document.
        doc_type (DocType, optional): The type of the document. Defaults to "pdf".
        x (int, optional): The number of questions / instructions to generate. Defaults to 5.
        num_distract (int, optional): The number of distractor documents to add. Defaults to 3.
        p (float, optional): Probability of including the oracle document in the context. Defaults to 1.0.
        model_name_qg (str, optional): The name of the HuggingFace model for question generation. Defaults to "t5-small".
        model_name_qa (str, optional): The name of the HuggingFace model for question-answering. Defaults to "deepset/roberta-base-squad2".
    """
    global ds
    chunk_id = chunks.index(chunk)

    # Use the HuggingFace model to generate questions
    qs = generate_questions_hf(chunk, x, model_name=model_name_qg)
    for q in qs:
        datapt = {
            "id": None,
            "type": None,
            "question": None,
            "context": None,
            "oracle_context": None,
            "cot_answer": None,
        }

        datapt["id"] = f"seed_task_{0 if not ds else ds.num_rows}"
        datapt["type"] = "api_call" if doc_type == "api" else "general"
        datapt["question"] = q

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

        d = {"title": ["placeholder_title"] * (num_distract + 1), "sentences": docs}
        datapt["context"] = d
        datapt["oracle_context"] = chunk
        # Add the answer generated by the HuggingFace model
        datapt["cot_answer"] = generate_answer_hf(q, chunk, model_name=model_name_qa)
        # Construct model instruction
        context = ""
        for doc in docs:
            context += "<DOCUMENT>" + str(doc) + "</DOCUMENT>\n"
        context += q
        datapt["instruction"] = context

        # Add to dataset
        if not ds:
            # Initialize dataset
            datapt["id"] = [datapt["id"]]
            datapt["type"] = [datapt["type"]]
            datapt["question"] = [datapt["question"]]
            datapt["context"] = [datapt["context"]]
            datapt["oracle_context"] = [datapt["oracle_context"]]
            datapt["cot_answer"] = [datapt["cot_answer"]]
            datapt["instruction"] = [datapt["instruction"]]
            ds = Dataset.from_dict(datapt)
        else:
            ds = ds.add_item(datapt)


def save_checkpoint(state: int, filename: str):
    """
    Saves the current state of processing to a file for recovery.

    Args:
        state (int): The state of the dataset (i.e. index in an iterable).
        filename (str): The filename to save the state to.
    """
    with open(filename, "w") as f:
        f.write(str(state))


def load_checkpoint(filename: str) -> int:
    """
    Loads the processing state from a checkpoint file.

    Args:
        filename (str): The filename to load the state from.

    Returns:
        int:  The state of the dataset (i.e. index in an iterable).
    """
    with open(filename, "r") as f:
        return int(f.read())


def main():
    global ds

    # Get command line arguments
    args = parse_args()

    CHUNK_SIZE = args.chunk_size
    NUM_DISTRACT_DOCS = args.distractors

    # Split the document into chunks
    chunks = get_chunks(args.datapath, args.doc_type, CHUNK_SIZE)

    ds = None

    num_chunks = len(chunks)

    if not args.fast:
        start = 0
        if os.path.exists("checkpoint.txt"):
            start = load_checkpoint("checkpoint.txt")

        for i in range((start // N) * N, len(chunks)):
            chunk = chunks[i]
            save_checkpoint(i, "checkpoint.txt")

            logger.info(f"Adding chunk {i}/{num_chunks}.")
            add_chunk_to_dataset(
                chunks, chunk, args.doc_type, args.questions, NUM_DISTRACT_DOCS
            )

            if (i + 1) % N == 0:
                ds.save_to_disk(args.output + "-checkpoints-" + str(i))
                # Set to None as N may not always be a factor of len(chunks), so if we exit,
                # we can check if ds is not None and if True, we will save it to disk.
                ds = None

        if ds:
            ds.save_to_disk(args.output + "-checkpoints-last")

        ds_list = []

        for filename in os.listdir(os.path.dirname(args.output)):
            if "-checkpoints-" in filename:
                for f in os.listdir(os.path.dirname(args.output) + "/" + filename):
                    if f.endswith(".arrow"):
                        ds_list.append(
                            Dataset.from_file(
                                os.path.dirname(args.output) + "/" + filename + "/" + f
                            )
                        )

        ds = concatenate_datasets(ds_list)

    else:
        for i, chunk in enumerate(chunks):
            logger.info(f"Adding chunk {i}/{num_chunks}.")
            add_chunk_to_dataset(
                chunks, chunk, args.doctype, args.questions, NUM_DISTRACT_DOCS
            )

        # Save the final dataset
        ds.save_to_disk(args.output)

        # Save as .jsonl format (dummy functionality)
        # TODO: Implement a conversion function if needed, this is just a placeholder
        logger.info("Converting dataset to the desired format...")

        if not args.fast:
            os.remove("checkpoint.txt")
            for filename in os.listdir(os.path.dirname(args.output)):
                if "-checkpoints-" in filename:
                    shutil.rmtree(os.path.dirname(args.output) + "/" + filename)


if __name__ == "__main__":
    logger.info("Starting the HuggingFace processing script...")
    main()
