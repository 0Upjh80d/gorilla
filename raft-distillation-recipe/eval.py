import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Literal, get_args

from dotenv import load_dotenv
from openai import RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.completion import Completion
from tenacity import (
    Retrying,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    wait_exponential,
)
from tqdm import tqdm

from client_utils import StatsCompleter, UsageStats, build_openai_client
from logconf import log_setup

log_setup()

load_dotenv()  # take environment variables from .env.

logger = logging.getLogger("eval")

ModeType = Literal["chat", "completion"]
mode_types = list(get_args(ModeType))


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
        "--question-file", type=str, required=True, help="Path to question file"
    )
    parser.add_argument(
        "--answer-file", type=str, required=True, help="Path to answer file"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4", help="The model to evaluate"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="chat",
        help="The model API mode. 'chat' or 'completion' mode. Defaults to 'chat' mode.",
        choices=mode_types,
    )
    parser.add_argument(
        "--input-prompt-key",
        type=str,
        default="instruction",
        help="The column to use as input prompt",
    )
    parser.add_argument(
        "--output-answer-key",
        type=str,
        default="answer",
        help="The column to use as output answer",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="The number of worker threads to use to evaluate the dataset",
    )
    parser.add_argument(
        "--env-prefix",
        type=str,
        default="EVAL",
        help="The OPENAI environment variable prefix. Defaults to EVAL for EVAL_OPENAI_BASE_URL and EVAL_OPENAI_API_KEY",
    )

    args = parser.parse_args()
    return args


# Wrap with retry decorator
@retry(
    wait=wait_exponential(multiplier=1, min=10, max=120),
    reraise=True,
    retry=retry_if_exception_type(RateLimitError),
    before_sleep=before_sleep_log(logger, logging.INFO),
)
def retry_complete(*args, **kwargs) -> Callable:
    return complete(*args, **kwargs)


# Evaluate a chat model
def get_openai_response_chat(
    prompt: str | list[ChatCompletionMessageParam],
) -> str | None:
    messages = [{"role": "user", "content": prompt}]
    response: ChatCompletion = completions_completer(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
        stop="<STOP>",
    )
    return response.choices[0].message.content


# Evaluate a completion model
def get_openai_response_completion(prompt: str) -> str | None:
    response: Completion = completions_completer(
        model=model, prompt=prompt, temperature=0.2, max_tokens=1024, stop="<STOP>"
    )
    return response.choices[0].text


def get_answer(input_json: dict[str, Any]) -> dict[str, Any]:
    prompt = input_json[prompt_key]
    try:
        result = get_openai_response(prompt)
        input_json[answer_key] = result
    except Exception as e:
        input_json["error"] = str(e)
    return input_json


def write_result_to_file(result: dict[str, Any], write_file_name: str) -> None:
    global file_write_lock
    with file_write_lock:
        with open(write_file_name, "a") as outfile:
            json.dump(result, outfile)
            outfile.write("\n")


if __name__ == "__main__":
    args = parse_args()

    model = args.model
    mode = args.mode
    prompt_key = args.input_prompt_key
    answer_key = args.output_answer_key

    logger.info(f"Using model: {model}.")
    logger.info(f"Using mode: {mode}.")
    logger.info(f"Using prompt key: {prompt_key}.")
    logger.info(f"Using answer key: {answer_key}.")

    client = build_openai_client(prefix=args.env_prefix)

    # Chat or completion mode function
    complete = (
        client.chat.completions.create if mode == "chat" else client.completions.create
    )

    # Wrap with statistics completer
    completions_completer = StatsCompleter(retry_complete)

    # Chat or completion mode function
    get_openai_response = (
        get_openai_response_chat if mode == "chat" else get_openai_response_completion
    )

    write_file_name = args.answer_file
    if os.path.isfile(write_file_name):
        logger.info(f"Removing existing file: {write_file_name}.")
        os.remove(write_file_name)

    num_workers = args.workers
    file_write_lock = mp.Lock()
    inputs = []
    question_file = args.question_file
    logger.info(f"Reading questions from: {question_file}.")
    with open(question_file, "r") as f:
        for line in f:
            inputs.append(json.loads(line))

    logger.info(f"Number of questions: {len(inputs)}")
    start_time = time.time()
    usage_stats = UsageStats()
    tps = 0
    retrying: Retrying = retry_complete.retry
    error_count = 0
    with tqdm(total=len(inputs), unit="answers") as pbar:
        with ThreadPoolExecutor(num_workers) as executor:
            futures = [executor.submit(get_answer, input) for input in inputs]

            for future in as_completed(futures):
                result = future.result()
                stats = completions_completer.get_stats_and_reset()
                if stats:
                    tps = stats.total_tokens / stats.duration
                    usage_stats += stats

                retry_stats = retrying.statistics
                if len(retry_stats.keys()) > 0:
                    logger.info(f"Retrying stats: {retry_stats}.")
                postfix = {"last tok/s": tps}
                if usage_stats.total_tokens > 0:
                    postfix["avg tok/s"] = (
                        usage_stats.total_tokens / usage_stats.duration
                    )
                if "error" in result:
                    error_count += 1
                    postfix["errors"] = error_count
                pbar.set_postfix(postfix)
                pbar.update(1)
                write_result_to_file(result, write_file_name)

    end_time = time.time()
    logger.info(f"Wrote evaluation results to {write_file_name}")
    logger.info(f"total time used: {end_time - start_time}")
    if error_count > 0:
        logger.error(f"Total errors: {error_count} out of {len(inputs)} questions")
