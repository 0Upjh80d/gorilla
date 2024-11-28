import argparse
import logging
from abc import ABC, abstractmethod
from typing import Any, Literal, get_args

from datasets import Dataset, load_dataset

from logconf import log_setup

log_setup()

logger = logging.getLogger("format")

DatasetFormat = Literal["hf", "completion", "chat", "eval"]
dataset_formats = list(get_args(DatasetFormat))

OutputDatasetType = Literal["parquet", "jsonl"]
output_dataset_types = list(get_args(OutputDatasetType))

InputDatasetType = Literal["arrow", "jsonl"]
input_dataset_types = list(get_args(InputDatasetType))

DEFAULT_CHAT_SYSTEM_PROMPT = "The following is a conversation with an AI assistant. The assistant is helpful, clever, friendly and gives concise and accurate answers."


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
        "--input", type=str, required=True, help="Input HuggingFace dataset file"
    )
    parser.add_argument(
        "--input-type",
        type=str,
        default="arrow",
        help="Format of the input dataset. Defaults to arrow",
        choices=input_dataset_types,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory to save the dataset to",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        required=True,
        help="Format to convert the dataset to",
        choices=dataset_formats,
    )
    parser.add_argument(
        "--output-type",
        type=str,
        default="jsonl",
        help="Type to export the dataset to. Defaults to jsonl.",
        choices=output_dataset_types,
    )
    parser.add_argument(
        "--output-chat-system-prompt",
        type=str,
        default=DEFAULT_CHAT_SYSTEM_PROMPT,
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
        "--output-completion-stop",
        type=str,
        default="<STOP>",
        help="The stop keyword to use for the completion format",
    )

    args = parser.parse_args()
    return args


class DatasetFormatter(ABC):
    """
    Base class for dataset formatters. Formatters rename columns, remove and add
    columns to match the expected target format structure. HF, Chat or Completion models file formats.
    https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
    """

    @abstractmethod
    def format(self, ds: Dataset, **kwargs) -> Dataset:
        pass


class DatasetExporter(ABC):
    """
    Base class for dataset exporters. Exporters export dataset to different file types, JSONL, Parquet, ...
    """

    @abstractmethod
    def export(self, ds: Dataset, output_path: str):
        pass


class DatasetConverter:
    formats: dict[DatasetFormat, DatasetFormatter]
    exporters: dict[OutputDatasetType, DatasetExporter]

    def __init__(self) -> None:
        self.formats = {
            "hf": HuggingFaceDatasetFormatter(),
            "completion": OpenAICompletionDatasetFormatter(),
            "chat": OpenAIChatDatasetFormatter(),
            "eval": EvalDatasetFormatter(),
        }
        self.exporters = {
            "parquet": ParquetDatasetExporter(),
            "jsonl": JsonlDatasetExporter(),
        }

    def convert(
        self,
        ds: Dataset,
        format: DatasetFormat,
        output_path: str,
        output_type: OutputDatasetType,
        **kwargs,
    ):
        """
        Formats and exports the dataset to a file.

        Args:
            ds (Dataset): The dataset to format and export.
            format (DatasetFormat): The format of the dataset.
            output_path (str): The path to export the dataset to.
            output_type (OutputDatasetType): The type of the output file.

        Raises:
            Exception: Raised if the format or output type is not supported.
        """
        if not format in self.formats:
            raise Exception(
                f"Output format {format} is not supported. Please select one of {self.formats.keys()}."
            )

        if not output_type in self.exporters:
            raise Exception(
                f"Output type {output_type} is not supported. Please select one of {self.exporters.keys()}."
            )

        formatter = self.formats[format]
        new_ds = formatter.format(ds, **kwargs)
        exporter = self.exporters[output_type]
        exporter.export(new_ds, output_path)


def _remove_all_columns_but(ds: Dataset, keep_columns: list[str]) -> Dataset:
    """
    Removes all columns from the dataset except the ones in keep_columns.

    Note: HuggingFace Dataset does not have a way to copy only specific columns of a
        Dataset so this help removes all columns but the ones specified.

    Args:
        ds (Dataset): The dataset to remove columns from.
        keep_columns (list[str]): The columns to keep.

    Raises:
        Exception: Raised if a column is not found in the dataset.

    Returns:
        Dataset: The dataset with only the columns in to keep.
    """
    remove_columns = list(ds.column_names)
    for keep in keep_columns:
        try:
            remove_columns.remove(keep)
        except ValueError:
            raise Exception(f"Column {keep} not found in {remove_columns}.")
    ds = ds.remove_columns(remove_columns)
    return ds


class HuggingFaceDatasetFormatter(DatasetFormatter):
    """Returns the HuggingFace Dataset as is without formatting."""

    def format(self, ds: Dataset) -> Dataset:
        return ds


class OpenAICompletionDatasetFormatter(DatasetFormatter):
    """
    Returns the Dataset in the OpenAI Completion Fine-tuning file format with two fields "prompt" and "completion".
    Field names can be customized because different systems have different expectations.

    https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
    """

    def format(
        self,
        ds: Dataset,
        prompt_column: str = "prompt",
        completion_column: str = "completion",
        stop: str = "<STOP>",
    ) -> Dataset:
        new_ds = ds.filter(
            lambda example: example["cot_answer"] and example["instruction"],
            desc="Filter out empty examples",
        )
        new_ds = new_ds.rename_columns({"instruction": prompt_column})
        new_ds = new_ds.map(
            lambda examples: {
                completion_column: [answer + stop for answer in examples["cot_answer"]]
            },
            batched=True,
            desc=f"Rename fields and add '{stop}' token",
        )
        return _remove_all_columns_but(new_ds, [prompt_column, completion_column])


class OpenAIChatDatasetFormatter(OpenAICompletionDatasetFormatter):
    """
    Returns the Dataset in the OpenAI Chat Fine-tuning file format with one field "messages".

    https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
    """

    def format(self, ds: Dataset, system_prompt: str, **kwargs) -> Dataset:
        new_ds = super().format(ds, stop="", **kwargs)

        def format_messages(row: dict) -> dict:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(
                [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": row["completion"]},
                ]
            )
            chat_row = {"messages": messages}
            return chat_row

        new_ds = new_ds.map(format_messages)
        return _remove_all_columns_but(new_ds, ["messages"])


def extract_final_answer(cot_answer: str) -> str:
    """
    Extracts the final answer from the cot_answer field

    Args:
        cot_answer (str): The Chain-of-Thought answer.

    Returns:
        str: The extracted final answer.
    """
    if cot_answer:
        return cot_answer.split("<ANSWER>: ")[-1]
    return None


def extract_context(instruction: str) -> str:
    """
    Extracts the context from the instruction field.
    Keeps all <DOCUMENTS/> and removes the last line with the question.

    Args:
        instruction (str): The instruction.

    Returns:
        str: The extracted context.
    """
    return "\n".join(instruction.split("\n")[:-1])


class EvalDatasetFormatter(DatasetFormatter):
    """
    Returns the Dataset in a format suitable for evaluation. Extracts final answer separates context from question.
    """

    def format(self, ds: Dataset) -> Dataset:
        new_ds = ds.filter(
            lambda example: example["cot_answer"]
            and example["instruction"]
            and example["context"],
            desc="Filter out empty examples",
        )
        new_ds = new_ds.rename_columns({"context": "context_sentences"})
        new_ds = new_ds.map(
            lambda examples: {
                "gold_final_answer": [
                    extract_final_answer(answer) for answer in examples["cot_answer"]
                ]
            },
            batched=True,
        )
        keep_columns = ["question", "gold_final_answer", "context"]
        if "answer" in new_ds.column_names:
            [keep_columns.append(col) for col in ["answer", "final_answer"]]
            new_ds = new_ds.map(
                lambda examples: {
                    "final_answer": [
                        extract_final_answer(answer) for answer in examples["answer"]
                    ]
                },
                batched=True,
            )
        new_ds = new_ds.map(
            lambda examples: {
                "context": [
                    extract_context(instruction)
                    for instruction in examples["instruction"]
                ]
            },
            batched=True,
        )
        return _remove_all_columns_but(new_ds, keep_columns)


def append_extension(path: str, extension: str) -> str:
    """
    If provided a path without the extension, appends it.

    Args:
        path (str): The path to the file (with or without the appropriate extension).
        extension (str): The extension to append.

    Returns:
        str: The path with the extension appended.
    """
    suffix = "." + extension
    if not path.endswith(suffix):
        path = path + suffix
    return path


class ParquetDatasetExporter(DatasetExporter):
    """Exports the Dataset to a JSONL file."""

    def export(self, ds: Dataset, output_path: str):
        ds.to_parquet(append_extension(output_path, "parquet"))


class JsonlDatasetExporter(DatasetExporter):
    """Exports the Dataset to a Parquet file."""

    def export(self, ds: Dataset, output_path: str):
        ds.to_parquet(append_extension(output_path, "jsonl"))


def main():
    args = parse_args()

    input_type = args.input_type
    # Datasets expect JSON when loading JSONL files
    if input_type == "jsonl":
        input_type = "json"

    ds = load_dataset(input_type, data_files={"train": args.input})["train"]
    logger.info(f"Dataset has {ds.num_rows} rows.")

    formatter = DatasetConverter()

    format_params = {}
    if args.output_chat_system_prompt and args.output_format == "chat":
        format_params["system_prompt"] = args.output_chat_system_prompt

    if args.output_format == "completion":
        format_params["prompt_column"] = args.output_completion_prompt_column
        format_params["completion_column"] = args.output_completion_completion_column
        format_params["stop"] = args.output_completion_stop

    logger.info(
        f"Converting {args.input_type} file {args.input} to {args.output_type} {args.output_format} file {args.output}."
    )

    formatter.convert(
        ds,
        args.output_format,
        args.output,
        args.output_type,
        **format_params,
    )


if __name__ == "__main__":
    main()
