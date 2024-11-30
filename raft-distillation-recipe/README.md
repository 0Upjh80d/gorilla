# Rretrieval-Augmented Fine-Tuning (RAFT)

## Table of Contents

- [Overview](#overview)
- [Dev Environment with Codespaces](#codespaces)
- [Install Dependencies](#install-dependencies)
- [Usage](#usage)
  - [RAFT Arguments](#raft-arguments)
  - [Using with OpenAI API](#using-with-openai-api)
  - [Using with Azure OpenAI API](#using-with-azure-openai-api)
  - [Configuration](#configuration)
- [Examples](#examples)
  - [Using with Azure Models](#using-with-azure-models)
  - [Using with HuggingFace Models Completely Locally](#using-with-huggingface-models-completely-locally)
- [Step-by-Step Generation Process](#step-by-step-generation-process)
  - [1. Chunk Generation](#1-chunk-generation)
  - [2. Question and Answer Generation](#2-question-and-answer-generation)
  - [3. Append Distractor Documents](#3-append-distractor-documents)
  - [4. Generate and Save the Dataset](#4-generate-and-save-the-dataset)
  - [5. Dataset Conversion](#5-dataset-conversion)
  - [6. Fine-tuning on Microsoft AI Studio](#6-fine-tuning-on-microsoft-ai-studio)
  - [7. Evaluate RAFT Model](#7-evaluate-raft-model)

## OVerview <a id="overview"></a

RAFT is a recipe to adapting LLMs to domain-specific RAG. You can learn more in our release-blogs [here](https://gorilla.cs.berkeley.edu/blogs/9_raft.html) and [here](https://techcommunity.microsoft.com/t5/ai-ai-platform-blog/bg-p/AIPlatformBlog). RAFT takes an input document from the user and creates a dataset using the document, consisting of synthetically generated `{ question, answer, documents }` triplets. The dataset can then be used to fine-tune models for improved question-answering and retrieval.

The input data from the user can be either a general text document (PDF, JSON, or TXT) for general QA or an API documentation in the API Zoo JSONL format for API calling.

## Dev Environment with Codespaces <a id ="codespaces"></a>

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/ShishirPatil/gorilla/tree/codespaces?devcontainer_path=.devcontainer%2Fraft%2Fdevcontainer.json)

Everything is setup automatically in the dev container, open a terminal into the `raft` folder:

> [!NOTE]
> The `raft` virtual environment will be activated in your shell when entering into the `raft` folder.

## Install Dependencies <a id="install-dependencies"></a>

Dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
```

## Usage <a id="usage"></a>

### RAFT Arguments <a id="raft-arguments"></a>

Here are the RAFT arguments:

- `--datapath` - If a file, the path at which the document is located. If a folder, the path at which to load all documents
- `--output` - The path at which to save the dataset
- `--output-format` - The format of the output dataset. Defaults to `hf` for HuggingFace. Can be one of `hf`, `completion`, `chat`, `eval`.
- `--output-type` - The type of the output dataset file. Defaults to `jsonl`. Can be one of `jsonl`, `parquet`.
- `--output-chat-system-prompt` - The system prompt to use when the output format is `chat`. Optional.
- `--output-completion-prompt-column` - The column (json field name) for the `prompt` / `instruction` when using the `completion` output format. Defaults to `prompt`.
- `--output-completion-completion-column` - The column (json field name) for the `completion` when using the `completion` output format. Defaults to `completion`.
- `--distractors` - The number of distractor documents to include per data point / triplet
- `--doc-type` - The type of the document, must be one of the accepted doctypes
  - Currently accepted doctypes: `pdf`, `txt`, `json`, `api`
  - Documents in `json` format must have a "text" attribute containing the content from which chunks are extracted
  - Documents in `api` format must follow the API json format detailed in the Gorilla [API Store](https://github.com/ShishirPatil/gorilla/blob/main/data/README.md)
- `--p` - The percentage of including the oracle documents in the context
- `--chunk-size` - The size of each chunk in number of tokens
- `--questions` - The number of data points / triplets to generate per chunk
- `--openai-key` - Your OpenAI key used to make queries to GPT-3.5 or GPT-4
- `--embedding-model` - The embedding model to use to encode documents chunks. Defaults to `text-embedding-ada-002`.
- `--completion-model` - The model to use to generate questions and answers. Defaults to `gpt-4`.
- `--system-prompt-key` - The system prompt key to use to generate the dataset. Defaults to `gpt`. Can by one of `gpt`, `llama`.
- `--workers` - The number of worker threads to use to generate the dataset. Defaults to 2.
- `--auto-clean-checkpoints` - Whether to auto clean the checkpoints after the dataset is generated. Defaults to `false`.
- `--qa-threshold` - The number of QA samples to generate after which to stop the generation process. Defaults to None, which means generating Q/A samples for all documents.

Additonal arguments for [RAFT Distillation Recipe repository](https://github.com/0Upjh80d/raft-distillation-recipe):

- `--embedding-env-prefix` - The prefix for the OpenAI environment variables. Defaults to `EMBEDDING`.
- `--completion-env-prefix` - The prefix for the OpenAI environment variables. Deaults to `COMPLETION`.

Please refer to the [`raft.py`](raft.py) script for more information how these arguments are used.

> [!NOTE]
> The `--fast` mode flag has been removed, checkpointing is now always active.

### Using with OpenAI API <a id="using-with-openai-api"></a>

Create a file `.env`. All standard OpenAI environment variables are supported. Run the following command with your desired arguments to generate the synthetic dataset.

```bash
python3 raft.py \
  --datapath PATH_TO_DATA \
  --output OUTPUT_PATH \
  --output-format "hf" \ # or completion or chat
  --distractors 3 \
  --p 1.0 \
  --doc-type "pdf" \
  --chunk-size 512 \
  --questions 5 \
  --openai-key YOUR_OPENAI_KEY
```

> [!NOTE]
> As an alternative to passing the OpenAI key with the `--openai-key` argument, you also store the standard OpenAI environment variables in a file called `.env` like so. All standard OpenAI environment variables are supported.

```yml
# OpenAI API
OPENAI_API_KEY=<YOUR_OPENAI_KEY>
```

The `raft.py` script does the following:

- Takes a document located at `PATH_TO_DATA` and breaks it into chunks of size `chunk-size` tokens if the data is a PDF, JSON, TXT or API endpoint (if the data is an API documentation). This is denoted by the `doc-type` argument.
- For each chunk, uses GPT-4 to synthetically generate `questions` question-answer pairs and adds `distractors` distractor chunks to each pair, creating `{ Q, A, D }` triplets. Each triplet represents one datapoint in the dataset, where:
  - `Q`: is denotes the question/use-case
  - `A`: is the answer
  - `D`: is the relevant chunk with distractor chunks.
- Each data point / triplet also contains other attributes (e.g. metadata), such as `id`, `type`, and `cot_answer`.
- Uses the HuggingFace Dataset API to create a dataset from all triplets and saves it at `OUTPUT_PATH` in the `.arrow` and `.jsonl` formats.

### Using with Azure OpenAI API <a id="using-with-azure-openai-api"></a>

Create a file `.env`. All standard Azure OpenAI environment variables are supported. Run the following command with your desired arguments to generate the synthetic dataset.

```yml
# Azure OpenAI API
AZURE_OPENAI_ENDPOINT=https://<ENDPOINT_SUB_DOMAIN>.openai.azure.com/
AZURE_OPENAI_API_KEY=<YOUR_AZURE_OPENAI_KEY>
OPENAI_API_VERSION=<API_VERSION>
```

> [!IMPORTANT]
> Ensure you strip the path from the endpoint and keep just the domain. The full base URL will be automatically built based on the other environment variables. In addition, if you used non-default Azure OpenAI deployment names, you'll need to specify them using the following CLI arguments:

```bash
--completion-model <YOUR_GPT_DEPLOYMENT_NAME>
--embedding-model <YOUR_EMBEDDING_DEPLOYMENT_NAME>
```

### Configuration <a id="configuration"></a>

When using a non-OpenAI endpoints, it is often the case that the endpoints for the embedding and completion models
are different. In that situation, it is possible to override default OpenAI and Azure OpenAI environment variables with `COMPLETION_` or `EMBEDDING_` prefixed environment variables. To configure different endpoints for the completion and embedding models, review this example:

```yml
# Llama 3 70b Instruct completion model
# Uses an OpenAI v1 compatible endpoint on Azure MaaS
COMPLETION_OPENAI_BASE_URL=https://Meta-Llama-3-70B-Instruct-<REPLACE_ME>-serverless.eastus2.inference.ai.azure.com/v1
COMPLETION_OPENAI_API_KEY=<REPLACE_ME>

# Ada 2 embedding model
# Uses an Azure OpenAI endpoint
EMBEDDING_AZURE_OPENAI_ENDPOINT=https://<REPLACE_ME>.openai.azure.com/
EMBEDDING_AZURE_OPENAI_API_KEY=<REPLACE_ME>
EMBEDDING_OPENAI_API_VERSION=<REPLACE_ME>
```

The CLI command to run the [`raft.py`](raft.py) will look like:

```bash
python3 raft.py \
    --datapath "$PWD/sample_data/UC_Berkeley.pdf" \
    --output "$PWD/output" \
    --distractors 3 \
    --doc-type "pdf" \
    --chunk-size 512 \
    --questions 5 \
    --completion-model "Meta-Llama-3-70B-Instruct-<REPLACE_ME>" \
    --embedding-model "text-embedding-ada-002"
```

> [!NOTE]
> The `--completion-model` and `--embedding-model` in the case of an Azure OpenAI endpoint must be set to the deployment names.

## Examples <a id="examples"></a>

### Using with Azure Models <a id="using-with-azure-models"></a>

This details the commands and process used to generate an example synthetic dataset. The document is a PDF of the [Wikipedia page on the United States of America](sample_data/United_States_PDF.pdf).

```bash
python3 raft.py \
  --datapath "./sample_data/United_States_PDF.pdf" \
  --output "./outputs" \
  --distractors 4 \
  --doc-type "pdf" \
  --chunk-size 512 \
  --questions 5 \
  --openai-key OPENAI_KEY
```

### Using with HuggingFace Models Completely Locally <a id="using-with-huggingface-models-completely-locally"></a>

To run the script completely locally, run the following command:

```bash
python3 raft_local.py \
  --datapath "./sample_data/UC_Berkeley_short.pdf" \
  --output "./outputs" \
  --distractors 4 \
  --doc-type "pdf" \
  --chunk-size 512 \
  --questions 5 \
  --fast
```

## Step-by-Step Generation Process <a id="step-by-step-generation-process"></as>

Following the commands in the section [above](#examples), we shall walk through the process of generating the synthetic dataset step by step.

### 1. Chunk Generation <a id="1-chunk-generation"></a>

RAFT takes in a document of PDF type from the specified path and breaks the text up into chunk documents of size 512 tokens. A sample chunk is as follows:

```python
"[CLS] United States of America Flag Coat of arms Motto : \" In God We Trust \" [ 1 ] Other traditional mottos : [ 2 ] \" E pluribus unum \" ( Latin ) \" Out of many, one \" \" Annuit cœptis \" ( Latin ) \" Providence favors our undertakings \" \" Novus ordo seclorum \" ( Latin ) \" New order of the ages \" Anthem : \" The Star - Spangled Banner \" [ 3 ] United States The United States of America ( USA or U. S. A. ), commonly know n as the United States ( US or U. S. ) or America, is a country primarily located in North America, between Canada and Mexico. It is a liberal democracy and republic of 50 federated states, a federal capital district ( Washington, D. C. ), and 326 Indian reservations that overlap with state boundaries. Outside the union of states, it asserts sovereignty over five major unincorporated island territories and various uninhabited islands. [ i ] The country has the world\'s third - largest land area, [ c ] largest maritime exclusive economic zone, and the third - largest population ( over 334 million ). [ j ] The federal government uses a presidential system with three separate branches : legislative, executive, and judicial. American territory was first settled by Paleo - Indians who migrated across the Bering land bridge over 12, 000 years ago. Colonization by the British began in 1607. Thirteen colonies eventually rebelled against the British Crown over taxation and political representation, declaring independence on July 4, 1776. Their victory in the American Revolutionary War ( 1775 – 83 ) resulted in a confederation of states before the U. S. Constitution and Bill of Rights were ratified. The young nation continued to acquire neighbor ing territories and spanned North America by the late 1840s. Longstanding disagreements over slavery led to the secession of the southern Confederate States of America, which were defeated by the remaining Union in the American Civil War ( 1861 – 65 ). Slavery was abolished, but discriminatory laws persisted in the South. By 1900, rapid industrialization established the United States as a great power and the world\'s largest economy. Following the Japanese attack on Pearl Harbor in December 1941, the United States joined the Allies of World War II. After their victory, it competed against the Soviet Union for dominance in nuclear and conventional"
```

### 2. Question and Answer Generation <a id="2-question-and-answer-generation"></a>

RAFT then uses GPT-4 to generate 5 questions per chunk as well as the label (answer) for each question. Proceeding with the previous example chunk, we might get the following sample questions and answers:

**Questions:**

```python
[
  "What is the official motto of the United States of America?",
  "How many states are there in the United States of America?",
  "Which territories does the United States claim sovereignty over, outside the union of states?",
  "When did the thirteen colonies declare independence from the British Crown?",
  "What caused the sucession of the southern Confederate States of America?",
]
```

**Answers:**

```python
[
  "'In God We Trust'",
  "50 federated states",
  "Five major unincorporated island territories.",
  "July 4, 1776",
  "Disagreements over slavery",
]
```

### 3. Append Distractor Documents <a id="3-append-distractor-documents"></a>

For each question-answer pair, append 4 randomly selected chunks as distractor documents to form the `{ Q, A, D }` triplet. Proceeding with the current example, a `{ Q, A, D }` triplet, or one datapoint, would look like:

```python
{
  "id": "seed_task_0",
  "type": "general",
  "question": "What is the official motto of the United States of America?",
  "context": {
    "sentences": [
      [
        "the Gulf of Mexico are prone to hurricanes, ... and enforces the Act. [ 189 ] As of 2022, the U. S",
        "energy from fossil fuel and the largest ... there are 19, 969 airports in the U. S., of which 5, 193 are designated",
        'weaponry, ideology, and international i ... Council. The first documentary evidence of the phrase " United States',
        "[CLS] United States of America Flag Coat of arms ... dominance in nuclear and conventional",
        "##om ic soft pow er. [ 405 ] [ 406 ] Nearly all present ... in the United States are advanced by global standards.",
      ]
    ],
    "title": [
      [
        "placeholder_title",
        "placeholder_title",
        "placeholder_title",
        "placeholder_title",
        "placeholder_title",
      ]
    ],
  },
  "oracle_context": "[CLS] United States of America Flag Coat of arms ... dominance in nuclear and conventional",
  "cot_answer": "To answer the question, we need to identify the ... \n\n<ANSWER>: 'In God We Trust'",
  "instruction": "<DOCUMENT> ... <\/DOCUMENT>\nWhat is the official motto of the United States of America?",
}
```

### 4. Generate and Save the Dataset <a id="4-generate-and-save-the-dataset"></a>

RAFT repeats [steps 2](#2-question-and-answer-generation) and [3](#3-append-distractor-documents) for each chunk and saves the dataset to the path specified by the `--output` argument.

### 5. Dataset Conversion <a id="5-dataset-conversion"></a>

Next, we convert the dataset to the format expected for fine-tuning. For instance, you need to convert the dataset to the format expected for fine-tuning a `completion` model in Azure with the following command:

```bash
python3 format.py \
  --input "./output/data-00000-of-00001.arrow" \
  --output "output.completion.jsonl" \
  --output-format "completion"
```

> [!NOTE]
> The [`format.py`](format.py) script also has its own arguments. Refer to it for more information or simply run `python format.py --help`.

```bash
python format.py --help
usage: format.py [-h] --input INPUT [--input-type {arrow,jsonl}] --output OUTPUT --output-format {hf,completion,chat,eval} [--output-type {parquet,jsonl}]
                 [--output-chat-system-prompt OUTPUT_CHAT_SYSTEM_PROMPT] [--output-completion-prompt-column OUTPUT_COMPLETION_PROMPT_COLUMN]
                 [--output-completion-completion-column OUTPUT_COMPLETION_COMPLETION_COLUMN] [--output-completion-stop OUTPUT_COMPLETION_STOP]

options:
  -h, --help            show this help message and exit
  --input INPUT         Input HuggingFace dataset file (default: None)
  --input-type {arrow,jsonl}
                        Format of the input dataset. Defaults to arrow (default: arrow)
  --output OUTPUT       Output directory to save the dataset to (default: None)
  --output-format {hf,completion,chat,eval}
                        Format to convert the dataset to (default: None)
  --output-type {parquet,jsonl}
                        Type to export the dataset to. Defaults to jsonl. (default: jsonl)
  --output-chat-system-prompt OUTPUT_CHAT_SYSTEM_PROMPT
                        The system prompt to use when the output format is chat (default: The following is a conversation with an AI assistant. The
                        assistant is helpful, clever, friendly and gives concise and accurate answers.)
  --output-completion-prompt-column OUTPUT_COMPLETION_PROMPT_COLUMN
                        The prompt column name to use for the completion format (default: prompt)
  --output-completion-completion-column OUTPUT_COMPLETION_COMPLETION_COLUMN
                        The completion column name to use for the completion format (default: completion)
  --output-completion-stop OUTPUT_COMPLETION_STOP
                        The stop keyword to use for the completion format (default: <STOP>)
```

> [!NOTE]
> If fine-tuning a chat model, then you need to use `--output-format chat` and optionally add the `--output-chat-system-prompt` parameter to configure the system prompt included in the dataset.

### 6. Fine-tuning on Microsoft AI Studio <a id="6-fine-tuning-on-microsoft-ai-studio"></a>

Once the dataset is prepared, follow the instructions in this [guide](azure-ai-studio-ft/howto.md) to fine-tune and deploy your own RAFT model.

> [!IMPORTANT]
> Make sure to use `prompt` as input and `completion` as output when fine-tuning a `completion` model and the `messages` column as input when fine-tuning a `chat` model.

### 7. Evaluate RAFT Model <a id="7-evaluate-raft-model"></a>

After deploying your model in Azure AI Studio, run the command below to evaluate the RAFT model. Make sure to fill in the `BASE_URL`, `API_KEY` and `MODEL_NAME` in the `.env`. These values can be found in Azure AI Studio.

```bash
python3 eval.py \
  --question-file "YOUR_EVAL_FILE" \
  --answer-file <YOUR_ANSWER_FILE>
```

The `YOUR_EVAL_FILE` should be JSONL file type with the following format:

```python
{
  "instruction": "<DOCUMENT> ... </DOCUMENT>\n<DOCUMENT> ... </DOCUMENT> ...\n{question}",
  "gold_answer": "{answer}"
}
```
