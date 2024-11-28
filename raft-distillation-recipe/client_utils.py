from __future__ import annotations

import logging
import os
import time
from abc import ABC
from threading import Lock
from typing import Any

from azure.identity import (
    DefaultAzureCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings
from openai import AzureOpenAI, OpenAI
from openai.types.chat.chat_completion import ChatCompletion

from env_config import read_env_config, set_env

logger = logging.getLogger("client_utils")


def build_openai_client(**kwargs: Any) -> OpenAI | AzureOpenAI:
    """
    Builds a OpenAI completions model client.

    Returns:
        OpenAI | AzureOpenAI: The OpenAI completions model client.
    """
    kwargs = _remove_empty_values(kwargs)
    use_prefix = kwargs["prefix"]
    env = read_env_config(use_prefix)
    with set_env(**env):
        if is_azure():
            auth_args = _get_azure_auth_client_args()
            client = AzureOpenAI(**auth_args, **kwargs)
        else:
            client = OpenAI(**kwargs)
    return client


def build_langchain_embeddings(
    **kwargs: Any,
) -> OpenAIEmbeddings | AzureOpenAIEmbeddings:
    """
    Builds a LangChain embeddings model client.

    Returns:
        OpenAIEmbeddings | AzureOpenAIEmbeddings: The LangChain embeddings model client.
    """
    kwargs = _remove_empty_values(kwargs)
    use_prefix = kwargs["prefix"]
    env = read_env_config(use_prefix)
    with set_env(**env):
        if is_azure():
            auth_args = _get_azure_auth_client_args()
            client = AzureOpenAIEmbeddings(**auth_args, **kwargs)
        else:
            client = OpenAIEmbeddings(**kwargs)
    return client


def _remove_empty_values(kwargs: dict) -> dict:
    """
    Removes empty keyword argument values from a dictionary.

    Args:
        kwargs (dict): A dictionary of keyword arguments.

    Returns:
        dict: The dictionary with empty keyword argument values removed.
    """
    return {k: v for k, v in kwargs.items() if v is not None}


def _get_azure_auth_client_args() -> dict:
    """
    Handle Azure OpenAI Keyless, Managed Identity and Key-based authentication.
    See: https://techcommunity.microsoft.com/t5/microsoft-developer-community/using-keyless-authentication-with-azure-openai/ba-p/4111521

    Returns:
        dict: The authentication arguments for Azure OpenAI.
    """
    client_args = {}
    if os.getenv("AZURE_OPENAI_KEY"):
        logger.info("Using Azure OpenAI Key-based authentication.")
        client_args["api_key"] = os.getenv("AZURE_OPENAI_KEY")
    else:
        if client_id := os.getenv("AZURE_OPENAI_CLIENT_ID"):
            # Authenticate using a user-assigned managed identity on Azure
            logger.info("Using Azure OpenAI Managed Identity Keyless authentication")
            azure_credential = ManagedIdentityCredential(client_id=client_id)
        else:
            # Authenticate using the default Azure credential chain
            logger.info(
                "Using Azure OpenAI Default Azure Credential Keyless authentication"
            )
            azure_credential = DefaultAzureCredential()

        client_args["azure_ad_token_provider"] = get_bearer_token_provider(
            azure_credential, "https://cognitiveservices.azure.com/.default"
        )

    client_args["api_version"] = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01")
    client_args["azure_endpoint"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    client_args["azure_deployment"] = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    return client_args


def is_azure() -> bool:
    """
    Checks if the OpenAI or Azure OpenAI environment variables are set.

    Returns:
        bool: Returns True if the Azure OpenAI environment variables are set,
            False otherwise.
    """
    azure = (
        "AZURE_OPENAI_ENDPOINT" in os.environ
        or "AZURE_OPENAI_KEY" in os.environ
        or "AZURE_OPENAI_AD_TOKEN" in os.environ
    )
    if azure:
        logger.debug("Using Azure OpenAI environment variables.")
    else:
        logger.debug("Using OpenAI environment variables.")
    return azure


def safe_min(a: Any, b: Any) -> Any:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def safe_max(a: Any, b: Any) -> Any:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


class UsageStats:
    def __init__(self) -> None:
        self.start = time.time()
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.total_tokens = 0
        self.end = None
        self.duration = 0
        self.calls = 0

    def __add__(self, other: UsageStats) -> UsageStats:
        stats = UsageStats()
        stats.start = safe_min(self.start, other.start)
        stats.end = safe_max(self.end, other.end)
        stats.completion_tokens = self.completion_tokens + other.completion_tokens
        stats.prompt_tokens = self.prompt_tokens + other.prompt_tokens
        stats.total_tokens = self.total_tokens + other.total_tokens
        stats.duration = self.duration + other.duration
        stats.calls = self.calls + other.calls
        return stats


class StatsCompleter(ABC):
    def __init__(self, create_func: OpenAI | AzureOpenAI):
        self.create_func = create_func
        self.stats = None
        self.lock = Lock()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        response: ChatCompletion = self.create_func(*args, **kwargs)
        self.lock.acquire()
        try:
            # If it doesn't already exist, create it
            if not self.stats:
                self.stats = UsageStats()
            self.stats.completion_tokens += response.usage.completion_tokens
            self.stats.prompt_tokens += response.usage.prompt_tokens
            self.stats.total_tokens += response.usage.total_tokens
            self.stats.calls += 1
            return response
        finally:
            self.lock.release()

    def get_stats_and_reset(self) -> UsageStats:
        self.lock.acquire()
        try:
            end = time.time()
            stats = self.stats
            if stats:
                stats.end = end
                stats.duration = end - self.stats.start
                self.stats = None
            return stats
        finally:
            self.lock.release()


class ChatCompleter(StatsCompleter):
    def __init__(self, client: OpenAI | AzureOpenAI):
        super().__init__(client.chat.completions.create)


class CompletionsCompleter(StatsCompleter):
    def __init__(self, client: OpenAI | AzureOpenAI):
        super().__init__(client.completions.create)
