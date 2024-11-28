import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from env_config import read_env_config


def test_openai_simple():
    """Simple OpenAI only environment variables. Non-related environment variables are ignored."""

    env = {
        "HOSTNAME": "localhost",
        "OPENAI_API_KEY": "<API_KEY_1>",
    }

    config = read_env_config("COMPLETION", env)

    assert "HOSTNAME" not in config
    assert config["OPENAI_API_KEY"] == "<API_KEY_1>"


def test_azure_simple():
    """Simple Azure OpenAI only environment variables. Non-related environment variables are ignored."""

    env = {
        "HOSTNAME": "localhost",
        "AZURE_OPENAI_ENDPOINT": "<ENDPOINT_1>",
        "AZURE_OPENAI_API_KEY": "<API_KEY_1>",
        "OPENAI_API_VERSION": "<VERSION_1>",
    }

    config = read_env_config("COMPLETION", env)

    assert "HOSTNAME" not in config
    assert config["AZURE_OPENAI_ENDPOINT"] == "<ENDPOINT_1>"
    assert config["AZURE_OPENAI_API_KEY"] == "<API_KEY_1>"
    assert config["OPENAI_API_VERSION"] == "<VERSION_1>"


def test_azure_override():
    """
    Tests that the COMPLETION configuration overrides the base configuration and that
    the EMBEDDING configuration doesn't interfere and that non-related environment
    variables are ignored.
    """

    env = {
        "HOSTNAME": "localhost",  # unrelated
        "AZURE_OPENAI_ENDPOINT": "<ENDPOINT_1>",
        "AZURE_OPENAI_API_KEY": "<API_KEY_1>",
        "OPENAI_API_VERSION": "<VERSION_1>",
        "COMPLETION_AZURE_OPENAI_ENDPOINT": "<ENDPOINT_2>",
        "COMPLETION_AZURE_OPENAI_API_KEY": "<API_KEY_2>",
        "EMBEDDING_AZURE_OPENAI_ENDPOINT": "<ENDPOINT_3>",
        "EMBEDDING_AZURE_OPENAI_API_KEY": "<API_KEY_3>",
    }

    comp_config = read_env_config("COMPLETION", env)
    assert "HOSTNAME" not in comp_config
    assert comp_config["AZURE_OPENAI_ENDPOINT"] == "<ENDPOINT_2>"
    assert comp_config["AZURE_OPENAI_API_KEY"] == "<API_KEY_2>"
    assert comp_config["OPENAI_API_VERSION"] == "<VERSION_1>"

    emb_config = read_env_config("EMBEDDING", env)
    assert "HOSTNAME" not in emb_config
    assert emb_config["AZURE_OPENAI_ENDPOINT"] == "<ENDPOINT_3>"
    assert emb_config["AZURE_OPENAI_API_KEY"] == "<API_KEY_3>"
    assert emb_config["OPENAI_API_VERSION"] == "<VERSION_1>"


def test_openai_override():
    """
    Tests that the COMPLETION configuration overrides the base configuration and that
    the EMBEDDING configuration doesn't interfere and that non-related environment
    variables are ignored.
    """

    env = {
        "HOSTNAME": "localhost",
        "OPENAI_ENDPOINT": "<ENDPOINT_1>",
        "OPENAI_API_KEY": "<API_KEY_1>",
        "OPENAI_API_VERSION": "<VERSION_1>",
        "COMPLETION_OPENAI_ENDPOINT": "<ENDPOINT_2>",
        "COMPLETION_OPENAI_API_KEY": "<API_KEY_2>",
        "EMBEDDING_OPENAI_ENDPOINT": "<ENDPOINT_3>",
        "EMBEDDING_OPENAI_API_KEY": "<API_KEY_3>",
    }

    comp_config = read_env_config("COMPLETION", env)
    assert "HOSTNAME" not in comp_config
    assert comp_config["OPENAI_ENDPOINT"] == "<ENDPOINT_2>"
    assert comp_config["OPENAI_API_KEY"] == "<API_KEY_2>"
    assert comp_config["OPENAI_API_VERSION"] == "<VERSION_1>"

    emb_config = read_env_config("EMBEDDING", env)
    assert "HOSTNAME" not in emb_config
    assert emb_config["OPENAI_ENDPOINT"] == "<ENDPOINT_3>"
    assert emb_config["OPENAI_API_KEY"] == "<API_KEY_3>"
    assert emb_config["OPENAI_API_VERSION"] == "<VERSION_1>"
