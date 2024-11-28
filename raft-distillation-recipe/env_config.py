import contextlib
import logging
import os

logger = logging.getLogger("env_config")


env_prefix_whitelist = ["OPENAI", "AZURE_OPENAI"]


def _obfuscate(value: str) -> str:
    """
    Obfuscate the value if it's a credential like a key, secret or token.

    Args:
        value (str): The value to obfuscate.

    Returns:
        str: The obfuscated value.
    """
    l = len(value)
    return "*" * (l - 4) + value[-4:]


def _log_env(use_prefix: str, config: dict):
    """
    Logs the resolved environment variables.

    Args:
        use_prefix (str): The prefix used to process the environment variables.
        config (dict): The config dictionary containing the processed environment variables.
    """
    log_prefix = f"'{use_prefix}'" if use_prefix else "no"
    logger.info(f"Resolved OpenAI environment variables with {log_prefix} prefix.")
    for key, value in config.items():
        if any(cred in key for cred in ["KEY", "SECRET", "TOKEN"]):
            value = _obfuscate(value)
        logger.info(f" - {key}={value}")


def read_env_config(use_prefix: str, env: dict = os.environ) -> dict:
    """
    Reads environment variables, processes them and returns a dictionary
    of the processed environment variables.

    Args:
        use_prefix (str): The prefix of the environment variables to process.
        env (dict, optional): The environment variables to process. Defaults to os.environ.

    Returns:
        dict: The config dictionary containing the processed environment variables.
    """
    config = {}
    for prefix in [None, use_prefix]:
        read_env_config_prefixed(prefix, config, env)
    _log_env(use_prefix, config)
    return config


def read_env_config_prefixed(prefix: str | None, config: dict, env: dict):
    """
    Reads environment variables, processes them and adds them to the config dictionary.

    Args:
        prefix (str | None): The prefix of the environment variables to process.
        config (dict): The config dictionary to add the processed environment variables to.
        env (dict): The environment variables to process.
    """
    prefix = format_prefix(prefix)
    # Loop through the environment variables
    for key in env:
        # For each of the whitelisted prefix, (i.e. OPENAI, AZURE_OPENAI),
        for env_prefix in env_prefix_whitelist:
            # Construct the combined prefix
            key_prefix = f"{prefix}{format_prefix(env_prefix)}"
            # Then check if any of the environment variables starts with the combined prefix
            if key.startswith(key_prefix):
                # If found, strip the JUST the prefix from the key (leaving the whitelisted prefix) and store in config
                stripped_key = key.removeprefix(prefix)
                config[stripped_key] = env[key]


def format_prefix(prefix: str | None) -> str:
    """
    Formats the prefix to be used in the environment variables.

    Args:
        prefix (str | None): The prefix.

    Returns:
        str: The formatted prefix to be used in the environment variables.
    """
    if prefix is not None and len(prefix) > 0 and not prefix.endswith("_"):
        return f"{prefix}_"
    elif prefix is None:
        return ""
    return prefix


@contextlib.contextmanager
def set_env(**kwargs):
    """
    Temporarily sets the processed environment variables.
    WARNING: This is not thread safe as the environment is updated for the whole process.

    >>> with set_env(PLUGINS_DIR='test/plugins'):
    ...   "PLUGINS_DIR" in os.environ
    True

    >>> "PLUGINS_DIR" in os.environ
    False
    """
    old_environ = dict(os.environ)
    os.environ.update(kwargs)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)
