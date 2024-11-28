import logging
import logging.config
import os
from types import ModuleType

import yaml


def log_setup():
    """
    Set up basic console logging. Root logger level can be set
    with ROOT_LOG_LEVEL environment variable.
    """
    # Load the configuration file if LOGGING_CONFIG is not set
    conf_path = os.getenv("LOGGING_CONFIG", None)
    if conf_path is None:
        conf_path = os.path.dirname(__file__) + "/logging.yaml"
    with open(conf_path, "rt") as f:
        config = yaml.safe_load(f.read())

    # Configure the logging module with the configuration file
    logging.config.dictConfig(config)

    install_default_record_field(logging, "progress", "")


def install_default_record_field(logging: ModuleType, field: str, value: str):
    """
    Wraps the log record factory to add a default progress field value.
    Required to avoid a KeyError when the progress field is not set.
    Such as when logging from a different thread

    Args:
        logging (ModuleType): The logging module.
        field (str): The name of the field to add.
        value (str): The default value to add to the field.
    """

    # Retrieves the current log record factory (a function used to create log records)
    # to ensure the original behavior is preserved and can be extended rather than
    # replaced entirely.
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        if not hasattr(record, field):
            record.progress = value
        return record

    # Replaces the default log record factory with the custom factory and ensures all
    # log records created after this point will have the `progress` field, even if the
    # log message or other logging context doesn't explicitly set it.
    logging.setLogRecordFactory(record_factory)
