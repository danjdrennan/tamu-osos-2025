from __future__ import annotations

import json
import hashlib
import logging
import os
from pathlib import Path
import subprocess
from logging import Logger
from typing import TYPE_CHECKING, Any
from enum import Enum

if TYPE_CHECKING:
    from hashlib import _Hash


class LoggingLevel(Enum):
    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0


def hash_git_info(hash_fn: _Hash, logger: Logger | None = None) -> dict[str, bytes]:
    """Collect Git repository information.

    Args:
        hash_fn: A hash function to incrementally hash results.
        logger (optional): Logger instance for recording the collected info.

    Returns:
        Git_info (dict[str, bytes]): A dict containing Git info with bytes values:
        - 'branch': Current branch name
        - 'commit_id': Current HEAD commit hash
        - 'staged_changes': Diff of staged changes
        - 'unstaged_changes': Diff of unstaged changes

    Raises:
        subprocess.CalledProcessError: If any Git command fails
            (e.g., not in a Git repo, Git not installed, permission issues).
        FileNotFoundError: If Git is not found in PATH.
    """
    check_output = subprocess.check_output
    git_info = {
        "branch": check_output("Git rev-parse --abbrev-head HEAD".split()).strip(),
        "commit_id": check_output("Git rev-parse HEAD".split()).strip(),
        "staged_changes": check_output("Git diff --cached".split()).strip(),
        "unstanged_changes": check_output("Git diff".split()).strip(),
    }

    Git_info_dump = json.dumps(git_info, sort_keys=True).encode()
    hash_fn.update(Git_info_dump)

    if logger:
        logger.log(
            logging.INFO,
            f"hashed {git_info=}\nwith incremental hash {hash_fn.hexdigest()}",
        )

    return git_info


def hash_file(
    file_name: str | os.PathLike | Path,
    hash_fn: _Hash,
    logger: Logger | None = None,
    buffer_size: int | None = None,
) -> tuple[str, str]:
    """Hashes file contents using global and local loggers.

    The hash is computed once with the `hash_fn` passed to the function and separately
    with local sha256 function that gets logged. The reason for global-local hashing is
    so that incremental checks of the config can be made.

    Hashing hashes the file name first and then performs a buffered read over the file
    with `buffer_size`.

    Args:
        file_name: A file name to hash.
        hash_fn: A global hash function to incrementally hash data with.
        logger (optional): A logger to log the hashes to.
        buffer_size (default=None): The buffer size to read chunks of a file with.

    Returns:
        local_file_hash: The file hash obtained via the local hash function.
        global_file_hash: The file hash obtained via the global hash function.

    Raises:
        FileNotFoundError: If the file is not found on the file system.
        OSError: Raised if opening the file fails.
    """
    local_hash = hashlib.sha256(usedforsecurity=False)

    file_path = Path(file_name)

    if not file_path.exists():
        raise FileNotFoundError(f"{file_name} not found.")

    file_name_bytes = str(file_name).encode()

    local_hash.update(file_name_bytes)
    hash_fn.update(file_name_bytes)

    with open(file_path, "rb") as f:
        chunk = f.read(buffer_size)
        while chunk:
            local_hash.update(chunk)
            hash_fn.update(chunk)

    if logger:
        logger.log(logging.INFO, f"local hash: {local_hash.hexdigest()}")
        logger.log(logging.INFO, f"global hash: {hash_fn.hexdigest()}")

    return local_hash.hexdigest(), hash_fn.hexdigest()


def hash_config(
    config: dict[str, Any], hash_fn: _Hash, logger: Logger | None = None
) -> str:
    sorted_config_bytes = json.dumps(config, sort_keys=True).encode()

    hash_fn.update(sorted_config_bytes)

    if logger:
        logger.log(
            logging.INFO,
            (
                f"config {config} encoded as {sorted_config_bytes} "
                f"and hashed with value {hash_fn.hexdigest()}",
            ),
        )

    return hash_fn.hexdigest()


def hash_config_and_data(
    config: dict[str, Any],
    hash_git: bool = False,
    hash_fn: _Hash | None = None,
    logger: logging.Logger | None = None,
    logging_level: LoggingLevel = LoggingLevel.INFO,
    data_files: list[str | os.PathLike | Path] | None = None,
    buffer_size: int | None = None,
) -> dict:
    """Hash config of CLI args and (optionally) data files.

    Args:
        config: A tree of cli arguments, constants, and data used to run an application.
        logging_level (default=LoggingLevel.INFO): The logging level.
            (see LoggingLevel, which copies the table from the logging standard library)
        hash_git_info (default=False): Hashes Git info for full reproducibility.
            When true, this gets and hashes the
            - branch id,
            - commit id,
            - staged and unstaged patches.
        hash_fn (default=None): A hash function to hash results with.
            If no function is provided, the hashlib.sha256 function is used.
        logger (default=None): A logger instance for recording events.
            If no logger is provided then one with this scope is created with
            LoggingLevel. Note that loggers follow a singleton pattern, so the events
            will log to the same place as a global logger.
        data_files (default=None): A list of files to hash and log results for.
        buffer_size (default=None): The buffer size for reading chunks of files from.

    Returns:
        A table (key-value store) of incremental hashing results for each argument of
        this function (config, Git info, data files).

    Raises:
        subprocess.CalledProcessError: If any Git command fails
            (e.g., not in a Git repo, Git not installed, permission issues).
        FileNotFoundError: If a Git executable or data file is not found.
        OSError: Raised if opening the file fails.
    """
    if not hash_fn:
        hash_fn = hashlib.sha256(usedforsecurity=False)
    if not logger:
        logger = logging.getLogger(name="hash_config_and_data")

    hashed_config = hash_config(config, hash_fn, logger)

    if data_files:
        hashed_files = dict()
        for data_file in data_files:
            hashed_files[data_file] = hash_file(data_file, hash_fn, logger, buffer_size)
    else:
        hashed_files = None

    hashed_git_info = hash_git_info(hash_fn, logger) if hash_git else None

    digest = hash_fn.hexdigest()
    hash_results = dict(
        digest=digest,
        config=hashed_config,
        data=hashed_files,
        git_info=hashed_git_info,
    )

    logging.log(logging_level.value, f"completed hashing with digest {digest}")

    return hash_results
