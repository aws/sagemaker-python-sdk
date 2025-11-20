"""Module for pulling logs from container"""

from __future__ import absolute_import
import logging
from threading import Thread
import queue
from datetime import datetime

from sagemaker.serve.utils.exceptions import (
    LocalModelLoadException,
    LocalModelOutOfMemoryException,
    LocalModelInvocationException,
    SkipTuningComboException,
)

logger = logging.getLogger(__name__)


def _get_logs(generator, logs, until):
    """Placeholder docstring"""
    now = datetime.now()
    try:
        for next_entry in generator:
            logs.put(next_entry, block=True, timeout=(until - now).total_seconds())

            now = datetime.now()
            if until < now:
                break
        logger.debug("Container logging done. All container logs processed.")
    except (StopIteration, ValueError):
        logger.debug("Container logging stopped. All container logs processed.")


def pull_logs(generator, stop, until, final_pull):
    """Placeholder docstring"""
    now = datetime.now()
    if until < now:
        return

    logs = queue.Queue(maxsize=10)
    log_puller = Thread(
        target=_get_logs,
        args=(
            generator,
            logs,
            until,
        ),
    )
    log_puller.start()

    while True:
        try:
            top = logs.get(block=True, timeout=(until - now).total_seconds())
            logger.debug(top)

            if "[INFO ]" in top and "OutOfMemoryError" in top:
                raise LocalModelOutOfMemoryException(top)
            if "CUDA out of memory. Tried to allocate" in top:
                raise LocalModelOutOfMemoryException(top)
            if "ai.djl.engine.EngineException: OOM" in top:
                raise LocalModelOutOfMemoryException(top)
            if "4xx.Count:1" in top or "5xx.Count:1" in top:
                raise LocalModelInvocationException(top)
            if "[ERROR]" in top or "Failed register workflow" in top:
                raise LocalModelLoadException(top)
            if "Address already in use" in top:
                raise LocalModelLoadException(top)
            if "not compatible with sharding" in top:
                raise SkipTuningComboException(top)
        except (
            LocalModelLoadException,
            LocalModelOutOfMemoryException,
            LocalModelInvocationException,
        ) as e:
            stop()
            log_puller.join(timeout=5)
            raise e
        except queue.Empty:
            now = datetime.now()
            if until < now:
                if not final_pull:
                    return

                stop()
                # allow logging thread some time to cleanup
                log_puller.join(timeout=5)
                if log_puller.is_alive():
                    raise Exception("Logging thread not terminating")
                return
