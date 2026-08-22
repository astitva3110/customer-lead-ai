"""Background writer so chat-history INSERT does not block RAG or the agent."""

from __future__ import annotations

import copy
import logging
import queue
import threading
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

_QUEUE_MAX = 256
_queue: queue.Queue[Callable[[], None]] = queue.Queue(maxsize=_QUEUE_MAX)
_worker_started = False
_lock = threading.Lock()


def submit_trace_write(write: Callable[[Any], None], trace: Any) -> None:
    """Return immediately. Drop the write if the queue is full rather than block."""
    _ensure_worker()
    snapshot = copy.deepcopy(trace)

    def job() -> None:
        write(snapshot)

    try:
        _queue.put_nowait(job)
    except queue.Full:
        logger.error("chat trace persist queue is full; dropping write so the chat path stays unblocked")


def _ensure_worker() -> None:
    global _worker_started
    with _lock:
        if _worker_started:
            return
        thread = threading.Thread(target=_run_worker, name="chat-trace-writer", daemon=True)
        thread.start()
        _worker_started = True


def _run_worker() -> None:
    while True:
        job = _queue.get()
        try:
            job()
        except Exception:
            logger.exception("chat trace persist failed")
        finally:
            _queue.task_done()
