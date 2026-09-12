"""
Shared logging setup used by both the CLI (main.py --verbose, prints to the terminal) and the
Streamlit UI (app.py's live "Run log" box). Keeping the format/level logic in one place means
both surfaces show the same level of detail — the nodes/, sources/, llm_client.py, embeddings.py
modules just call logger.info(...) and don't care which surface is listening.
"""
import logging
import queue

LOG_FORMAT = "%(asctime)s  %(levelname)-7s %(name)-16s  %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Libraries that emit very chatty DEBUG-level wire tracing (one line per TCP event / HTTP frame)
# that's almost never useful for tracking what the *agent* is doing — it drowns out our own
# nodes/llm_client/sources/embeddings messages. Capped at INFO regardless of the app's own log
# level, so "show everything" means everything about the agent's own steps, not raw socket
# chatter. If you're debugging an actual network-layer issue, temporarily comment this out.
_NOISY_LOGGERS = ["httpx", "httpcore", "urllib3", "sentence_transformers", "huggingface_hub", "filelock"]


def quiet_noisy_third_party_loggers() -> None:
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.INFO)


def configure_console_logging(verbose: bool = False) -> None:
    """Used by main.py. INFO shows every search query, paper count, LLM call and retry;
    WARNING (the default) shows only problems."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                         format=LOG_FORMAT, datefmt=DATE_FORMAT)
    quiet_noisy_third_party_loggers()


class QueueLogHandler(logging.Handler):
    """
    Pushes formatted log lines onto a queue.Queue instead of printing them.

    app.py runs the agent graph in a background thread (so the UI stays responsive) and attaches
    one of these to the root logger before starting it. The main Streamlit thread then polls the
    queue in a loop and renders new lines into the live log box — this is what makes the box
    update in real time instead of only after the whole run finishes.
    """

    def __init__(self, log_queue: "queue.Queue[str]"):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.log_queue.put(self.format(record))
        except Exception:
            pass  # a logging failure must never take down the agent run
