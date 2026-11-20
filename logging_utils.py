"""
Shared logging setup used by both the CLI (main.py --verbose, prints to the terminal) and the
Streamlit UI (app.py's live "Run log" box). Keeping the format/level logic in one place means
both surfaces show the same level of detail — the nodes/, sources/, llm_client.py, embeddings.py
modules just call logger.info(...) and don't care which surface is listening.
"""
import logging
import queue
import re

LOG_FORMAT = "%(asctime)s  %(levelname)-7s %(name)-16s  %(message)s"
DATE_FORMAT = "%H:%M:%S"

# Libraries that emit very chatty DEBUG-level wire tracing (one line per TCP event / HTTP frame)
# that's almost never useful for tracking what the *agent* is doing — it drowns out our own
# nodes/llm_client/sources/embeddings messages. Capped at INFO regardless of the app's own log
# level, so "show everything" means everything about the agent's own steps, not raw socket
# chatter. If you're debugging an actual network-layer issue, temporarily comment this out.
_NOISY_LOGGERS = ["httpx", "httpcore", "urllib3", "sentence_transformers", "huggingface_hub", "filelock"]


# Secrets in log lines. httpx logs every request URL at INFO ("HTTP Request: GET https://api.openalex.org/
# works?...&api_key=..."), and the Streamlit "Full run log" box + its download button showed those keys.
# Query parameters that carry a key/token/contact address keep their name, lose their value.
_SECRET_PARAM_RE = re.compile(
    r"(?i)([?&;](?:api[_-]?key|apikey|key|token|access[_-]?token|auth|secret|password|mailto|email)=)[^&\s\"'#<>]+")
# Bare keys of the services this project uses (Anthropic, Langfuse) wherever they appear.
_SECRET_TOKEN_RE = re.compile(r"\b(?:sk-ant-[A-Za-z0-9_\-]{8,}|sk-lf-[A-Za-z0-9\-]{8,}|pk-lf-[A-Za-z0-9\-]{8,})")
SECRET_MASK = "***"


def mask_secrets(text: str) -> str:
    """'...&api_key=abc123&page=2' -> '...&api_key=***&page=2'. Idempotent."""
    if not text or not isinstance(text, str):
        return text
    text = _SECRET_PARAM_RE.sub(lambda m: m.group(1) + SECRET_MASK, text)
    return _SECRET_TOKEN_RE.sub(SECRET_MASK, text)


class SecretMaskingFilter(logging.Filter):
    """Rewrites a record's message with mask_secrets() before any handler formats it. Attached to
    the noisy third-party loggers themselves (so it applies whatever handler is listening) and to
    this project's own handlers (console + the UI's QueueLogHandler)."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        masked = mask_secrets(msg)
        if masked != msg:
            record.msg, record.args = masked, None
        if record.exc_info and not record.exc_text:
            record.exc_text = mask_secrets(logging.Formatter().formatException(record.exc_info))
        elif record.exc_text:
            record.exc_text = mask_secrets(record.exc_text)
        return True


_MASK_FILTER = SecretMaskingFilter()


def _add_mask_filter(target) -> None:
    if not any(isinstance(f, SecretMaskingFilter) for f in target.filters):
        target.addFilter(_MASK_FILTER)


def quiet_noisy_third_party_loggers() -> None:
    for name in _NOISY_LOGGERS:
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        _add_mask_filter(lg)


def configure_console_logging(verbose: bool = False) -> None:
    """Used by main.py. INFO shows every search query, paper count, LLM call and retry;
    WARNING (the default) shows only problems."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING,
                         format=LOG_FORMAT, datefmt=DATE_FORMAT)
    for h in logging.getLogger().handlers:
        _add_mask_filter(h)
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
        _add_mask_filter(self)   # the UI shows and offers these lines for download - never with a key

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.log_queue.put(self.format(record))
        except Exception:
            pass  # a logging failure must never take down the agent run
