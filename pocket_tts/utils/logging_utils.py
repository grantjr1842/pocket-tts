import logging
import sys
from contextlib import contextmanager

import structlog


class PocketTTSFilter(logging.Filter):
    def filter(self, record):
        return record.name.startswith("pocket_tts")


def configure_structlog(log_level):
    """Configure structlog processors and formatting."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


@contextmanager
def enable_logging(library_name, level):
    # Configure structlog
    configure_structlog(level)

    # Get the specific logger and its parent
    logger = logging.getLogger(library_name)
    parent_logger = logging.getLogger("pocket_tts")

    # Store original configuration
    old_level = logger.level
    old_parent_level = parent_logger.level
    old_handlers = parent_logger.handlers.copy()

    # Configure logging format for pocket_tts logger
    parent_logger.setLevel(level)

    # Clear existing handlers and add our custom formatter with filter
    parent_logger.handlers.clear()

    # Create a handler that redirects standard logging to structlog
    # For now, we'll keep the StreamHandler but format it to be more structured-friendly or just simple text
    # Since we are introducing structlog, we might want to use it primarily.
    # However, to be less invasive, let's keep the standard handler for the CLI output for now,
    # but ensure the PerformanceMonitor uses structlog directly.

    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    handler.addFilter(PocketTTSFilter())
    parent_logger.addHandler(handler)
    parent_logger.propagate = False

    try:
        yield logger
    finally:
        # Restore original configuration
        logger.setLevel(old_level)
        parent_logger.setLevel(old_parent_level)
        parent_logger.handlers.clear()
        for h in old_handlers:
            parent_logger.addHandler(h)
