import logging
import sys
from typing import Any

import structlog

from lexrag.config import settings

def _setup_logging() -> None:
    # Set the root logger's log level
    log_level = getattr(logging, settings.app_log_level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=log_level)

    # Configure structlog processors
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Use JSON rendering in production, rich colored console in development
    if settings.app_environment.lower() == "production":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

# Initialize logging setup immediately
_setup_logging()

def get_logger(name: str) -> Any:
    """Get a structlog logger instance by name.
    
    Args:
        name: The name of the logger (usually __name__).
        
    Returns:
        A configured structlog logger.
    """
    return structlog.get_logger(name)
