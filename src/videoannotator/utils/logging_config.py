"""Enhanced logging configuration for VideoAnnotator API Server.

Provides structured logging with file rotation, request tracking, and
comprehensive debugging support.
"""

import hashlib
import json
import logging
import logging.handlers
import os
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

LOG_DIR_ENV = "VIDEOANNOTATOR_LOG_DIR"


def _default_logs_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "videoannotator" / "logs"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / "videoannotator"
    # XDG puts logs under state, not data.
    base = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return base / "videoannotator" / "logs"


def logs_dir() -> Path:
    """The log directory, as an absolute path.

    `VIDEOANNOTATOR_LOG_DIR`, defaulting to the platform's per-user log location, so
    starting the server from another directory doesn't scatter logs.
    """
    configured = os.environ.get(LOG_DIR_ENV)
    root = Path(configured).expanduser() if configured else _default_logs_dir()
    return root.resolve()


_resolve_logs_dir = logs_dir


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "filename": record.filename,
            "line": record.lineno,
        }

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
                "exc_info",
                "exc_text",
                "stack_info",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry)


# The terminal says only what a person running the server needs: messages sent
# to this logger, and warnings and errors. Everything still goes to logs/.
CONSOLE_LOGGER = "videoannotator.console"
VERBOSE_ENV = "VIDEOANNOTATOR_VERBOSE"
_LIBRARY_LOGGERS = ("py", "torch", "ultralytics", "transformers", "whisper")


def key_fingerprint(token: str) -> str:
    """Identifies an API key in logs without revealing any of it."""
    return "key:" + hashlib.sha256(token.encode()).hexdigest()[:8]


def console_verbose() -> bool:
    """`videoannotator server --verbose`: every INFO line in the terminal, as before."""
    return os.environ.get(VERBOSE_ENV, "").lower() in ("1", "true", "yes")


def console() -> logging.Logger:
    """The logger for lines meant for the person at the terminal."""
    return logging.getLogger(CONSOLE_LOGGER)


class ConsoleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if console_verbose():
            return True
        if record.name == CONSOLE_LOGGER:
            return True
        # Library warnings (deprecations, CUDA notices) are for the log files.
        if record.name.split(".")[0] in _LIBRARY_LOGGERS:
            return record.levelno >= logging.ERROR
        return record.levelno >= logging.WARNING


class ConsoleFormatter(logging.Formatter):
    """`15:12:03 Started clip.mp4 …`, `15:12:03 [WARNING] …`; tracebacks stay in
    logs/errors.log unless --verbose."""

    def format(self, record: logging.LogRecord) -> str:
        when = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        tag = (
            "[ERROR] "
            if record.levelno >= logging.ERROR
            else "[WARNING] "
            if record.levelno >= logging.WARNING
            else ""
        )
        source = (
            f"{record.name}: "
            if console_verbose() and record.name != CONSOLE_LOGGER
            else ""
        )
        line = f"{when} {tag}{source}{record.getMessage()}"
        if record.exc_info:
            if console_verbose():
                line += "\n" + "".join(traceback.format_exception(*record.exc_info))
            else:
                line += " (details in logs/errors.log)"
        return line


class VideoAnnotatorLoggingConfig:
    """Configuration class for VideoAnnotator logging system."""

    def __init__(self, logs_dir: str | Path | None = None, log_level: str = "INFO"):
        """Initialize logging paths and default log level."""
        self.logs_dir = Path(logs_dir) if logs_dir is not None else _resolve_logs_dir()
        self.log_level = log_level.upper()
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Create log files
        self.api_log_file = self.logs_dir / "api_server.log"
        self.error_log_file = self.logs_dir / "errors.log"
        self.request_log_file = self.logs_dir / "api_requests.log"
        self.debug_log_file = self.logs_dir / "debug.log"

    def setup_logging(
        self, capture_warnings: bool = True, capture_stdstreams: bool = False
    ) -> dict[str, logging.Logger]:
        """Set up comprehensive logging system."""
        # Clear any existing handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set root level
        root_logger.setLevel(getattr(logging, self.log_level))

        loggers = {}

        # Optionally capture Python warnings and redirect them to the logging system
        if capture_warnings:
            # This routes warnings.warn(...) into the logging subsystem under 'py.warnings'
            logging.captureWarnings(True)

        # 1. Console Logger (human-readable)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.addFilter(ConsoleFilter())
        console_handler.setFormatter(ConsoleFormatter())

        # 2. Main API Server Logger (rotating file)
        api_handler = logging.handlers.RotatingFileHandler(
            self.api_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        api_handler.setLevel(getattr(logging, self.log_level))
        api_formatter = StructuredFormatter()
        api_handler.setFormatter(api_formatter)

        # 3. Error Logger (separate file for errors only)
        error_handler = logging.handlers.RotatingFileHandler(
            self.error_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=10,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = StructuredFormatter()
        error_handler.setFormatter(error_formatter)

        # 4. Request Logger (API request/response tracking)
        request_handler = logging.handlers.RotatingFileHandler(
            self.request_log_file,
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=3,
            encoding="utf-8",
        )
        request_handler.setLevel(logging.INFO)
        request_formatter = StructuredFormatter()
        request_handler.setFormatter(request_formatter)

        # 5. Debug Logger (verbose logging)
        if self.log_level == "DEBUG":
            debug_handler = logging.handlers.RotatingFileHandler(
                self.debug_log_file,
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=2,
                encoding="utf-8",
            )
            debug_handler.setLevel(logging.DEBUG)
            debug_formatter = StructuredFormatter()
            debug_handler.setFormatter(debug_formatter)

        # Configure specific loggers

        # Main API logger
        api_logger = logging.getLogger("videoannotator.api")
        # Clear existing handlers to prevent duplicates
        api_logger.handlers.clear()
        api_logger.addHandler(console_handler)
        api_logger.addHandler(api_handler)
        api_logger.addHandler(error_handler)
        loggers["api"] = api_logger

        terminal_logger = console()
        terminal_logger.handlers.clear()
        terminal_logger.addHandler(console_handler)
        terminal_logger.addHandler(api_handler)
        terminal_logger.setLevel(logging.INFO)
        terminal_logger.propagate = False
        loggers["console"] = terminal_logger

        # Request logger
        request_logger = logging.getLogger("videoannotator.requests")
        request_logger.addHandler(request_handler)
        request_logger.addHandler(error_handler)
        loggers["requests"] = request_logger

        # Pipeline logger
        pipeline_logger = logging.getLogger("videoannotator.pipelines")
        pipeline_logger.addHandler(api_handler)
        pipeline_logger.addHandler(error_handler)
        if self.log_level == "DEBUG":
            pipeline_logger.addHandler(debug_handler)
        loggers["pipelines"] = pipeline_logger

        # Database logger
        db_logger = logging.getLogger("videoannotator.database")
        db_logger.addHandler(api_handler)
        db_logger.addHandler(error_handler)
        loggers["database"] = db_logger

        # Debug logger
        if self.log_level == "DEBUG":
            debug_logger = logging.getLogger("videoannotator.debug")
            debug_logger.addHandler(debug_handler)
            debug_logger.addHandler(console_handler)
            loggers["debug"] = debug_logger

        # Attach handlers to important third-party loggers so their messages
        # (e.g., CUDA / PyTorch compatibility warnings) are captured.
        third_party_names = ["torch", "ultralytics", "transformers", "whisper"]
        for name in third_party_names:
            try:
                third_logger = logging.getLogger(name)
                # clear to avoid duplicate handlers
                third_logger.handlers.clear()
                third_logger.addHandler(console_handler)
                third_logger.addHandler(api_handler)
                third_logger.addHandler(error_handler)
                third_logger.setLevel(logging.WARNING)
            except Exception:
                # Non-fatal: continue
                pass

        # Configure other third-party logger level adjustments
        self._configure_third_party_loggers()

        # Route Python warnings (via 'py.warnings') into our handlers so they appear in logs
        if capture_warnings:
            py_warn_logger = logging.getLogger("py.warnings")
            py_warn_logger.handlers.clear()
            # Warnings are generally WARNING-level, attach console and api handlers
            py_warn_logger.addHandler(console_handler)
            py_warn_logger.addHandler(api_handler)
            py_warn_logger.addHandler(error_handler)
            py_warn_logger.setLevel(logging.WARNING)

        # Optionally redirect stdout/stderr into the logging system. Useful to capture
        # stray print() calls (e.g., legacy code printing CUDA status) into structured logs.
        if capture_stdstreams:

            class StreamToLogger:
                def __init__(self, logger, level=logging.INFO):
                    self.logger = logger
                    self.level = level
                    self._buffer = ""

                def write(self, message):
                    if message and message.strip():
                        # Preserve newlines in messages
                        for line in message.rstrip().splitlines():
                            self.logger.log(self.level, line)

                def flush(self):
                    pass

            sys.stdout = StreamToLogger(
                api_logger if "api" in locals() else logging.getLogger(), logging.INFO
            )
            sys.stderr = StreamToLogger(
                error_handler if False else logging.getLogger(), logging.ERROR
            )

        return loggers

    def _configure_third_party_loggers(self):
        """Configure third-party library logging levels."""
        import warnings

        # Filter specific deprecation warnings
        warnings.filterwarnings(
            "ignore",
            message=".*torchaudio._backend.list_audio_backends has been deprecated.*",
        )
        warnings.filterwarnings(
            "ignore", message=".*torchaudio._backend.utils.info has been deprecated.*"
        )
        warnings.filterwarnings(
            "ignore",
            message=".*torchaudio._backend.common.AudioMetaData has been deprecated.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=".*In 2.9, this function's implementation will be changed.*",
        )
        warnings.filterwarnings(
            "ignore", message=".*std\\(\\): degrees of freedom is <= 0.*"
        )
        warnings.filterwarnings(
            "ignore", message=".*Performing inference on CPU when CUDA is available.*"
        )
        warnings.filterwarnings(
            "ignore", message=".*FP16 is not supported on CPU; using FP32 instead.*"
        )

        # Reduce noise from third-party libraries
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("fastapi").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

        # Pipeline-related loggers
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("whisper").setLevel(logging.WARNING)

    def get_request_logger(self) -> logging.Logger:
        """Get the request logger for API request/response tracking."""
        return logging.getLogger("videoannotator.requests")

    def get_api_logger(self) -> logging.Logger:
        """Get the main API logger."""
        return logging.getLogger("videoannotator.api")

    def get_pipeline_logger(self) -> logging.Logger:
        """Get the pipeline logger."""
        return logging.getLogger("videoannotator.pipelines")

    def get_database_logger(self) -> logging.Logger:
        """Get the database logger."""
        return logging.getLogger("videoannotator.database")


# Global logging configuration instance
_logging_config: VideoAnnotatorLoggingConfig | None = None


def setup_videoannotator_logging(
    logs_dir: str | Path | None = None,
    log_level: str = "INFO",
    capture_warnings: bool = True,
    capture_stdstreams: bool = False,
) -> dict[str, logging.Logger]:
    """Set up VideoAnnotator logging system.

    New optional flags:
    - capture_warnings: Route Python warnings into the logging system (default: True)
    - capture_stdstreams: Redirect stdout/stderr into loggers (default: False)
    """
    global _logging_config
    _logging_config = VideoAnnotatorLoggingConfig(logs_dir, log_level)
    result = _logging_config.setup_logging(
        capture_warnings=capture_warnings, capture_stdstreams=capture_stdstreams
    )
    # Spec 033 FR-016: no model server key in any log, whatever logs it.
    from ..secrets_store import install_key_redaction

    install_key_redaction()
    return result


def get_logger(name: str) -> logging.Logger:
    """Get a logger by name."""
    if _logging_config is None:
        setup_videoannotator_logging()

    if _logging_config is not None:
        logger_map = {
            "api": _logging_config.get_api_logger(),
            "requests": _logging_config.get_request_logger(),
            "pipelines": _logging_config.get_pipeline_logger(),
            "database": _logging_config.get_database_logger(),
        }
        return logger_map.get(name, logging.getLogger(f"videoannotator.{name}"))

    return logging.getLogger(f"videoannotator.{name}")


@contextmanager
def log_execution_time(logger: logging.Logger, operation: str, **context):
    """Context manager to log execution time of operations."""
    start_time = datetime.now()
    try:
        logger.info(f"Starting {operation}", extra={"operation": operation, **context})
        yield
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"Completed {operation} in {execution_time:.2f}s",
            extra={"operation": operation, "execution_time": execution_time, **context},
        )
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(
            f"Failed {operation} after {execution_time:.2f}s: {e}",
            extra={
                "operation": operation,
                "execution_time": execution_time,
                "error": str(e),
                **context,
            },
        )
        raise


def log_api_request(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    response_time: float,
    request_id: str | None = None,
    user_id: str | None = None,
    **extra_context,
):
    """Log API request with structured information."""
    logger.info(
        f"{method} {path} - {status_code} - {response_time:.3f}s",
        extra={
            "request_id": request_id,
            "user_id": user_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "response_time": response_time,
            "endpoint": f"{method} {path}",
            **extra_context,
        },
    )


def log_pipeline_execution(
    logger: logging.Logger,
    pipeline_name: str,
    video_path: str,
    status: str,
    execution_time: float | None = None,
    error: str | None = None,
    **extra_context,
):
    """Log pipeline execution with structured information."""
    log_data = {
        "pipeline": pipeline_name,
        "video_path": video_path,
        "status": status,
        **extra_context,
    }

    if execution_time is not None:
        log_data["execution_time"] = execution_time

    if error:
        log_data["error"] = error
        logger.error(f"Pipeline {pipeline_name} failed: {error}", extra=log_data)
    else:
        message = f"Pipeline {pipeline_name} {status}"
        if execution_time:
            message += f" in {execution_time:.2f}s"
        logger.info(message, extra=log_data)
