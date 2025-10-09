"""
Enhanced logging configuration for VideoAnnotator API Server.

Provides structured logging with file rotation, request tracking, 
and comprehensive debugging support.
"""

import logging
import logging.handlers
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime
import traceback
from contextlib import contextmanager

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
            "line": record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs',
                          'relativeCreated', 'thread', 'threadName', 'processName', 'process',
                          'message', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry)

class APIRequestFormatter(logging.Formatter):
    """Human-readable formatter for API requests."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human readability."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        
        # Standard log format
        base_format = f"{timestamp} - {record.levelname:8} - {record.name} - {record.getMessage()}"
        
        # Add extra context if available
        if hasattr(record, 'request_id'):
            base_format = f"[{record.request_id}] {base_format}"
        
        if hasattr(record, 'user_id'):
            base_format = f"[user:{record.user_id}] {base_format}"
        
        if hasattr(record, 'endpoint'):
            base_format = f"{base_format} - {record.endpoint}"
        
        return base_format

class VideoAnnotatorLoggingConfig:
    """Configuration class for VideoAnnotator logging system."""
    
    def __init__(self, logs_dir: str = "logs", log_level: str = "INFO"):
        self.logs_dir = Path(logs_dir)
        self.log_level = log_level.upper()
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create log files
        self.api_log_file = self.logs_dir / "api_server.log"
        self.error_log_file = self.logs_dir / "errors.log"
        self.request_log_file = self.logs_dir / "api_requests.log"
        self.debug_log_file = self.logs_dir / "debug.log"
        
    def setup_logging(self, capture_warnings: bool = True, capture_stdstreams: bool = False) -> Dict[str, logging.Logger]:
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
        console_formatter = APIRequestFormatter()
        console_handler.setFormatter(console_formatter)
        
        # 2. Main API Server Logger (rotating file)
        api_handler = logging.handlers.RotatingFileHandler(
            self.api_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        api_handler.setLevel(getattr(logging, self.log_level))
        api_formatter = StructuredFormatter()
        api_handler.setFormatter(api_formatter)
        
        # 3. Error Logger (separate file for errors only)
        error_handler = logging.handlers.RotatingFileHandler(
            self.error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = StructuredFormatter()
        error_handler.setFormatter(error_formatter)
        
        # 4. Request Logger (API request/response tracking)
        request_handler = logging.handlers.RotatingFileHandler(
            self.request_log_file,
            maxBytes=20*1024*1024,  # 20MB
            backupCount=3,
            encoding='utf-8'
        )
        request_handler.setLevel(logging.INFO)
        request_formatter = StructuredFormatter()
        request_handler.setFormatter(request_formatter)
        
        # 5. Debug Logger (verbose logging)
        if self.log_level == "DEBUG":
            debug_handler = logging.handlers.RotatingFileHandler(
                self.debug_log_file,
                maxBytes=50*1024*1024,  # 50MB
                backupCount=2,
                encoding='utf-8'
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
            py_warn_logger = logging.getLogger('py.warnings')
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
                    self._buffer = ''

                def write(self, message):
                    if message and message.strip():
                        # Preserve newlines in messages
                        for line in message.rstrip().splitlines():
                            self.logger.log(self.level, line)

                def flush(self):
                    pass

            sys.stdout = StreamToLogger(api_logger if 'api' in locals() else logging.getLogger(), logging.INFO)
            sys.stderr = StreamToLogger(error_handler if False else logging.getLogger(), logging.ERROR)
        
        return loggers
    
    def _configure_third_party_loggers(self):
        """Configure third-party library logging levels."""
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
_logging_config: Optional[VideoAnnotatorLoggingConfig] = None

def setup_videoannotator_logging(logs_dir: str = "logs", log_level: str = "INFO",
                                 capture_warnings: bool = True,
                                 capture_stdstreams: bool = False) -> Dict[str, logging.Logger]:
    """Set up VideoAnnotator logging system.

    New optional flags:
    - capture_warnings: Route Python warnings into the logging system (default: True)
    - capture_stdstreams: Redirect stdout/stderr into loggers (default: False)
    """
    global _logging_config
    _logging_config = VideoAnnotatorLoggingConfig(logs_dir, log_level)
    return _logging_config.setup_logging(capture_warnings=capture_warnings,
                                         capture_stdstreams=capture_stdstreams)

def get_logger(name: str) -> logging.Logger:
    """Get a logger by name."""
    if _logging_config is None:
        setup_videoannotator_logging()
    
    logger_map = {
        "api": _logging_config.get_api_logger(),
        "requests": _logging_config.get_request_logger(), 
        "pipelines": _logging_config.get_pipeline_logger(),
        "database": _logging_config.get_database_logger()
    }
    
    return logger_map.get(name, logging.getLogger(f"videoannotator.{name}"))

@contextmanager
def log_execution_time(logger: logging.Logger, operation: str, **context):
    """Context manager to log execution time of operations."""
    start_time = datetime.now()
    try:
        logger.info(f"Starting {operation}", extra={"operation": operation, **context})
        yield
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Completed {operation} in {execution_time:.2f}s", 
                   extra={"operation": operation, "execution_time": execution_time, **context})
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Failed {operation} after {execution_time:.2f}s: {e}", 
                    extra={"operation": operation, "execution_time": execution_time, 
                          "error": str(e), **context})
        raise

def log_api_request(logger: logging.Logger, method: str, path: str, 
                   status_code: int, response_time: float, 
                   request_id: Optional[str] = None, 
                   user_id: Optional[str] = None, 
                   **extra_context):
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
            **extra_context
        }
    )

def log_pipeline_execution(logger: logging.Logger, pipeline_name: str, 
                          video_path: str, status: str, 
                          execution_time: Optional[float] = None,
                          error: Optional[str] = None,
                          **extra_context):
    """Log pipeline execution with structured information."""
    log_data = {
        "pipeline": pipeline_name,
        "video_path": video_path,
        "status": status,
        **extra_context
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