import logging

__all__ = ["logger"]


class Logger:
    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._verbose = True

    def info(self, message: str, verbose: bool = True):
        if verbose:
            self._logger.info(message)

    def warning(self, message: str, verbose: bool = True):
        self._logger.warning(message)

    def error(self, message: str, verbose: bool = True):
        self._logger.error(message)

    # Delegate other methods to the underlying logger
    def __getattr__(self, name):
        return getattr(self._logger, name)


def _setup_logger() -> Logger:
    from rich.console import Console
    from rich.logging import RichHandler

    base_logger = logging.getLogger(__name__)
    base_logger.setLevel(logging.INFO)
    console = Console(force_terminal=True)
    if console.is_jupyter is True:
        console.is_jupyter = False
    ch = RichHandler(show_path=False, console=console, show_time=False)
    base_logger.addHandler(ch)
    base_logger.propagate = False

    return Logger(base_logger)


logger = _setup_logger()
