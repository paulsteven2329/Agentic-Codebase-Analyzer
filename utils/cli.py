from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


@dataclass
class CliTheme:
    accent: str = "cyan"
    success: str = "green"
    warning: str = "yellow"
    error: str = "red"
    muted: str = "bright_black"


class CliReporter:
    def __init__(self, accent: str = "cyan") -> None:
        self.console = Console()
        self.theme = CliTheme(accent=accent)
        self._start_time = time.perf_counter()

    def print_banner(self, target_path: str) -> None:
        self.console.print(
            Panel.fit(
                f"[bold {self.theme.accent}]Agentic Codebase Analyzer[/bold {self.theme.accent}]\n"
                f"[{self.theme.muted}]Target:[/{self.theme.muted}] {target_path}",
                border_style=self.theme.accent,
                title="[bold]Analysis Session[/bold]",
            )
        )

    @contextmanager
    def stage(self, description: str) -> Iterator[None]:
        with Progress(
            SpinnerColumn(style=self.theme.accent),
            TextColumn(f"[bold {self.theme.accent}]{{task.description}}[/bold {self.theme.accent}]"),
            BarColumn(bar_width=24),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            task_id = progress.add_task(description, total=None)
            try:
                yield
            finally:
                progress.update(task_id, completed=1)
        self.console.print(f"[{self.theme.success}]OK[/{self.theme.success}] {description}")

    @contextmanager
    def live_status(self, message: str) -> Iterator["CliReporter"]:
        with self.console.status(f"[bold {self.theme.accent}]{message}[/bold {self.theme.accent}]") as status:
            self._status = status
            try:
                yield self
            finally:
                self._status = None

    def update_status(self, message: str) -> None:
        status = getattr(self, "_status", None)
        if status is not None:
            status.update(f"[bold {self.theme.accent}]{message}[/bold {self.theme.accent}]")

    def info(self, message: str) -> None:
        self.console.print(f"[{self.theme.accent}]•[/{self.theme.accent}] {message}")

    def error(self, message: str) -> None:
        self.console.print(f"[{self.theme.error}]x[/{self.theme.error}] {message}")

    def print_summary(self, summary: str) -> None:
        elapsed = time.perf_counter() - self._start_time
        self.console.print()
        self.console.print(
            Panel.fit(
                f"[bold {self.theme.success}]Report ready[/bold {self.theme.success}]  [{self.theme.muted}]{elapsed:.1f}s[/{self.theme.muted}]",
                border_style=self.theme.success,
                title="[bold]Completed[/bold]",
            )
        )
        self.console.print()
        self.console.print(summary)
