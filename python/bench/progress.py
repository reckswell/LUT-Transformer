"""Progress bars with time estimates (tqdm) for every long job: benchmark sweeps, the eval harness
and training. Alpha (2026-11-19): whenever benchmarks or training run, show how long they will take,
so there is a rough estimate to work from.

    from bench.progress import progress_bar, WeightedBar

    for step in progress_bar(range(num_steps), desc="train", unit="step"):   # a plain bar
        ...

    bar = WeightedBar(total_s=95.0, count=4, label="sweep", unit="rows")   # bar in estimated seconds
    bar.advance(12.5)          # 12.5 estimated seconds of work done
    bar.advance(3.0, items=1)  # ... and one more row finished
    bar.close()

House style, the same for every bar:
  - tqdm.auto: a text bar in a terminal, a widget in a Jupyter notebook.
  - Redraws at most twice a second (mininterval 0.5 s).
  - tqdm's monitor thread is off (tqdm.monitor_interval = 0), so no background thread wakes up
    during a timed region or a power window.
  - A bar is updated only between timed calls: after the stop timestamp and before the next start,
    never inside a timed region or a power window. bench/timing.time_fn's after_call hook is that
    place. So a bar never adds to a measured latency or to the energy of a power window.
  - show=None (the default) shows the bar when stderr is a terminal and hides it when output goes to
    a file or a pipe (pytest, logs). show=True forces it on, show=False off.

WeightedBar is for loops whose steps take very different times, such as sweep rows at 128 and at
32K tokens of context. Its units are estimated seconds, not calls, so its remaining time is tqdm's
elapsed / done * (total - done) in estimated seconds: the plan's estimate, corrected by how fast the
work has actually gone so far. The count of finished items (rows, repeats) is in its label.
"""

from __future__ import annotations

from typing import Any, Iterable

from tqdm import tqdm as _tqdm_class
from tqdm.auto import tqdm

MININTERVAL_S = 0.5
WEIGHTED_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]{postfix}"


def _quiet_tqdm() -> None:
    """No tqdm monitor thread: it would wake every few seconds, possibly inside a timed region."""
    _tqdm_class.monitor_interval = 0


def progress_bar(iterable: Iterable[Any] | None = None, *, total: float | None = None, desc: str = "",
                 unit: str = "it", show: bool | None = None, position: int | None = None,
                 leave: bool = True, **kwargs: Any) -> Any:
    """A tqdm bar in the house style (see the module docstring). Same arguments as tqdm, plus show."""
    _quiet_tqdm()
    return tqdm(iterable, total=total, desc=desc, unit=unit, mininterval=MININTERVAL_S,
                dynamic_ncols=True, disable=None if show is None else not show,
                position=position, leave=leave, **kwargs)


def format_seconds(seconds: float | None) -> str:
    """H:MM:SS, or "?" when unknown."""
    if seconds is None or seconds != seconds:   # None or NaN
        return "?"
    seconds = int(round(max(0.0, seconds)))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def write(message: str) -> None:
    """Print a line without breaking the bars on screen."""
    tqdm.write(message)


class WeightedBar:
    """A progress bar measured in estimated seconds, labelled with a count of finished items.

    total_s   the plan's estimate for all the work, in seconds
    count     how many items (rows, repeats) the work has, shown as "label done/count unit"
    """

    def __init__(self, total_s: float, count: int, *, label: str, unit: str, show: bool | None = None,
                 position: int | None = None, leave: bool = True) -> None:
        self.total_s = max(float(total_s), 1e-6)
        self.count = count
        self.done = 0
        self.label = label
        self.unit = unit
        self._spent = 0.0
        self._bar = progress_bar(total=self.total_s, desc=self._desc(), show=show, position=position,
                                 leave=leave, bar_format=WEIGHTED_FORMAT)

    def _desc(self) -> str:
        return f"{self.label} {self.done}/{self.count} {self.unit}"

    @property
    def spent_s(self) -> float:
        """Estimated seconds of work marked done so far."""
        return self._spent

    def advance(self, estimated_s: float, items: int = 0) -> None:
        """Mark estimated_s seconds of planned work done and `items` more items finished. The bar
        stops just short of 100% until close(), so an underestimate never shows as finished."""
        room = self.total_s * 0.999 - self._spent
        step = min(max(0.0, float(estimated_s)), max(0.0, room))
        self._spent += step
        self.done += items
        if items:
            self._bar.set_description_str(self._desc(), refresh=False)
        self._bar.update(step)

    def set_label(self, label: str) -> None:
        self.label = label
        self._bar.set_description_str(self._desc(), refresh=False)

    def set_note(self, note: str) -> None:
        """Short text after the time estimate, such as the current context length."""
        self._bar.set_postfix_str(note, refresh=False)

    def close(self) -> None:
        """Fill the bar to 100% and close it."""
        self._bar.update(self.total_s - self._spent)
        self._spent = self.total_s
        self._bar.close()
