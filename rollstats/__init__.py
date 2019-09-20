import array
import math

from collections import deque
from typing import Any, Callable, List, Optional, Sequence, Union, SupportsFloat

nan = float("nan")

FloatFunc = Callable[[SupportsFloat], float]
HookFunc = Callable[[SupportsFloat], Any]


def var(S: float, n: float) -> float:
    """Sample variance"""
    return (S / (n - 1)) if n > 1 else nan


def std(S: float, n: float) -> float:
    """Sample standard deviation"""
    return math.sqrt(S / (n - 1)) if n > 1 else nan


def zscore(S: float, n: float, value: float, M: float) -> float:
    """Sample z-score"""
    return (value - M) / std(S, n) if S > 0 and n > 0 else nan


def pop_var(S: float, n: float) -> float:
    """Population variance"""
    return (S / n) if n > 1 else nan


def pop_std(S: float, n: float) -> float:
    """Population standard deviation"""
    return math.sqrt(S / n) if n > 1 else nan


class MemoryFloat(object):
    """A floating point number that knows its own history.
    Every time its save() function gets called, the current value gets appended to the history.
    """

    __slots__ = ["value", "history", "hooks"]

    def __init__(self, value: float = nan):
        self.value = value
        self.history = self.new_history_container()
        self.hooks = []  # List[FloatFunc]

    @classmethod
    def new_history_container(cls, data: Sequence = None):
        """Using an array for the history seems to save about 20% memory according to profiling."""
        if data:
            return array.array("d", data)
        else:
            return array.array("d")

    @classmethod
    def transform(cls, *floats: "MemoryFloat", func: FloatFunc) -> "MemoryFloat":
        values_now = [value.value for value in floats]
        value = func(*values_now)
        history = []
        for time in range(len(floats[0])):
            values_at_time = [value.history[time] for value in floats]
            history.append(func(*values_at_time))
        result = MemoryFloat(value)
        result.history = cls.new_history_container(history)
        return result

    def save(self) -> None:
        self.history.append(self.value)
        for hook in self.hooks:
            hook(self)

    def copy(self) -> "MemoryFloat":
        result = MemoryFloat(self.value)
        result.history = self.new_history_container(self.history)
        return result

    def add_hook(self, hook: HookFunc) -> None:
        self.hooks.append(hook)

    @classmethod
    def connect(cls, *inputs: "MemoryFloat", output: "MemoryFloat", func: FloatFunc):
        pool = set()
        all_ids = {id(input) for input in inputs}

        def hook(input: "MemoryFloat") -> None:
            nonlocal pool
            pool.add(id(input))
            if set(pool) == all_ids:
                output.assign(func(*inputs))
                output.save()
                pool = set()

        for input in inputs:
            input.add_hook(hook)

    def __str__(self) -> str:
        return "MemoryFloat(v={:.2f}, history={})".format(
            self.value, ["{:.2f}".format(h) for h in self.history]
        )

    def __repr__(self) -> str:
        return str(self)

    # Trivial operator overloads below.

    def assign(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        return float(self.value)

    def __len__(self) -> int:
        return len(self.history)

    def __eq__(self, other) -> bool:
        return self.value == other

    def __lt__(self, other) -> bool:
        return self.value < other

    def __le__(self, other) -> bool:
        return self.value <= other

    def __gt__(self, other) -> bool:
        return self.value > other

    def __ge__(self, other) -> bool:
        return self.value >= other

    def __add__(self, other) -> float:
        return self.value + other

    def __radd__(self, other) -> float:
        return other + self.value

    def __iadd__(self, other) -> "MemoryFloat":
        self.value += other
        return self

    def __isub__(self, other) -> "MemoryFloat":
        self.value -= other
        return self

    def __sub__(self, other) -> float:
        return self.value - other

    def __rsub__(self, other) -> float:
        return other - self.value

    def __rtruediv__(self, other) -> float:
        return other / self.value

    def __truediv__(self, other) -> float:
        return self.value / other


class Container(object):
    def __init__(
        self,
        data: Optional[Sequence] = None,
        window_size: Union[int, float] = float("inf"),
    ):
        """Initialize the data and all metadata"""
        # Data container - if any data is provided in th initializer,
        # it will be filled at the end.
        self.data = deque()

        # Set the window size
        self.window_size = window_size

        # The current value.
        self.value = MemoryFloat(nan)

        # The number of samples in the window.
        # This is always less than or equal to the window size.
        self.n = MemoryFloat(0)

        # The current mean.
        self.M = MemoryFloat(nan)

        # The current sum.
        self.sum = MemoryFloat(0)

        # The current sum of squared differences from the mean.
        # Used to calculate the variance and standard deviation.
        self.S = MemoryFloat(nan)

        # The current sum of reciprocals.
        # Used to calculate the harmonic mean.
        self.reciprocal_sum = MemoryFloat(nan)

        # When adding a new memory float, add it to self.mem_floats
        # so it gets saved on every push.
        self.mem_floats = (
            self.value,
            self.n,
            self.S,
            self.M,
            self.sum,
            self.reciprocal_sum,
        )

        # Dumb corner case: if the window size is <0,
        # there is no need to do anything when pushing
        if self.window_size <= 0:
            self.push = lambda datapoints: None

        # Push any initial data
        if data:
            self.push(*data)

    def subscribe(self, varname: str, *inputs: "MemoryFloat", func: FloatFunc) -> None:
        output = MemoryFloat(nan)
        setattr(self, varname, output)
        MemoryFloat.connect(*inputs, output=output, func=func)

    def subscribe_var(self) -> None:
        self.subscribe("var", self.S, self.n, func=var)

    def subscribe_std(self) -> None:
        self.subscribe("std", self.S, self.n, func=std)

    def subscribe_pop_var(self) -> None:
        self.subscribe("pop_var", self.S, self.n, func=pop_var)

    def subscribe_pop_std(self) -> None:
        self.subscribe("pop_std", self.S, self.n, func=pop_std)

    def subscribe_z_score(self) -> None:
        self.subscribe("zscore", self.S, self.n, self.value, self.M, func=zscore)

    def subscribe_mean(self) -> None:
        self.subscribe("mean", self.M, func=lambda x: x)

    def subscribe_harmonic_mean(self) -> None:
        self.subscribe(
            "harmonic_mean", self.reciprocal_sum, self.n, func=lambda rec, n: n / rec
        )

    def __getitem__(self, item: Union[int, slice]) -> Union[float, List[float]]:
        """Enable slicing syntax on the container."""
        if isinstance(item, slice):
            return list(self.data)[item]
        return self.data[item]

    def __len__(self) -> int:
        """Enable checking the length of the container."""
        return len(self.data)

    def __eq__(self, other) -> bool:
        """Enable checking containers against each other for inequality"""
        if isinstance(other, self.__class__):
            return self.window_size == other.window_size and self.data == other.data
        else:
            return False

    def push(self, *datapoints: float) -> None:
        self.data.extend(datapoints)

        for datapoint in datapoints:
            self.value.assign(datapoint)
            if self.n >= self.window_size:
                self._pop()

            if datapoint == 0:
                reciprocal = nan
            else:
                reciprocal = 1 / datapoint

            self.n += 1
            self.sum += datapoint
            if self.n == 1:
                self.S.assign(0)
                self.M.assign(datapoint)
                self.reciprocal_sum.assign(reciprocal)
            else:
                prev_M = self.M.value
                cur_diff = datapoint - prev_M
                self.M += cur_diff / self.n
                self.S += cur_diff * (datapoint - self.M)
                self.reciprocal_sum += reciprocal

            self.save()

    def _pop(self) -> None:
        out = self.data.popleft()
        prev_M = self.M.value

        self.n -= 1
        self.sum -= out
        if self.n == 0:
            self.S.assign(nan)
            self.M.assign(nan)
            self.reciprocal_sum.assign(nan)
        else:
            cur_diff = out - self.M
            self.M -= cur_diff / self.n
            self.S -= cur_diff * (out - prev_M)

            if out == 0:
                self.reciprocal_sum.assign(nan)
            else:
                self.reciprocal_sum -= 1 / out

    def save(self) -> None:
        for mem_float in self.mem_floats:
            mem_float.save()
