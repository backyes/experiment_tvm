from __future__ import annotations

import dataclasses as dcls
import functools
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, Tuple, Type, TypeVar, final, overload

import torch
from numpy import ndarray
from rich.logging import RichHandler
from torch import Tensor
from torch import device as Device
from torch import dtype as DType

from koila import constants
from koila.interfaces import BatchInfo, Runnable, RunnableTensor
from koila.prepasses import PrePass, PrePassFunc

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())

T = TypeVar("T", covariant=True)
V = TypeVar("V", contravariant=True)


@final
@dataclass(frozen=True)
class LazyFunction(Generic[V]):
    func: Callable[..., Tensor]
    prepass_func: PrePassFunc

    def __call__(self, *args: Any, **kwargs: Any) -> DelayedTensor:
        lazy_args = tuple(delayed(arg) for arg in args)
        lazy_kwargs = dict((k, delayed(v)) for (k, v) in kwargs.items())
        prepass = self.prepass_func(*args, **kwargs)
        return DelayedTensor(self.func, prepass, *lazy_args, **lazy_kwargs)

    def __get__(self, obj: V, objtype: Type[V]) -> Callable[..., DelayedTensor]:
        assert isinstance(obj, objtype), [type(obj), objtype]
        if obj is None:
            return self
        else:
            return functools.partial(self, obj)


@final
@dataclass(init=False)
class DelayedTensor(RunnableTensor):
    func: Callable[..., Tensor]
    prepass: PrePass
    args: Tuple[Runnable[Any], ...] = dcls.field(default_factory=tuple)
    kwargs: Dict[str, Runnable[Any]] = dcls.field(default_factory=dict)

    def __init__(
        self,
        func: Callable[..., Tensor],
        prepass: PrePass,
        *args: RunnableTensor | Tensor,
        **kwargs: RunnableTensor | Tensor,
    ) -> None:
        self.func = func
        self.prepass = prepass
        self.args = tuple(delayed(arg) for arg in args)
        self.kwargs = dict((k, delayed(v)) for (k, v) in kwargs.items())

    def __str__(self) -> str:
        return f"{self.func}(*{self.args}, **{self.kwargs}) -> {self.prepass}"

    @property
    def batch(self) -> BatchInfo | None:
        return self.prepass.batch

    @property
    def dtype(self) -> DType:
        return self.prepass.dtype

    @property
    def device(self) -> str | Device:
        return self.prepass.device

    def run(self, partial: range | None = None) -> Tensor:
        del partial

        real_args = [arg.run() for arg in self.args]
        real_kwargs = {k: v.run() for (k, v) in self.kwargs.items()}

        result = self.func(*real_args, **real_kwargs)

        assert self.prepass.shape == result.shape
        return result

    @overload
    def size(self) -> Tuple[int, ...]:
        ...

    @overload
    def size(self, dim: int) -> int:
        ...

    def size(self, dim: int | None = None) -> int | Tuple[int, ...]:
        shape = self.prepass.shape

        if dim is not None:
            return shape[dim]
        else:
            return shape


class ReadyTensor(Tensor, RunnableTensor):
    """
    Immediate tensor is a thin wrapper for the `Tensor` class. It's basically a tensor.
    """

    batch: BatchInfo | None = None

    device: str | Device = constants.ANY_DEVICE

    def run(self, partial: range | None = None) -> Tensor:
        del partial

        return self


@overload
def delayed(input: RunnableTensor) -> RunnableTensor:
    ...


@overload
def delayed(input: Tensor | ndarray) -> RunnableTensor:
    ...


def delayed(input: RunnableTensor | Tensor | ndarray) -> RunnableTensor:
    if isinstance(input, Runnable):
        return input

    if isinstance(input, ndarray):
        tensor = torch.from_numpy(input)

    if isinstance(input, Tensor):
        return tensor.as_subclass(ReadyTensor)  # type: ignore

    raise ValueError
