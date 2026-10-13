"""Per-module backend switching.

``set_backend(statistics, 'mlx')`` replaces the functions of a module that
have a registered *kernel* with dispatching wrappers.  A wrapper moves its
data to the module's device and runs the kernel there.  Kernels return
``NotImplemented`` to fall back to the original function -- for inputs that
are too small to be worth moving, are not homogeneous ints/floats (such as
Fraction or Decimal), or are edge cases whose exceptions the original
function defines.  ``set_backend(statistics, 'cpu')`` restores the module.

Only lookups through the module are affected: names bound earlier with
``from statistics import mean`` keep referring to the original function.
"""

import contextlib
import functools
import importlib
import math
import operator
import statistics
import sys
import threading
import types
from fractions import Fraction

from . import DeviceArray, Device, to_device, _checked_device
from ._backends import INT_CODES

__all__ = ['set_backend', 'get_backend', 'backend', 'register_kernel']

#: Inputs shorter than this stay on the CPU: moving them costs more than
#: the computation itself.
DEFAULT_MIN_SIZE = 4096

# module name -> {function name: kernel}
_kernels = {}
# module name -> _Config for modules switched away from the CPU
_configs = {}
_lock = threading.RLock()


class _Config:
    __slots__ = ('device', 'min_size', 'originals')

    def __init__(self, device, min_size, originals):
        self.device = device
        self.min_size = min_size
        self.originals = originals


def register_kernel(module, name, kernel):
    """Register *kernel* as the device implementation of *module.name*.

    The kernel is called as ``kernel(original, device, min_size, *args,
    **kwargs)`` and returns the result, or ``NotImplemented`` to make the
    wrapper call ``original(*args, **kwargs)`` instead.
    """
    modname = _modname(module)
    with _lock:
        _kernels.setdefault(modname, {})[name] = kernel
        config = _configs.get(modname)
        if config is not None:
            _install(sys.modules[modname], name, config)


def _modname(module):
    if isinstance(module, types.ModuleType):
        return module.__name__
    if isinstance(module, str):
        return module
    raise TypeError(f'module must be a module or a str, '
                    f'not {type(module).__name__}')


def set_backend(module, device, *, min_size=DEFAULT_MIN_SIZE):
    """Run the supported functions of *module* on *device*.

    *module* is a module object or name (``'statistics'``, ``'math'``,
    ``'functools'``, ``'itertools'``, or any module with registered
    kernels).  Use ``device='cpu'`` to restore the original functions.
    Inputs with fewer than *min_size* elements always run on the CPU.
    """
    modname = _modname(module)
    with _lock:
        if modname not in _kernels:
            raise ValueError(f'no device kernels are registered for module '
                             f'{modname!r}; supported modules: '
                             f'{sorted(_kernels)}')
        dev = Device(device)
        mod = importlib.import_module(modname)
        if dev.type == 'cpu':
            config = _configs.pop(modname, None)
            if config is not None:
                for name, original in config.originals.items():
                    setattr(mod, name, original)
            return
        dev = _checked_device(dev)
        if min_size < 0:
            raise ValueError('min_size must be non-negative')
        config = _configs.get(modname)
        if config is None:
            config = _configs[modname] = _Config(dev, min_size, {})
        config.device = dev
        config.min_size = min_size
        for name in _kernels[modname]:
            _install(mod, name, config)


def _install(mod, name, config):
    if name in config.originals:
        return
    original = getattr(mod, name)
    kernel = _kernels[mod.__name__][name]

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        result = kernel(original, config.device, config.min_size,
                        *args, **kwargs)
        if result is NotImplemented:
            return original(*args, **kwargs)
        return result

    config.originals[name] = original
    setattr(mod, name, wrapper)


def get_backend(module):
    """Return the :class:`Device` that *module* currently runs on."""
    config = _configs.get(_modname(module))
    return Device('cpu') if config is None else config.device


@contextlib.contextmanager
def backend(module, device, *, min_size=DEFAULT_MIN_SIZE):
    """Context manager that temporarily calls :func:`set_backend`."""
    modname = _modname(module)
    with _lock:
        config = _configs.get(modname)
        saved = (get_backend(modname),
                 config.min_size if config else DEFAULT_MIN_SIZE)
        set_backend(modname, device, min_size=min_size)
    try:
        yield get_backend(modname)
    finally:
        set_backend(modname, saved[0], min_size=saved[1])


# -- kernels ---------------------------------------------------------------

def as_device(data, device, min_size):
    """Return *data* as a DeviceArray on *device*, or None.

    DeviceArrays are used where they are (computation follows the data).
    None is returned when *data* is small or not homogeneous numeric.
    *data* must have a length: pass iterators through _materialize() first
    so that the fallback can still see all of the data.
    """
    if isinstance(data, DeviceArray):
        return data
    if len(data) < min_size:
        return None
    try:
        return to_device(data, device)
    except (TypeError, OverflowError):
        # Not representable on the device; other errors are bugs and
        # must not be hidden by a silent fallback.
        return None


def _materialize(data):
    if isinstance(data, DeviceArray) or hasattr(data, '__len__'):
        return data
    return list(data)


def _int_mean(x):
    # Exact, like statistics.mean() on ints.
    q = Fraction(x.sum(), len(x))
    return q.numerator if q.denominator == 1 else float(q)


def _centered(x):
    """Return *x* minus its mean, as float data."""
    return x._binary_scalar('subtract', x.sum() / len(x))


def _sum_sq_dev(x):
    """Return (n, sum of squared deviations) for DeviceArray *x*.

    For int data the result is an exact Fraction, as in statistics.
    """
    n = len(x)
    if x.typecode in INT_CODES:
        s1 = x.sum()
        s2 = x.dot(x)
        return n, Fraction(n * s2 - s1 * s1, n)
    d = _centered(x)
    return n, d.dot(d)


def _stat_kernel(name, min_n):
    def deco(func):
        def kernel(original, device, min_size, data, *args, **kwargs):
            if args or kwargs:
                return original(_materialize(data), *args, **kwargs)
            data = _materialize(data)
            x = as_device(data, device, min_size)
            if x is None or len(x) < min_n:
                return original(data)
            return func(x)
        kernel.__name__ = name
        register_kernel('statistics', name, kernel)
        return func
    return deco


def _convert(value):
    """Match the return type of statistics functions on int data."""
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return value.numerator
        return float(value)
    return value


@_stat_kernel('fmean', 1)
def _fmean(x):
    return float(x.sum()) / len(x)


@_stat_kernel('mean', 1)
def _mean(x):
    if x.typecode in INT_CODES:
        return _int_mean(x)
    return x.sum() / len(x)


@_stat_kernel('median', 1)
def _median(x):
    n = len(x)
    s = x.sorted()
    i = n // 2
    if n % 2:
        return s[i]
    return (s[i - 1] + s[i]) / 2


@_stat_kernel('variance', 2)
def _variance(x):
    n, ss = _sum_sq_dev(x)
    return _convert(ss / (n - 1))


@_stat_kernel('pvariance', 1)
def _pvariance(x):
    n, ss = _sum_sq_dev(x)
    return _convert(ss / n)


@_stat_kernel('stdev', 2)
def _stdev(x):
    n, ss = _sum_sq_dev(x)
    var = ss / (n - 1)
    if isinstance(var, Fraction):
        return statistics._float_sqrt_of_frac(var.numerator, var.denominator)
    return math.sqrt(var)


@_stat_kernel('pstdev', 1)
def _pstdev(x):
    n, ss = _sum_sq_dev(x)
    var = ss / n
    if isinstance(var, Fraction):
        return statistics._float_sqrt_of_frac(var.numerator, var.denominator)
    return math.sqrt(var)


def _geometric_mean(original, device, min_size, data):
    import accel
    data = _materialize(data)
    x = as_device(data, device, min_size)
    if x is None or not len(x) or x.min() <= 0:
        return original(data)
    return math.exp(accel.map(math.log, x).sum() / len(x))

register_kernel('statistics', 'geometric_mean', _geometric_mean)


def _pair(device, min_size, x, y, min_n):
    """Convert both inputs or return None."""
    dx = as_device(x, device, min_size)
    dy = as_device(y, device, min_size) if dx is not None else None
    if dx is None or dy is None or len(dx) != len(dy) or len(dx) < min_n:
        return None
    if dy.device != dx.device:
        dy = dy.to_device(dx.device)
    return dx, dy


def _covariance(original, device, min_size, x, y, /):
    x, y = _materialize(x), _materialize(y)
    pair = _pair(device, min_size, x, y, 2)
    if pair is None:
        return original(x, y)
    dx, dy = pair
    return _centered(dx).dot(_centered(dy)) / (len(dx) - 1)

register_kernel('statistics', 'covariance', _covariance)


def _correlation(original, device, min_size, x, y, /, *, method='linear'):
    x, y = _materialize(x), _materialize(y)
    pair = None
    if method == 'linear':
        pair = _pair(device, min_size, x, y, 2)
    if pair is None:
        return original(x, y, method=method)
    cx, cy = _centered(pair[0]), _centered(pair[1])
    sxx, syy = cx.dot(cx), cy.dot(cy)
    if not sxx or not syy:
        return original(x, y, method=method)
    return cx.dot(cy) / math.sqrt(sxx * syy)

register_kernel('statistics', 'correlation', _correlation)


# -- math --------------------------------------------------------------------

def _prod(original, device, min_size, iterable, /, *, start=1):
    data = _materialize(iterable)
    x = None if start != 1 else as_device(data, device, min_size)
    if x is None:
        return original(data, start=start)
    return x.prod()

register_kernel('math', 'prod', _prod)


def _sumprod(original, device, min_size, p, q, /):
    p, q = _materialize(p), _materialize(q)
    pair = _pair(device, min_size, p, q, 1)
    if pair is None:
        return original(p, q)
    return pair[0].dot(pair[1])

register_kernel('math', 'sumprod', _sumprod)


# -- functional tools ----------------------------------------------------------

_REDUCERS = {
    operator.add: DeviceArray.sum,
    operator.mul: DeviceArray.prod,
    max: DeviceArray.max,
    min: DeviceArray.min,
}

_MISSING = object()


def _reduce(original, device, min_size, function, iterable, /,
            initial=_MISSING):
    data = _materialize(iterable)
    args = (function, data) if initial is _MISSING else (
        function, data, initial)
    reducer = _REDUCERS.get(function)
    x = None if reducer is None else as_device(data, device, min_size)
    if x is None or not len(x):
        return original(*args)
    result = reducer(x)
    if initial is not _MISSING:
        result = function(initial, result)
    return result

register_kernel('functools', 'reduce', _reduce)


_SCANS = {
    None: DeviceArray.cumsum,
    operator.add: DeviceArray.cumsum,
    operator.mul: DeviceArray.cumprod,
}


def _accumulate(original, device, min_size, iterable, func=None, *,
                initial=None):
    data = _materialize(iterable)
    scan = _SCANS.get(func) if initial is None else None
    x = None if scan is None else as_device(data, device, min_size)
    if x is None:
        return original(data, func, initial=initial)
    return iter(scan(x))

register_kernel('itertools', 'accumulate', _accumulate)
