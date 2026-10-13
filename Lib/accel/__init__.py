"""Run numeric collections and standard library functions on a compute device.

    >>> import accel, statistics
    >>> xs = accel.to_device([1.0, 2.0, 3.0, 4.0], 'mlx')   # Metal GPU
    >>> xs.device
    accel.Device('mlx:gpu')
    >>> statistics.fmean(xs)                                  # runs on the GPU
    2.5
    >>> accel.set_backend(statistics, 'mlx')                  # whole module

Only homogeneous int and float data can live on a device; everything else
transparently falls back to the regular CPU implementation.
"""

import array
import math
import operator

from ._backends import (Backend, CPUBackend, MLXBackend, register_backend,
                        get_backend as _get_backend, backend_names,
                        FLOAT_CODES, INT_CODES, NUMERIC_CODES)

__all__ = ['Device', 'DeviceArray', 'Backend', 'CPUBackend', 'MLXBackend',
           'device', 'to_device', 'is_available', 'available_devices',
           'get_default_device', 'set_default_device', 'register_backend',
           'set_backend', 'get_backend', 'backend', 'register_kernel']

_ALIASES = {
    'metal': 'mlx:gpu',
    'gpu': 'mlx:gpu',
}


class Device:
    """A compute device, such as ``Device('cpu')`` or ``Device('mlx:gpu')``.

    The spec is ``type[:stream]``.  ``'metal'`` and ``'gpu'`` are aliases
    for ``'mlx:gpu'``.  When the stream is omitted, the backend's default
    stream is used (``gpu`` for MLX).
    """

    __slots__ = ('_type', '_stream')

    def __init__(self, spec):
        if isinstance(spec, Device):
            self._type, self._stream = spec._type, spec._stream
            return
        if not isinstance(spec, str):
            raise TypeError(f'device must be a str or Device, '
                            f'not {type(spec).__name__}')
        spec = _ALIASES.get(spec, spec)
        type_, _, stream = spec.partition(':')
        be = _get_backend(type_)
        if not stream:
            stream = be.streams[0]
        elif stream not in be.streams:
            raise ValueError(f'device type {type_!r} has no stream '
                             f'{stream!r}; expected one of {be.streams}')
        self._type = type_
        self._stream = stream

    @property
    def type(self):
        return self._type

    @property
    def stream(self):
        return self._stream

    @property
    def backend(self):
        return _get_backend(self._type)

    def __str__(self):
        return f'{self._type}:{self._stream}' if self._stream else self._type

    def __repr__(self):
        return f'accel.Device({str(self)!r})'

    def __eq__(self, other):
        if isinstance(other, str):
            try:
                other = Device(other)
            except (TypeError, ValueError):
                return False
        if not isinstance(other, Device):
            return NotImplemented
        return (self._type, self._stream) == (other._type, other._stream)

    def __hash__(self):
        return hash((self._type, self._stream))

    def __reduce__(self):
        return (Device, (str(self),))


device = Device

_default_device = Device('cpu')


def get_default_device():
    """Return the device used when no device is given explicitly."""
    return _default_device


def set_default_device(dev):
    """Set the device used by :func:`to_device` when none is given."""
    global _default_device
    _default_device = _checked_device(dev)


def is_available(dev):
    """Return True if *dev* (a device type or spec) can be used."""
    try:
        if isinstance(dev, str) and ':' not in dev and dev not in _ALIASES:
            return _get_backend(dev).is_available()
        dev = Device(dev)
    except (TypeError, ValueError):
        return False
    return dev.backend.is_available(dev.stream)


def available_devices():
    """Return a list of all usable devices."""
    result = []
    for name in backend_names():
        be = _get_backend(name)
        if not be.is_available():
            continue
        for stream in be.streams:
            if be.is_available(stream):
                result.append(Device(f'{name}:{stream}' if stream else name))
    return result


def _checked_device(dev):
    if dev is None:
        return _default_device
    dev = Device(dev)
    if not dev.backend.is_available(dev.stream):
        raise RuntimeError(f'device {str(dev)!r} is not available')
    return dev


# Python ints are unbounded, device ints are int64.  Operations whose
# result may not fit are computed exactly on the CPU instead.
_INT_LIMIT = 2**62


def _infer_typecode(values):
    has_float = False
    for v in values:
        t = type(v)
        if t is float:
            has_float = True
        elif t is not int and t is not bool:
            raise TypeError(f'cannot move {t.__name__!r} elements to a '
                            f'device; only int and float are supported')
    return 'd' if has_float else 'q'


def _host_array(obj, typecode=None):
    """Convert *obj* to an array.array of a numeric typecode."""
    if isinstance(obj, array.array):
        if obj.typecode not in NUMERIC_CODES:
            raise TypeError(f'array typecode {obj.typecode!r} cannot be '
                            f'moved to a device')
        if typecode is None or typecode == obj.typecode:
            return obj
        return array.array(typecode, obj.tolist())
    if isinstance(obj, (bytes, bytearray, memoryview)):
        with memoryview(obj) as mv:
            fmt = mv.format.lstrip('@=')
            if mv.ndim != 1 or fmt not in NUMERIC_CODES:
                raise TypeError(f'memoryview format {mv.format!r} cannot '
                                f'be moved to a device')
            if fmt != mv.format:
                buf = array.array(fmt)
                buf.frombytes(mv.tobytes())
            else:
                buf = array.array(fmt, mv.tolist())
        return _host_array(buf, typecode)
    values = obj if isinstance(obj, (list, tuple, range)) else list(obj)
    if typecode is None:
        typecode = _infer_typecode(values)
    elif typecode not in NUMERIC_CODES:
        raise ValueError(f'typecode must be one of '
                         f'{"".join(sorted(NUMERIC_CODES))!r}')
    try:
        return array.array(typecode, values)
    except OverflowError:
        raise OverflowError('value does not fit in a device integer; '
                            'use typecode="d" or keep the data on the CPU'
                            ) from None


def to_device(obj, device=None, *, typecode=None):
    """Return a :class:`DeviceArray` with the contents of *obj* on *device*.

    *obj* may be a DeviceArray, an :class:`array.array`, a 1-D numeric
    buffer (``memoryview``), or any iterable of ints and floats.  Lists of
    ints are stored as int64 (``'q'``), lists containing floats as float64
    (``'d'``); pass *typecode* to choose another :mod:`array` typecode.
    If *device* is None, the default device is used.
    """
    dev = _checked_device(device)
    if isinstance(obj, DeviceArray):
        if typecode is not None and typecode != obj.typecode:
            return to_device(obj.toarray(), dev, typecode=typecode)
        return obj.to_device(dev)
    buf = _host_array(obj, typecode)
    handle = dev.backend.from_host(buf, dev.stream)
    return DeviceArray._new(dev, handle, buf.typecode)


class DeviceArray:
    """A read-only, homogeneous numeric sequence stored on a :class:`Device`.

    Create one with :func:`to_device` or :meth:`array.array.to_device`.
    Indexing, iteration, ``len()`` and the buffer protocol work as for an
    :class:`array.array`.  Reductions run on the device.
    """

    __slots__ = ('_device', '_handle', '_typecode', '__weakref__')

    def __init__(self, *args, **kwargs):
        raise TypeError('use accel.to_device() to create a DeviceArray')

    @classmethod
    def _new(cls, dev, handle, typecode):
        self = object.__new__(cls)
        self._device = dev
        self._handle = handle
        self._typecode = typecode
        return self

    @property
    def device(self):
        return self._device

    @property
    def typecode(self):
        """The :mod:`array` typecode of the data (as seen from Python)."""
        return self._typecode

    @property
    def storage_typecode(self):
        """The typecode actually used on the device.  This differs from
        :attr:`typecode` when the device lacks the type, for example
        float64 (``'d'``) data on the Metal GPU is stored as float32."""
        return self._backend.storage_typecode(self._typecode,
                                              self._device.stream)

    @property
    def _backend(self):
        return self._device.backend

    def _wrap(self, handle, typecode=None):
        return DeviceArray._new(self._device, handle,
                                typecode or self._typecode)

    # -- conversion --------------------------------------------------------

    def to_device(self, device):
        """Return this data on *device* (self if it is already there)."""
        dev = _checked_device(device)
        if dev == self._device:
            return self
        if dev.type == self._device.type:
            handle = dev.backend.move(self._handle, dev.stream,
                                      self._typecode)
            if handle is not None:
                return DeviceArray._new(dev, handle, self._typecode)
        handle = dev.backend.from_host(self.toarray(), dev.stream)
        return DeviceArray._new(dev, handle, self._typecode)

    def cpu(self):
        """Return this data on the ``cpu`` device."""
        return self.to_device('cpu')

    def toarray(self):
        """Return the data as a new :class:`array.array`."""
        return self._backend.to_host(self._handle, self._typecode)

    def tolist(self):
        return self.toarray().tolist()

    def __buffer__(self, flags):
        return memoryview(self.toarray())

    # -- sequence protocol --------------------------------------------------

    def __len__(self):
        return self._backend.length(self._handle)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self._wrap(self._backend.getitem(self._handle, key))
        n = len(self)
        i = operator.index(key)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError('DeviceArray index out of range')
        value = self._backend.getitem(self._handle, i)
        return float(value) if self._typecode in FLOAT_CODES else int(value)

    def __iter__(self):
        return iter(self.toarray())

    def __reversed__(self):
        return reversed(self.toarray())

    def __contains__(self, value):
        return value in self.toarray()

    def __eq__(self, other):
        if not isinstance(other, DeviceArray):
            return NotImplemented
        return self.tolist() == other.tolist()

    __hash__ = None

    def __repr__(self):
        return (f'accel.DeviceArray({self._typecode!r}, {self.tolist()!r}, '
                f'device={str(self._device)!r})')

    def __reduce__(self):
        return (to_device, (self.toarray(), str(self._device)))

    # -- reductions ---------------------------------------------------------

    def _abs_max(self):
        if not len(self):
            return 0
        be = self._backend
        return max(abs(be.min(self._handle)), abs(be.max(self._handle)))

    def _int_ok(self, bound):
        # The cpu backend computes with Python ints and is always exact.
        return (self._device.type == 'cpu' or self._typecode in FLOAT_CODES
                or bound < _INT_LIMIT)

    def _exact(self):
        """This data as a CPU DeviceArray, for exact fallbacks."""
        return self if self._device.type == 'cpu' else self.cpu()

    def sum(self):
        if not self._int_ok(len(self) * self._abs_max()):
            return self._exact().sum()
        return self._backend.sum(self._handle)

    def prod(self):
        if self._typecode in INT_CODES and self._device.type != 'cpu':
            m = self._abs_max()
            if m > 1 and len(self) * m.bit_length() >= 62:
                return self._exact().prod()
        return self._backend.prod(self._handle)

    def min(self):
        if not len(self):
            raise ValueError('min() of an empty DeviceArray')
        return self._backend.min(self._handle)

    def max(self):
        if not len(self):
            raise ValueError('max() of an empty DeviceArray')
        return self._backend.max(self._handle)

    def mean(self):
        n = len(self)
        if not n:
            raise ValueError('mean() of an empty DeviceArray')
        return self.sum() / n

    def dot(self, other):
        other = _coerce(other, self._device)
        if len(other) != len(self):
            raise ValueError('inputs are not the same length')
        bound = len(self) * self._abs_max() * other._abs_max()
        if not (self._int_ok(bound) and other._int_ok(bound)):
            return self._exact().dot(other._exact())
        return self._backend.dot(self._handle, other._handle)

    # -- array-valued operations ---------------------------------------------

    def _binary_scalar(self, op, scalar):
        """Element-wise *op* with a float *scalar*; returns float data."""
        handle = self._backend.binary(op, self._handle, float(scalar))
        return self._wrap(handle, 'd')

    def sorted(self, *, reverse=False):
        return self._wrap(self._backend.sort(self._handle, reverse=reverse))

    def cumsum(self):
        if not self._int_ok(len(self) * self._abs_max()):
            return self._exact().cumsum().to_device(self._device)
        return self._wrap(self._backend.cumsum(self._handle))

    def cumprod(self):
        if self._typecode in INT_CODES and self._device.type != 'cpu':
            m = self._abs_max()
            if m > 1 and len(self) * m.bit_length() >= 62:
                return self._exact().cumprod().to_device(self._device)
        return self._wrap(self._backend.cumprod(self._handle))


def _coerce(obj, dev):
    if isinstance(obj, DeviceArray):
        return obj.to_device(dev)
    return to_device(obj, dev)


# Callables that map() knows how to vectorize.
_UNARY = {
    abs: 'abs', math.fabs: 'abs', operator.abs: 'abs',
    operator.neg: 'negative',
    math.sqrt: 'sqrt', math.exp: 'exp', math.log: 'log',
    math.sin: 'sin', math.cos: 'cos', math.tan: 'tan', math.tanh: 'tanh',
}
_BINARY = {
    operator.add: 'add', operator.sub: 'subtract',
    operator.mul: 'multiply', operator.truediv: 'divide',
    operator.pow: 'power', math.pow: 'power',
    max: 'maximum', min: 'minimum',
}
_FLOAT_RESULT = frozenset({'sqrt', 'exp', 'log', 'sin', 'cos', 'tan', 'tanh',
                           'divide'})
# Largest argument of exp() that does not overflow, per storage type.
_EXP_LIMIT = {'f': 88.0, 'd': 709.0}


def _needs_exact(op, x, y=None):
    """Return True if computing *op* on the device could give a different
    result (or no exception) where Python would raise or be exact."""
    if op == 'sqrt':
        return len(x) and x.min() < 0
    if op == 'log':
        return len(x) and x.min() <= 0
    if op == 'exp':
        return len(x) and x.max() > _EXP_LIMIT.get(x.storage_typecode, 709.0)
    if op == 'divide':
        return len(y) and y._backend.min(
            y._backend.unary('abs', y._handle)) == 0
    if op == 'power':
        return True if x.typecode in INT_CODES else (
            len(x) and x.min() < 0)
    if x.typecode in FLOAT_CODES and (y is None or y.typecode in FLOAT_CODES):
        return False
    mx = x._abs_max()
    my = y._abs_max() if y is not None else 0
    if op == 'multiply':
        return mx * my >= _INT_LIMIT
    return mx + my >= _INT_LIMIT


def map(function, *iterables, device=None):
    """Vectorized ``map(function, *iterables)`` that runs on a device.

    *function* must be one of the functions listed in the documentation,
    such as :func:`math.sqrt` or :func:`operator.mul`.  Each argument is a
    DeviceArray or an iterable of numbers; all must have the same length.
    The computation runs on *device*, or on the device of the first
    DeviceArray argument, or on the default device.  Inputs for which
    the device could give a different result than Python (math domain
    errors, division by zero, integer overflow) are computed exactly on
    the CPU instead, raising the same exceptions as :func:`builtins.map`
    would.  Returns a :class:`DeviceArray`.
    """
    if len(iterables) == 1:
        table, kind = _UNARY, 'unary'
    elif len(iterables) == 2:
        table, kind = _BINARY, 'binary'
    else:
        raise TypeError('accel.map() takes one or two iterables')
    try:
        op = table[function]
    except (KeyError, TypeError):
        raise TypeError(f'accel.map() cannot vectorize {function!r} with '
                        f'{len(iterables)} argument(s)') from None
    if device is None:
        device = next((a.device for a in iterables
                       if isinstance(a, DeviceArray)), None)
    dev = _checked_device(device)
    args = [_coerce(a, dev) for a in iterables]
    if kind == 'binary' and len(args[0]) != len(args[1]):
        raise ValueError('accel.map() arguments must have the same length')
    typecode = args[0].typecode
    if op in _FLOAT_RESULT or any(a.typecode in FLOAT_CODES for a in args):
        typecode = 'd'
    if dev.type != 'cpu' and _needs_exact(op, *args):
        result = map(function, *[a.cpu() for a in args])
        return result.to_device(dev)
    be = dev.backend
    if kind == 'unary':
        handle = be.unary(op, args[0]._handle)
    else:
        handle = be.binary(op, args[0]._handle, args[1]._handle)
    return DeviceArray._new(dev, handle, typecode)


from ._dispatch import set_backend, get_backend, backend, register_kernel
