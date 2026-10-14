"""Compute backends for the accel package.

A backend owns the storage of a DeviceArray (an opaque *handle*) and
implements a small set of vectorized primitives on it.  Two backends ship
with the standard library:

* ``cpu`` -- pure Python over :class:`array.array`; always available and
  bit-for-bit compatible with the regular standard library functions.
* ``mlx`` -- Apple MLX (https://github.com/ml-explore/mlx), which runs on
  the Metal GPU (stream ``gpu``) or on the CPU (stream ``cpu``).  It is
  only available when the third-party ``mlx`` package is installed.

Third parties can add backends with :func:`register_backend`.
"""

import array
import math

__all__ = ['Backend', 'CPUBackend', 'MLXBackend',
           'register_backend', 'get_backend', 'backend_names']

FLOAT_CODES = frozenset('fd')
INT_CODES = frozenset('bBhHiIlLqQ')
NUMERIC_CODES = FLOAT_CODES | INT_CODES

# accel.set_backend(math, ...) replaces these; keep the originals.
_math_prod = math.prod
_math_sumprod = math.sumprod

# Element-wise operations every backend must implement.
UNARY_OPS = frozenset({'abs', 'negative', 'sqrt', 'exp', 'log',
                       'sin', 'cos', 'tan', 'tanh'})
BINARY_OPS = frozenset({'add', 'subtract', 'multiply', 'divide', 'power',
                        'maximum', 'minimum'})


class Backend:
    """Base class for compute backends.

    *handle* arguments are whatever :meth:`from_host` returned.  Reductions
    return Python scalars; element-wise operations and :meth:`sort` return
    new handles.  *typecode* is always the logical :mod:`array` typecode of
    the data, which the backend may store in a narrower type (for example
    the Metal GPU has no float64 and stores ``'d'`` data as float32).
    """

    name = None
    #: Streams (sub-devices) this backend supports; the first is the default.
    streams = ('',)

    def is_available(self, stream=None):
        """Return True if the backend (or one of its *streams*) works."""
        return True

    def storage_typecode(self, typecode, stream):
        """Return the typecode actually used to store *typecode* data."""
        return typecode

    def from_host(self, buf, stream):
        """Copy an :class:`array.array` *buf* onto *stream*."""
        raise NotImplementedError

    def to_host(self, handle, typecode):
        """Return the data as a new :class:`array.array` of *typecode*."""
        raise NotImplementedError

    def move(self, handle, stream, typecode):
        """Return *handle* moved to another *stream* of this backend, or
        None to make the caller copy through the host."""
        return None

    def length(self, handle):
        raise NotImplementedError

    def getitem(self, handle, key):
        """Return element *key* (an int) as a Python scalar, or a new
        handle for a slice."""
        raise NotImplementedError

    def sum(self, handle):
        raise NotImplementedError

    def prod(self, handle):
        raise NotImplementedError

    def min(self, handle):
        raise NotImplementedError

    def max(self, handle):
        raise NotImplementedError

    def dot(self, a, b):
        raise NotImplementedError

    def sort(self, handle, reverse=False):
        raise NotImplementedError

    def cumsum(self, handle):
        raise NotImplementedError

    def cumprod(self, handle):
        raise NotImplementedError

    def unary(self, op, handle):
        raise NotImplementedError

    def binary(self, op, a, b):
        """*a* and *b* are each a handle or a Python scalar."""
        raise NotImplementedError

    def __repr__(self):
        return f'<{type(self).__name__} {self.name!r}>'


_CPU_UNARY = {
    'abs': abs,
    'negative': lambda x: -x,
    'sqrt': math.sqrt,
    'exp': math.exp,
    'log': math.log,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
    'tanh': math.tanh,
}

_CPU_BINARY = {
    'add': lambda x, y: x + y,
    'subtract': lambda x, y: x - y,
    'multiply': lambda x, y: x * y,
    'divide': lambda x, y: x / y,
    'power': lambda x, y: x ** y,
    'maximum': max,
    'minimum': min,
}


class CPUBackend(Backend):
    """Reference backend: plain Python over :class:`array.array`."""

    name = 'cpu'

    def from_host(self, buf, stream):
        return array.array(buf.typecode, buf)

    def to_host(self, handle, typecode):
        if handle.typecode == typecode:
            return array.array(typecode, handle)
        return array.array(typecode, handle.tolist())

    def length(self, handle):
        return len(handle)

    def getitem(self, handle, key):
        return handle[key]

    def sum(self, handle):
        if handle.typecode in FLOAT_CODES:
            return math.fsum(handle)
        return sum(handle)

    def prod(self, handle):
        return _math_prod(handle)

    def min(self, handle):
        return min(handle)

    def max(self, handle):
        return max(handle)

    def dot(self, a, b):
        return _math_sumprod(a, b)

    def sort(self, handle, reverse=False):
        return array.array(handle.typecode, sorted(handle, reverse=reverse))

    def cumsum(self, handle):
        return self._new(handle, _accumulate(handle, lambda x, y: x + y))

    def cumprod(self, handle):
        return self._new(handle, _accumulate(handle, lambda x, y: x * y))

    def unary(self, op, handle):
        return self._new(handle, list(map(_CPU_UNARY[op], handle)))

    def binary(self, op, a, b):
        fn = _CPU_BINARY[op]
        if isinstance(a, array.array) and isinstance(b, array.array):
            values = list(map(fn, a, b))
        elif isinstance(a, array.array):
            values = [fn(x, b) for x in a]
        else:
            values = [fn(a, y) for y in b]
        like = a if isinstance(a, array.array) else b
        return self._new(like, values)

    @staticmethod
    def _new(like, values):
        if values and any(isinstance(v, float) for v in values):
            return array.array('d', values)
        try:
            return array.array(like.typecode, values)
        except OverflowError:
            raise OverflowError('result does not fit in the array type')


def _accumulate(it, fn):
    out = []
    acc = None
    for i, x in enumerate(it):
        acc = x if i == 0 else fn(acc, x)
        out.append(acc)
    return out


# typecode -> mlx dtype name
_MLX_DTYPES = {
    'b': 'int8', 'B': 'uint8', 'h': 'int16', 'H': 'uint16',
    'i': 'int32', 'I': 'uint32', 'l': 'int64', 'L': 'uint64',
    'q': 'int64', 'Q': 'uint64', 'f': 'float32', 'd': 'float64',
}


class MLXBackend(Backend):
    """Apple MLX backend (Metal GPU or CPU).

    MLX arrays live in unified memory, so moving data between the ``gpu``
    and ``cpu`` streams does not copy; the stream only decides where the
    computation runs.  The Metal GPU has no float64: ``'d'`` data on the
    ``gpu`` stream is computed in float32, while the ``cpu`` stream keeps
    full float64 precision.
    """

    name = 'mlx'
    streams = ('gpu', 'cpu')

    def __init__(self):
        self._mx = None

    @property
    def mx(self):
        if self._mx is None:
            import mlx.core
            self._mx = mlx.core
        return self._mx

    def is_available(self, stream=None):
        try:
            mx = self.mx
        except ImportError:
            return False
        if stream == 'gpu':
            return mx.metal.is_available()
        return True

    def storage_typecode(self, typecode, stream):
        if stream == 'gpu' and typecode == 'd':
            return 'f'
        return typecode

    def _device(self, stream):
        mx = self.mx
        if stream == 'gpu':
            if not mx.metal.is_available():
                raise RuntimeError('MLX was built without Metal support')
            return mx.gpu
        return mx.cpu

    def from_host(self, buf, stream):
        mx = self.mx
        # array.array exports the buffer protocol: this is a single memcpy.
        data = mx.array(buf, dtype=getattr(mx, _MLX_DTYPES[buf.typecode]))
        return self.move(_MLXHandle(data, stream), stream, buf.typecode)

    def to_host(self, handle, typecode):
        mx = self.mx
        dtype = getattr(mx, _MLX_DTYPES[typecode])
        data = handle.data
        if data.dtype != dtype:
            data = data.astype(dtype, stream=mx.cpu)
        out = array.array(typecode)
        out.frombytes(bytes(memoryview(data)))
        return out

    def move(self, handle, stream, typecode):
        # Unified memory: only the stream changes; a copy is made only
        # when the storage type differs (float64 is not usable on the GPU).
        mx = self.mx
        dtype = getattr(mx, _MLX_DTYPES[self.storage_typecode(typecode,
                                                              stream)])
        data = handle.data
        if data.dtype != dtype:
            narrowed = data.astype(dtype, stream=mx.cpu)
            if (dtype == mx.float32 and data.dtype == mx.float64 and
                    mx.sum(mx.isinf(narrowed, stream=mx.cpu), stream=mx.cpu).item() !=
                    mx.sum(mx.isinf(data, stream=mx.cpu), stream=mx.cpu).item()):
                raise OverflowError('value too large for float32, the '
                                    'largest float type of the Metal GPU; '
                                    "use the 'mlx:cpu' device")
            data = narrowed
        return _MLXHandle(data, stream)

    def length(self, handle):
        return handle.data.size

    def getitem(self, handle, key):
        # MLX's default stream is the GPU when Metal is available, so every
        # operation is pinned to the handle's stream explicitly.
        with self.mx.stream(self._device(handle.stream)):
            if isinstance(key, slice):
                return _MLXHandle(handle.data[key], handle.stream)
            return handle.data[key].item()

    def _reduce(self, name, handle):
        mx = self.mx
        with mx.stream(self._device(handle.stream)):
            return getattr(mx, name)(handle.data).item()

    def sum(self, handle):
        return self._reduce('sum', handle)

    def prod(self, handle):
        return self._reduce('prod', handle)

    def min(self, handle):
        return self._reduce('min', handle)

    def max(self, handle):
        return self._reduce('max', handle)

    def dot(self, a, b):
        mx = self.mx
        x, y = self._operands([a, b], a.stream, False)
        with mx.stream(self._device(a.stream)):
            if mx.issubdtype(x.dtype, mx.floating):
                return mx.inner(x, y).item()
            # matmul only supports floating types
            return mx.sum(x * y).item()

    def sort(self, handle, reverse=False):
        mx = self.mx
        with mx.stream(self._device(handle.stream)):
            data = mx.sort(handle.data)
            if reverse:
                data = data[::-1]
        return _MLXHandle(data, handle.stream)

    def _scan(self, name, handle):
        mx = self.mx
        with mx.stream(self._device(handle.stream)):
            return _MLXHandle(getattr(mx, name)(handle.data), handle.stream)

    def cumsum(self, handle):
        return self._scan('cumsum', handle)

    def cumprod(self, handle):
        return self._scan('cumprod', handle)

    def _operands(self, operands, stream, floating):
        # MLX promotes ints to float32 even on the CPU stream; promote to
        # float64 there instead so that 'mlx:cpu' keeps double precision.
        mx = self.mx
        ftype = mx.float32 if stream == 'gpu' else mx.float64
        floating = floating or any(
            isinstance(v, float) or (isinstance(v, _MLXHandle) and
                                     mx.issubdtype(v.data.dtype, mx.floating))
            for v in operands)
        result = []
        for v in operands:
            v = v.data if isinstance(v, _MLXHandle) else v
            if floating:
                if isinstance(v, (int, float)):
                    v = mx.array(v, dtype=ftype)
                elif not mx.issubdtype(v.dtype, mx.floating):
                    v = v.astype(ftype, stream=self._device(stream))
            result.append(v)
        return result

    def unary(self, op, handle):
        mx = self.mx
        floating = op not in ('abs', 'negative')
        x, = self._operands([handle], handle.stream, floating)
        with mx.stream(self._device(handle.stream)):
            return _MLXHandle(getattr(mx, op)(x), handle.stream)

    def binary(self, op, a, b):
        mx = self.mx
        stream = a.stream if isinstance(a, _MLXHandle) else b.stream
        x, y = self._operands([a, b], stream, op == 'divide')
        with mx.stream(self._device(stream)):
            return _MLXHandle(getattr(mx, op)(x, y), stream)


class _MLXHandle:
    __slots__ = ('data', 'stream')

    def __init__(self, data, stream):
        self.data = data
        self.stream = stream


_registry = {}


def register_backend(backend):
    """Register a :class:`Backend` instance under its ``name``."""
    if not isinstance(backend, Backend):
        raise TypeError(f'expected a Backend instance, '
                        f'got {type(backend).__name__}')
    if not backend.name or ':' in backend.name:
        raise ValueError(f'invalid backend name: {backend.name!r}')
    _registry[backend.name] = backend


def get_backend(name):
    try:
        return _registry[name]
    except KeyError:
        raise ValueError(f'unknown device type {name!r}; '
                         f'expected one of {sorted(_registry)}') from None


def backend_names():
    return list(_registry)


register_backend(CPUBackend())
register_backend(MLXBackend())
