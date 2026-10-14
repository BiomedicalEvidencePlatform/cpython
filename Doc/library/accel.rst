:mod:`!accel` --- Compute devices for numeric collections
========================================================

.. module:: accel
   :synopsis: Run numeric collections and standard library functions on a
              GPU or other compute device.

.. versionadded:: next

**Source code:** :source:`Lib/accel/`

--------------

This module moves homogeneous numeric data to a compute *device*, such as
the Metal GPU of an Apple silicon Mac through `MLX
<https://github.com/ml-explore/mlx>`_, and runs standard library functions
on it.  It offers two levels of control, similar to ``tensor.to(device)``
in array libraries:

* Per *object*: :func:`to_device` (or :meth:`array.array.to_device`)
  returns a :class:`DeviceArray`, whose reductions run on its device.
  Functions of modules switched to a device use DeviceArrays where they
  are, without copying them.
* Per *module*: :func:`set_backend` makes the supported functions of
  :mod:`statistics`, :mod:`math`, :mod:`functools` and :mod:`itertools`
  move their inputs to a device and compute there.

::

   >>> import accel, random, statistics
   >>> accel.available_devices()
   [accel.Device('cpu'), accel.Device('mlx:gpu'), accel.Device('mlx:cpu')]
   >>> data = [random.random() for _ in range(1_000_000)]
   >>> accel.set_backend(statistics, 'mlx')
   >>> statistics.stdev(data)            # computed on the GPU
   0.2886...
   >>> accel.set_backend(statistics, 'cpu')

   >>> xs = accel.to_device(data, 'mlx')
   >>> xs.sum(), xs.max(), statistics.median(xs)

Only :class:`int` and :class:`float` data can move to a device.  Any other
input -- strings, :class:`~fractions.Fraction`, :class:`~decimal.Decimal`,
mixed types, ints that do not fit in 64 bits, or fewer elements than the
module's *min_size* -- is passed unchanged to the regular implementation,
so switching a module to a device never changes which inputs it accepts.

The ``mlx`` device requires the third-party ``mlx`` package.  Built-in
lists, dicts and sets cannot be moved to a device: their items are
arbitrary Python objects, not machine numbers.  Convert them with
:func:`to_device` first.


Precision and exactness
-----------------------

The Metal GPU has no 64-bit floating-point type, so float data on
``mlx:gpu`` is computed in single precision (about 7 significant digits).
Use ``mlx:cpu``, which has the same unified memory, when you need double
precision.  Floating-point sums on any device other than ``cpu`` are not
correctly rounded the way :func:`math.fsum` and :mod:`statistics` are;
results differ in the last few bits.

Operations whose device result could differ from Python's in a bigger
way are computed exactly on the CPU instead:

* integer results that may not fit in 64 bits (Python ints are unbounded);
* :func:`map` inputs that would raise in Python -- negative numbers for
  :func:`math.sqrt`, non-positive numbers for :func:`math.log`, division
  by zero and :func:`math.exp` overflow.  The same exception is raised.

:mod:`statistics` functions on integer data return the same exact results
and types as without a device.  Exceptions such as
:exc:`statistics.StatisticsError` always come from the original function.


Devices
-------

.. class:: Device(spec)

   A compute device.  *spec* is a string ``'type[:stream]'``, or another
   Device.  The built-in devices are:

   ===============  ====================================================
   Spec             Meaning
   ===============  ====================================================
   ``'cpu'``        Pure Python; always available.  Exact.
   ``'mlx:gpu'``    MLX on the Metal GPU.  Aliases: ``'mlx'``,
                    ``'metal'``, ``'gpu'``.
   ``'mlx:cpu'``    MLX on the CPU, with float64 support.
   ===============  ====================================================

   MLX arrays live in unified memory, so moving data between ``mlx:gpu``
   and ``mlx:cpu`` does not copy it (except to change float64 to float32).

   Devices compare equal to their spec strings.

   .. attribute:: type

      The device type, such as ``'mlx'``.

   .. attribute:: stream

      The stream (sub-device) within the backend, such as ``'gpu'``.

.. function:: available_devices()

   Return a list of the devices that can be used.

.. function:: is_available(device)

   Return ``True`` if *device* (a device type or spec) can be used.

.. function:: get_default_device()
              set_default_device(device)

   Get or set the device :func:`to_device` uses when none is given.
   It is ``'cpu'`` initially.


Device arrays
-------------

.. function:: to_device(obj, device=None, *, typecode=None)

   Return a :class:`DeviceArray` with the contents of *obj* on *device*.

   *obj* may be a :class:`DeviceArray`, an :class:`array.array`, a
   one-dimensional numeric buffer such as a :class:`memoryview`, or any
   iterable of ints and floats.  Lists of ints are stored as 64-bit
   integers (typecode ``'q'``), lists containing a float as 64-bit floats
   (``'d'``).  Pass an :mod:`array` *typecode* to choose another type.

   Raises :exc:`TypeError` for non-numeric data and :exc:`OverflowError`
   if a value does not fit in the type.

.. class:: DeviceArray

   A read-only sequence of numbers stored on a :class:`Device`.  It
   supports :func:`len`, indexing, slicing (the result stays on the
   device), iteration, ``in``, ``==`` with other DeviceArrays, pickling,
   and the buffer protocol (which copies the data to the host).

   .. attribute:: device

   .. attribute:: typecode

      The :mod:`array` typecode of the data as seen from Python.

   .. attribute:: storage_typecode

      The typecode used on the device; ``'f'`` for ``'d'`` data on the
      Metal GPU.

   .. method:: to_device(device)
               cpu()

      Return the data on *device* (on ``'cpu'``), or ``self`` if it is
      already there.

   .. method:: toarray()
               tolist()

      Copy the data to a new :class:`array.array` or :class:`list`.

   .. method:: sum()
               prod()
               min()
               max()
               mean()
               dot(other)

      Reductions computed on the device; they return Python numbers.

   .. method:: sorted(*, reverse=False)
               cumsum()
               cumprod()

      Return a new DeviceArray on the same device.

.. function:: map(function, *iterables, device=None)

   Vectorized :func:`map`: return a :class:`DeviceArray` with *function*
   applied to the elements of one or two equal-length *iterables*.

   *function* must be one of :func:`abs`, :func:`math.fabs`,
   :func:`operator.neg`, :func:`math.sqrt`, :func:`math.exp`,
   :func:`math.log`, :func:`math.sin`, :func:`math.cos`, :func:`math.tan`,
   :func:`math.tanh` (one iterable) or :func:`operator.add`,
   :func:`operator.sub`, :func:`operator.mul`, :func:`operator.truediv`,
   :func:`operator.pow`, :func:`math.pow`, :func:`max`, :func:`min` (two
   iterables).

   The computation runs on *device*, else on the device of the first
   DeviceArray argument, else on the default device.


Module backends
---------------

.. function:: set_backend(module, device, *, min_size=4096)

   Run the supported functions of *module* (a module or its name) on
   *device*.  Pass ``'cpu'`` to restore the original functions.

   Inputs with fewer than *min_size* elements stay on the CPU, because
   moving them costs more than the computation.  :class:`DeviceArray`
   inputs are always used on their own device.

   The supported functions are:

   * :mod:`statistics`: :func:`~statistics.mean`, :func:`~statistics.fmean`
     (without weights), :func:`~statistics.geometric_mean`,
     :func:`~statistics.median`, :func:`~statistics.variance`,
     :func:`~statistics.pvariance`, :func:`~statistics.stdev`,
     :func:`~statistics.pstdev` (without a precomputed mean),
     :func:`~statistics.covariance` and :func:`~statistics.correlation`
     (``method='linear'``).
   * :mod:`math`: :func:`~math.prod` and :func:`~math.sumprod`.
     :func:`math.fsum` is deliberately not accelerated: its purpose is a
     correctly rounded result.
   * :mod:`functools`: :func:`~functools.reduce` with :func:`operator.add`,
     :func:`operator.mul`, :func:`max` or :func:`min`.
   * :mod:`itertools`: :func:`~itertools.accumulate` with no function,
     :func:`operator.add` or :func:`operator.mul`, and no *initial*.
     The results are computed eagerly.

   The functions are replaced in the module's namespace, so only calls
   through the module are affected: a name bound earlier with
   ``from statistics import mean`` still refers to the original.  The
   replacements keep the original in their ``__wrapped__`` attribute.

.. function:: get_backend(module)

   Return the :class:`Device` that *module* runs on.

.. function:: backend(module, device, *, min_size=4096)

   Context manager that switches *module* to *device* and restores the
   previous device on exit.

.. function:: register_kernel(module, name, kernel)

   Register a device implementation of the function *module.name*, for
   modules of your own or functions not listed above.  The kernel is
   called as ``kernel(original, device, min_size, *args, **kwargs)`` and
   returns the result, or :data:`NotImplemented` to call
   ``original(*args, **kwargs)`` instead.


Backends
--------

.. class:: Backend

   Base class for compute backends.  A backend stores the data of a
   :class:`DeviceArray` in an opaque handle and implements a small set of
   primitives on it; see the docstrings in :source:`Lib/accel/_backends.py`.
   :class:`CPUBackend` and :class:`MLXBackend` are the built-in ones.

.. function:: register_backend(backend)

   Register a :class:`Backend` instance under its ``name``, making it
   usable as a device type.
