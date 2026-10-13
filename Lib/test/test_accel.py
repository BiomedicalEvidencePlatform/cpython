import array
import functools
import itertools
import math
import operator
import pickle
import random
import statistics
import unittest
from fractions import Fraction

import accel
from accel import _backends


class FakeBackend(_backends.CPUBackend):
    """A non-cpu device that computes like the cpu backend and counts
    calls, so that dispatching can be tested without MLX."""

    name = 'fake'
    streams = ('main', 'other')

    def __init__(self):
        self.calls = 0

    def from_host(self, buf, stream):
        self.calls += 1
        return super().from_host(buf, stream)


fake = FakeBackend()
accel.register_backend(fake)


class DeviceTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(str(accel.Device('cpu')), 'cpu')
        self.assertEqual(str(accel.Device('fake')), 'fake:main')
        self.assertEqual(str(accel.Device('fake:other')), 'fake:other')
        self.assertEqual(accel.Device('metal'), accel.Device('mlx:gpu'))
        self.assertEqual(accel.Device('mlx'), 'mlx:gpu')
        self.assertEqual(accel.Device(accel.Device('fake')), 'fake:main')
        self.assertEqual(repr(accel.Device('cpu')), "accel.Device('cpu')")
        self.assertEqual(hash(accel.Device('fake')),
                         hash(accel.Device('fake:main')))

    def test_parse_errors(self):
        with self.assertRaises(ValueError):
            accel.Device('tpu')
        with self.assertRaises(ValueError):
            accel.Device('fake:nope')
        with self.assertRaises(TypeError):
            accel.Device(0)

    def test_pickle(self):
        dev = accel.Device('fake:other')
        self.assertEqual(pickle.loads(pickle.dumps(dev)), dev)

    def test_available(self):
        self.assertTrue(accel.is_available('cpu'))
        self.assertTrue(accel.is_available('fake'))
        self.assertFalse(accel.is_available('tpu'))
        self.assertIn(accel.Device('cpu'), accel.available_devices())

    def test_default_device(self):
        self.assertEqual(accel.get_default_device(), 'cpu')
        accel.set_default_device('fake')
        try:
            self.assertEqual(accel.to_device([1]).device, 'fake')
        finally:
            accel.set_default_device('cpu')

    def test_register_backend_errors(self):
        with self.assertRaises(TypeError):
            accel.register_backend(object())
        bad = FakeBackend()
        bad.name = 'a:b'
        with self.assertRaises(ValueError):
            accel.register_backend(bad)


class DeviceArrayTests(unittest.TestCase):
    def test_typecode_inference(self):
        self.assertEqual(accel.to_device([1, 2, True]).typecode, 'q')
        self.assertEqual(accel.to_device([1, 2.5]).typecode, 'd')
        self.assertEqual(accel.to_device(range(3)).typecode, 'q')
        self.assertEqual(accel.to_device(iter([1.0])).typecode, 'd')
        self.assertEqual(accel.to_device([1], typecode='f').typecode, 'f')
        self.assertEqual(accel.to_device([]).typecode, 'q')

    def test_unsupported(self):
        for data in (['a'], [Fraction(1, 2)], [1j], [None]):
            with self.subTest(data=data), self.assertRaises(TypeError):
                accel.to_device(data)
        with self.assertRaises(TypeError):
            accel.to_device(array.array('w', 'abc'))
        with self.assertRaises(OverflowError):
            accel.to_device([2**70])
        with self.assertRaises(ValueError):
            accel.to_device([1], typecode='w')
        with self.assertRaises(TypeError):
            accel.DeviceArray()

    def test_from_buffers(self):
        a = array.array('i', [1, -2, 3])
        self.assertEqual(accel.to_device(a).tolist(), [1, -2, 3])
        self.assertEqual(accel.to_device(memoryview(a)).typecode, 'i')
        self.assertEqual(accel.to_device(b'\x01\x02').tolist(), [1, 2])
        with self.assertRaises(TypeError):
            accel.to_device(memoryview(b'abcd').cast('B', (2, 2)))

    def test_array_to_device(self):
        a = array.array('d', [1.0, 2.5])
        x = a.to_device()
        self.assertIsInstance(x, accel.DeviceArray)
        self.assertEqual(x.device, 'cpu')
        self.assertEqual(x.toarray(), a)
        self.assertEqual(a.to_device('fake').device, 'fake:main')
        self.assertEqual(a.to_device(typecode='f').typecode, 'f')
        with self.assertRaises(ValueError):
            a.to_device('tpu')
        with self.assertRaises(TypeError):
            array.array('w', 'x').to_device()

    def test_sequence(self):
        x = accel.to_device([3, 1, 2], 'fake')
        self.assertEqual(len(x), 3)
        self.assertEqual(x[0], 3)
        self.assertEqual(x[-1], 2)
        self.assertIsInstance(x[0], int)
        self.assertEqual(x[1:].tolist(), [1, 2])
        self.assertEqual(x[1:].device, 'fake:main')
        self.assertEqual(list(x), [3, 1, 2])
        self.assertEqual(list(reversed(x)), [2, 1, 3])
        self.assertIn(1, x)
        self.assertNotIn(5, x)
        with self.assertRaises(IndexError):
            x[3]
        with self.assertRaises(IndexError):
            x[-4]
        self.assertIsInstance(accel.to_device([1.0])[0], float)

    def test_equality_and_hash(self):
        x = accel.to_device([1, 2])
        self.assertEqual(x, accel.to_device([1, 2], 'fake'))
        self.assertNotEqual(x, accel.to_device([2, 1]))
        self.assertNotEqual(x, [1, 2])
        with self.assertRaises(TypeError):
            hash(x)

    def test_buffer_protocol(self):
        x = accel.to_device([1.5, 2.5], 'fake')
        with memoryview(x) as mv:
            self.assertEqual(mv.format, 'd')
            self.assertEqual(mv.tolist(), [1.5, 2.5])

    def test_repr_and_pickle(self):
        x = accel.to_device([1, 2], 'fake')
        self.assertEqual(repr(x),
                         "accel.DeviceArray('q', [1, 2], device='fake:main')")
        y = pickle.loads(pickle.dumps(x))
        self.assertEqual(y, x)
        self.assertEqual(y.device, x.device)

    def test_to_device(self):
        x = accel.to_device([1, 2])
        self.assertIs(x.to_device('cpu'), x)
        self.assertIs(accel.to_device(x, 'cpu'), x)
        y = x.to_device('fake')
        self.assertEqual(y.device, 'fake')
        self.assertEqual(y.cpu().device, 'cpu')
        self.assertEqual(y.cpu(), x)
        self.assertEqual(accel.to_device(x, typecode='d').typecode, 'd')

    def test_reductions(self):
        for dev in ('cpu', 'fake'):
            with self.subTest(dev=dev):
                x = accel.to_device([3, -1, 2], dev)
                self.assertEqual(x.sum(), 4)
                self.assertEqual(x.prod(), -6)
                self.assertEqual(x.min(), -1)
                self.assertEqual(x.max(), 3)
                self.assertEqual(x.mean(), 4 / 3)
                self.assertEqual(x.dot([1, 1, 1]), 4)
                self.assertEqual(x.sorted().tolist(), [-1, 2, 3])
                self.assertEqual(x.sorted(reverse=True).tolist(), [3, 2, -1])
                self.assertEqual(x.cumsum().tolist(), [3, 2, 4])
                self.assertEqual(x.cumprod().tolist(), [3, -3, -6])
                empty = accel.to_device([], dev)
                self.assertEqual(empty.sum(), 0)
                for method in (empty.min, empty.max, empty.mean):
                    with self.assertRaises(ValueError):
                        method()
                with self.assertRaises(ValueError):
                    x.dot([1])

    def test_float_sum_is_exact_on_cpu(self):
        data = [1e100, 1.0, -1e100] * 10
        self.assertEqual(accel.to_device(data).sum(), math.fsum(data))

    def test_int_overflow_is_exact(self):
        # Device integers are int64; results that may not fit are
        # computed with Python ints instead.
        big = [2**61] * 4
        x = accel.to_device(big, 'fake')
        self.assertEqual(x.sum(), 2**63)
        self.assertEqual(x.dot(x), 4 * 2**122)
        self.assertEqual(accel.to_device(range(1, 30), 'fake').prod(),
                         math.prod(range(1, 30)))
        with self.assertRaises(OverflowError):
            x.cumsum()


class MapTests(unittest.TestCase):
    def check(self, func, *args, dev='fake'):
        expected = list(map(func, *args))
        result = accel.map(func, *args, device=dev)
        self.assertIsInstance(result, accel.DeviceArray)
        self.assertEqual(result.device, dev)
        self.assertEqual(result.tolist(), expected)
        return result

    def test_unary(self):
        for func in (abs, math.fabs, operator.neg, math.sqrt, math.exp,
                     math.log, math.sin, math.cos, math.tan, math.tanh):
            with self.subTest(func=func):
                self.check(func, [0.5, 1.0, 4.0])
        self.assertEqual(self.check(abs, [-1, 2]).typecode, 'q')
        self.assertEqual(self.check(math.sqrt, [4, 9]).typecode, 'd')

    def test_binary(self):
        for func in (operator.add, operator.sub, operator.mul,
                     operator.truediv, max, min):
            with self.subTest(func=func):
                self.check(func, [1, 2, 3], [3, 2, 1])
        self.check(operator.pow, [2.0, 3.0], [2.0, 0.5])
        self.check(math.pow, [2.0, 3.0], [2.0, 0.5])
        self.assertEqual(self.check(operator.truediv, [1, 2], [2, 2])
                         .typecode, 'd')

    def test_device_selection(self):
        x = accel.to_device([1.0], 'fake:other')
        self.assertEqual(accel.map(math.sqrt, x).device, 'fake:other')
        self.assertEqual(accel.map(math.sqrt, [1.0]).device, 'cpu')
        self.assertEqual(accel.map(operator.add, [1.0], x).device,
                         'fake:other')

    def test_python_exceptions(self):
        # Inputs where a device would return nan/inf or wrap around are
        # computed exactly, raising what builtins.map() would raise.
        cases = [
            (ValueError, math.sqrt, [4.0, -1.0]),
            (ValueError, math.log, [1.0, 0.0]),
            (OverflowError, math.exp, [1.0, 1000.0]),
            (ZeroDivisionError, operator.truediv, [1, 2], [1, 0]),
            (OverflowError, operator.mul, [2**40], [2**40]),
        ]
        for exc, func, *args in cases:
            with self.subTest(func=func), self.assertRaises(exc):
                accel.map(func, *args, device='fake')

    def test_errors(self):
        with self.assertRaises(TypeError):
            accel.map(str, [1])
        with self.assertRaises(TypeError):
            accel.map(lambda x: x, [1])
        with self.assertRaises(TypeError):
            accel.map(math.sqrt, [1], [2])
        with self.assertRaises(TypeError):
            accel.map(operator.add, [1], [2], [3])
        with self.assertRaises(ValueError):
            accel.map(operator.add, [1], [2, 3])


class ModuleBackendTests(unittest.TestCase):
    def setUp(self):
        self.originals = {name: getattr(statistics, name)
                          for name in ('mean', 'fmean', 'median')}
        self.addCleanup(accel.set_backend, statistics, 'cpu')
        self.addCleanup(accel.set_backend, math, 'cpu')
        self.addCleanup(accel.set_backend, functools, 'cpu')
        self.addCleanup(accel.set_backend, itertools, 'cpu')

    def test_switch_and_restore(self):
        self.assertEqual(accel.get_backend(statistics), 'cpu')
        accel.set_backend(statistics, 'fake')
        self.assertEqual(accel.get_backend(statistics), 'fake')
        self.assertEqual(accel.get_backend('statistics'), 'fake')
        self.assertIsNot(statistics.mean, self.originals['mean'])
        self.assertIs(statistics.mean.__wrapped__, self.originals['mean'])
        self.assertEqual(statistics.mean.__name__, 'mean')
        # Switching again keeps a single level of wrapping.
        accel.set_backend(statistics, 'fake:other')
        self.assertIs(statistics.mean.__wrapped__, self.originals['mean'])
        accel.set_backend(statistics, 'cpu')
        for name, func in self.originals.items():
            self.assertIs(getattr(statistics, name), func)
        self.assertEqual(accel.get_backend(statistics), 'cpu')

    def test_context_manager(self):
        with accel.backend(statistics, 'fake', min_size=0) as dev:
            self.assertEqual(dev, 'fake')
            self.assertEqual(accel.get_backend(statistics), 'fake')
            with accel.backend(statistics, 'cpu'):
                self.assertIs(statistics.mean, self.originals['mean'])
            self.assertEqual(accel.get_backend(statistics), 'fake')
        self.assertIs(statistics.mean, self.originals['mean'])

    def test_errors(self):
        with self.assertRaises(ValueError):
            accel.set_backend('os', 'fake')
        with self.assertRaises(TypeError):
            accel.set_backend(1, 'fake')
        with self.assertRaises(ValueError):
            accel.set_backend(statistics, 'tpu')
        with self.assertRaises(ValueError):
            accel.set_backend(statistics, 'fake', min_size=-1)

    def test_min_size(self):
        accel.set_backend(statistics, 'fake', min_size=10)
        calls = fake.calls
        self.assertEqual(statistics.mean([1, 2, 3]), 2)
        self.assertEqual(fake.calls, calls)
        self.assertEqual(statistics.mean(range(10)), 4.5)
        self.assertEqual(fake.calls, calls + 1)

    def test_statistics_matches(self):
        random.seed(12345)
        floats = [random.gauss(10, 3) for _ in range(500)]
        ints = [random.randrange(-1000, 1000) for _ in range(500)]
        positive = [abs(v) + 1 for v in floats]
        funcs = ['fmean', 'mean', 'median', 'variance', 'pvariance',
                 'stdev', 'pstdev']
        expected = {}
        for name in funcs:
            for kind, data in (('float', floats), ('int', ints),
                               ('odd', ints[:-1])):
                expected[name, kind] = getattr(statistics, name)(data)
        expected['geometric_mean'] = statistics.geometric_mean(positive)
        expected['covariance'] = statistics.covariance(floats, ints)
        expected['correlation'] = statistics.correlation(floats, ints)

        accel.set_backend(statistics, 'fake', min_size=0)
        calls = fake.calls
        for name in funcs:
            for kind, data in (('float', floats), ('int', ints),
                               ('odd', ints[:-1])):
                with self.subTest(name=name, kind=kind):
                    got = getattr(statistics, name)(data)
                    want = expected[name, kind]
                    self.assertIs(type(got), type(want))
                    if kind == 'float':
                        self.assertAlmostEqual(got, want, places=9)
                    else:
                        # Exact on integer data, like statistics.
                        self.assertEqual(got, want)
        self.assertAlmostEqual(statistics.geometric_mean(positive),
                               expected['geometric_mean'], places=9)
        self.assertAlmostEqual(statistics.covariance(floats, ints),
                               expected['covariance'], places=6)
        self.assertAlmostEqual(statistics.correlation(floats, ints),
                               expected['correlation'], places=12)
        self.assertGreater(fake.calls, calls)

    def test_statistics_fallbacks(self):
        accel.set_backend(statistics, 'fake', min_size=0)
        calls = fake.calls
        # Not numeric, exact types, and arguments the kernels don't handle.
        self.assertEqual(statistics.median(['a', 'c', 'b']), 'b')
        self.assertEqual(statistics.mean([Fraction(1, 3)] * 3),
                         Fraction(1, 3))
        self.assertEqual(statistics.fmean([1, 2], weights=[1, 3]), 1.75)
        self.assertEqual(statistics.variance([1, 2, 3], xbar=2), 1)
        self.assertEqual(statistics.correlation([1, 2, 3], [3, 1, 2],
                                                method='ranked'), -0.5)
        self.assertAlmostEqual(statistics.geometric_mean([1.0, 2.0, 4.0]),
                               2.0)
        self.assertEqual(fake.calls, calls + 1)  # geometric_mean
        # Errors come from the original functions.
        with self.assertRaises(statistics.StatisticsError):
            statistics.mean([])
        with self.assertRaises(statistics.StatisticsError):
            statistics.variance([1])
        with self.assertRaises(statistics.StatisticsError):
            statistics.covariance([1, 2], [1])
        with self.assertRaises(statistics.StatisticsError):
            statistics.correlation([1, 1, 1], [1, 2, 3])
        with self.assertRaises(statistics.StatisticsError):
            statistics.geometric_mean([1.0, -1.0])

    def test_iterators_are_not_lost(self):
        accel.set_backend(statistics, 'fake', min_size=0)
        self.assertEqual(statistics.mean(iter([1, 2, 3])), 2)
        self.assertEqual(statistics.median(iter(['a', 'b', 'c'])), 'b')
        self.assertEqual(statistics.covariance(iter([1, 2, 3]),
                                               iter([1, 2, 3])), 1.0)
        accel.set_backend(math, 'fake', min_size=0)
        self.assertEqual(math.prod(iter(['a', 2])), 'aa')

    def test_device_arrays_stay_on_their_device(self):
        accel.set_backend(statistics, 'fake', min_size=1000)
        x = accel.to_device([1.0, 2.0, 4.0], 'fake:other')
        calls = fake.calls
        self.assertEqual(statistics.fmean(x), 7 / 3)
        self.assertEqual(statistics.median(x), 2.0)
        self.assertEqual(fake.calls, calls)
        # With the module on the cpu, DeviceArrays are plain iterables.
        accel.set_backend(statistics, 'cpu')
        self.assertEqual(statistics.fmean(x), 7 / 3)

    def test_math(self):
        accel.set_backend(math, 'fake', min_size=0)
        self.assertEqual(math.prod([1.5, 2.0, 4.0]), 12.0)
        self.assertEqual(math.prod(range(1, 40)), math.factorial(39))
        self.assertEqual(math.prod([2, 3], start=2), 12)
        self.assertEqual(math.prod([]), 1)
        self.assertEqual(math.sumprod([1, 2, 3], [4, 5, 6]), 32)
        self.assertEqual(math.sumprod([1.5, 2.0], [2.0, 4.0]), 11.0)
        with self.assertRaises(ValueError):
            math.sumprod([1, 2], [1])

    def test_functools_reduce(self):
        accel.set_backend(functools, 'fake', min_size=0)
        reduce = functools.reduce
        self.assertEqual(reduce(operator.add, [1, 2, 3]), 6)
        self.assertEqual(reduce(operator.add, [1, 2, 3], 10), 16)
        self.assertEqual(reduce(operator.add, [1, 2, 3], initial=10), 16)
        self.assertEqual(reduce(operator.mul, range(1, 6)), 120)
        self.assertEqual(reduce(max, [3, 9, 2]), 9)
        self.assertEqual(reduce(min, [3, 9, 2], -1), -1)
        self.assertEqual(reduce(operator.add, ['a', 'b']), 'ab')
        self.assertEqual(reduce(lambda a, b: a - b, [10, 1, 2]), 7)
        self.assertEqual(reduce(operator.add, [], 5), 5)
        with self.assertRaises(TypeError):
            reduce(operator.add, [])

    def test_itertools_accumulate(self):
        accel.set_backend(itertools, 'fake', min_size=0)
        acc = itertools.accumulate
        self.assertEqual(list(acc([1, 2, 3])), [1, 3, 6])
        self.assertEqual(list(acc([1, 2, 3], operator.mul)), [1, 2, 6])
        self.assertEqual(list(acc([1.5, 2.5])), [1.5, 4.0])
        self.assertEqual(list(acc([1, 2], initial=10)), [10, 11, 13])
        self.assertEqual(list(acc([1, 5, 2], max)), [1, 5, 5])
        self.assertEqual(list(acc(['a', 'b'])), ['a', 'ab'])

    def test_register_kernel(self):
        import accel._dispatch as dispatch
        self.addCleanup(dispatch._kernels['statistics'].pop, 'mode', None)
        accel.set_backend(statistics, 'fake')
        seen = []
        def kernel(original, device, min_size, data):
            seen.append(device)
            return NotImplemented
        accel.register_kernel(statistics, 'mode', kernel)
        self.assertEqual(statistics.mode([1, 1, 2]), 1)
        self.assertEqual(seen, [accel.Device('fake')])
        accel.set_backend(statistics, 'cpu')
        self.assertEqual(statistics.mode([1, 1, 2]), 1)
        self.assertEqual(len(seen), 1)


@unittest.skipUnless(accel.is_available('mlx'), 'requires mlx')
class MLXTests(unittest.TestCase):
    def devices(self):
        return [d for d in accel.available_devices() if d.type == 'mlx']

    def test_roundtrip(self):
        for dev in self.devices():
            for typecode in 'bBhHiIlLqQfd':
                with self.subTest(dev=dev, typecode=typecode):
                    a = array.array(typecode, [0, 1, 2, 100])
                    x = a.to_device(dev)
                    self.assertEqual(x.device, dev)
                    self.assertEqual(x.toarray(), a)

    def test_float64_precision(self):
        x = accel.to_device([0.1, 1 / 3], 'mlx:cpu')
        self.assertEqual(x.storage_typecode, 'd')
        self.assertEqual(x.tolist(), [0.1, 1 / 3])
        self.assertEqual(accel.map(math.sqrt, [2], device='mlx:cpu')[0],
                         math.sqrt(2))
        self.assertEqual(accel.to_device([1, 2], 'mlx:cpu').mean(), 1.5)

    @unittest.skipUnless(accel.is_available('mlx:gpu'), 'requires Metal')
    def test_streams_share_memory(self):
        x = accel.to_device([1, 2, 3], 'mlx:cpu')
        y = x.to_device('mlx:gpu')
        self.assertEqual(y.device, 'mlx:gpu')
        self.assertEqual(y.tolist(), [1, 2, 3])

    @unittest.skipUnless(accel.is_available('mlx:gpu'), 'requires Metal')
    def test_gpu_float32(self):
        x = accel.to_device([0.1, 0.2], 'mlx:gpu')
        self.assertEqual(x.storage_typecode, 'f')
        self.assertAlmostEqual(x.sum(), 0.3, places=6)
        back = x.to_device('mlx:cpu')
        self.assertEqual(back.storage_typecode, 'd')
        self.assertAlmostEqual(back[0], 0.1, places=6)
        with self.assertRaises(OverflowError):
            accel.map(math.exp, [100.0], device='mlx:gpu')

    def test_exact_fallbacks(self):
        for dev in self.devices():
            with self.subTest(dev=dev):
                big = accel.to_device([2**61] * 4, dev)
                self.assertEqual(big.sum(), 2**63)
                with self.assertRaises(ValueError):
                    accel.map(math.sqrt, [-1.0], device=dev)
                with self.assertRaises(ZeroDivisionError):
                    accel.map(operator.truediv, [1], [0], device=dev)

    def test_statistics(self):
        random.seed(0)
        data = [random.random() for _ in range(5000)]
        ints = [random.randrange(-100, 100) for _ in range(5000)]
        expected = (statistics.fmean(data), statistics.median(data),
                    statistics.stdev(data), statistics.variance(ints))
        for dev in self.devices():
            places = 9 if dev.stream == 'cpu' else 4
            with self.subTest(dev=dev), accel.backend(statistics, dev):
                self.assertAlmostEqual(statistics.fmean(data), expected[0],
                                       places=places)
                self.assertAlmostEqual(statistics.median(data), expected[1],
                                       places=places)
                self.assertAlmostEqual(statistics.stdev(data), expected[2],
                                       places=places)
                self.assertEqual(statistics.variance(ints), expected[3])


if __name__ == '__main__':
    unittest.main()
