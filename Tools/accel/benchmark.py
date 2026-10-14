"""Benchmark statistics functions on accel devices.

Compares the regular standard library functions with the same functions
switched to each available accel device by accel.set_backend().  The data
is moved to each device before timing, so the device columns measure the
computation only; the cost of the move is reported separately.

Usage:

    python Tools/accel/benchmark.py [-n SIZE] [-r REPEAT] [FUNC ...]

The mlx devices require the third-party mlx package.
"""

import argparse
import random
import statistics
import time

import accel


DEFAULT_FUNCS = ['fmean', 'stdev', 'median']


def best_time(func, repeat):
    """Return the best wall time of *repeat* calls, in milliseconds,
    after one warm-up call."""
    func()
    best = float('inf')
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best * 1e3


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('funcs', nargs='*', default=DEFAULT_FUNCS,
                        metavar='FUNC',
                        help=f'statistics functions to time '
                             f'(default: {" ".join(DEFAULT_FUNCS)})')
    parser.add_argument('-n', '--size', type=int, default=10_000_000,
                        help='number of floats (default: %(default)s)')
    parser.add_argument('-r', '--repeat', type=int, default=3,
                        help='timed runs per measurement; the best is '
                             'reported (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    devices = [d for d in accel.available_devices() if d.type != 'cpu']
    random.seed(args.seed)
    data = [random.random() for _ in range(args.size)]

    print(f'{args.size:,} floats, best of {args.repeat}')
    on_device = {}
    for dev in devices:
        ms = best_time(lambda: accel.to_device(data, dev), args.repeat)
        print(f'to_device(list, {str(dev)!r}): {ms:.0f} ms')
        on_device[dev] = accel.to_device(data, dev)
    print()

    header = ['function', 'stdlib'] + [str(dev) for dev in devices]
    rows = []
    for name in args.funcs:
        # Look the function up on every call: set_backend() replaces it in
        # the module, so a reference taken earlier would stay on the CPU.
        row = [f'statistics.{name}',
               best_time(lambda: getattr(statistics, name)(data),
                         args.repeat)]
        for dev in devices:
            with accel.backend(statistics, dev):
                x = on_device[dev]
                row.append(best_time(lambda: getattr(statistics, name)(x),
                                     args.repeat))
        rows.append(row)

    print('| ' + ' | '.join(header) + ' |')
    print('|' + '---|' * len(header))
    for name, *times in rows:
        print(f'| `{name}` | ' + ' | '.join(f'{t:.1f} ms' for t in times)
              + ' |')


if __name__ == '__main__':
    main()
