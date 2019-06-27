import cProfile
import pstats
import time

import rollstats

def parametrized_test(window, pushes):
    container = rollstats.Container(window_size=window)
    container.subscribe_z_score()
    for i in range(pushes):
        container.push(i)

def profiled_test():
    container = rollstats.Container(window_size=100)
    for i in range(100000):
        container.push(i)

def run_profiled_test():
    statfile = 'stats'
    cProfile.run('profiled_test()', statfile)
    stats = pstats.Stats(statfile)
    stats.strip_dirs().sort_stats('tottime').print_stats(10)

def run_parametrized_test():
    windows = [-1, 0, 1, 10, 100, 1000, 10000, 100000]
    pushess = [1, 10, 100, 1000, 10000, 100000]
    results = {}
    for window in windows:
        for pushes in pushess:
            print(f"pushes: {pushes}, window: {window} ... ", end="")

            start = time.perf_counter()
            parametrized_test(window, pushes)
            stop = time.perf_counter()

            diff = stop - start
            diff_per = diff / pushes
            print(f"{diff_per*1e6:.2f} us per push, {diff:.2f} s in total.")
            results[(window, pushes)] = diff_per
    return results

def run_test():
    print("Running parametrized test...")
    print(run_parametrized_test())

    print("\nRunning profiled test...")
    run_profiled_test()


if __name__ == "__main__":
    run_test()