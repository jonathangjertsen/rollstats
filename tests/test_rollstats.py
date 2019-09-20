import array
from collections import deque
import itertools
import math

from pytest import approx

import rollstats

nan = rollstats.nan


def check_lists_approx_equal(history, accepted):
    """Helper function to check that two lists with NaNs are approximately equal"""
    for hist, acc in itertools.zip_longest(history, accepted):
        if math.isnan(acc):
            assert math.isnan(hist)
        else:
            assert hist == approx(acc)


def test_create():
    """It should be possible to create a Container"""
    container = rollstats.Container()


def test_create_with_window():
    """It should be possible to create a Container with a window"""
    window_size = 100
    container = rollstats.Container(window_size=window_size)


def test_create_with_data():
    """It should be possible to create a Container with data.
    If no window_size is provided, the data should be the same coming in as out"""
    data = [1, 2, 3, 4, 5, 6]
    for container_type in (list, deque):
        container = rollstats.Container(data=container_type(data))
        assert container.data == deque(data)


def test_create_with_data_and_window():
    """Same as the test above, but the window size should cause only the last part
    to remain in the output"""
    container = rollstats.Container(data=[1, 2, 3], window_size=2)
    assert container.data == deque([2, 3])


def test_create_with_zero_window():
    """Negative window size => pushing has no effect"""
    container = rollstats.Container(window_size=0)

    for i in range(10):
        container.push(i)
        assert container.data == deque()


def test_create_with_one_window():
    """Negative window size => pushing has no effect"""
    container = rollstats.Container(window_size=1)

    for i in range(10):
        container.push(i)
        assert container.data == deque([i])


def test_create_with_negative_window():
    """Negative window size => pushing has no effect"""
    container = rollstats.Container(window_size=-100)

    for i in range(10):
        container.push(i)
        assert container.data == deque()


def test_push():
    """When creating a Container with no window, slicing should work as expected"""
    container = rollstats.Container()
    container.push(1)
    container.push(2)
    assert container[0] == 1
    assert container[1] == 2
    assert container[:] == [1, 2]
    assert container[1:] == [2]
    assert len(container) == 2


def test_push_n():
    """Check that container.n has the right history"""
    container = rollstats.Container(data=[0, 0, 0, 0, 0, 0], window_size=3)
    check_lists_approx_equal(container.n.history, [1, 2, 3, 3, 3, 3])


def test_push_equivalence():
    """Check that equivalence checks work as they should"""
    containers = [rollstats.Container(window_size=n) for n in (4, 4, 4, 5, 5)]

    # Push one-by-one
    containers[0].push(1)
    containers[0].push(2)
    containers[0].push(3)
    containers[0].push(4)
    containers[0].push(5)

    # Push all at once (should equal the first strategy)
    containers[1].push(1, 2, 3, 4, 5)

    # Push all at once, don't push the first (should equal the first and second)
    containers[2].push(2, 3, 4, 5)

    # Same input data, different window (should not equal the others)
    containers[3].push(1, 2, 3, 4, 5)

    # Same output data, different window (should not equal the others)
    containers[4].push(2, 3, 4, 5)

    # Check all equalities
    assert containers[0] == containers[0]
    assert containers[0] == containers[1]
    assert containers[0] == containers[2]
    assert containers[0] != containers[3]
    assert containers[0] != containers[4]
    assert containers[1] == containers[1]
    assert containers[1] == containers[0]
    assert containers[1] == containers[2]
    assert containers[1] != containers[3]
    assert containers[1] != containers[4]
    assert containers[2] == containers[0]
    assert containers[2] == containers[1]
    assert containers[2] == containers[2]
    assert containers[2] != containers[3]
    assert containers[2] != containers[4]
    assert containers[3] != containers[0]
    assert containers[3] != containers[1]
    assert containers[3] != containers[2]
    assert containers[3] == containers[3]
    assert containers[3] != containers[4]
    assert containers[4] != containers[0]
    assert containers[4] != containers[1]
    assert containers[4] != containers[2]
    assert containers[4] != containers[3]
    assert containers[4] == containers[4]


def test_equivalence_with_irrelevant_types():
    container = rollstats.Container(window_size=4)
    container.push(1, 2, 3)
    assert container != None
    assert container != [1, 2, 3]
    assert container != 1


def test_value():
    """Check that .value() always returns the last value"""
    container = rollstats.Container()
    assert math.isnan(container.value)

    container.push(1)
    assert container.value == 1

    container.push(2, 3, 4, 5)
    assert container.value == 5

    check_lists_approx_equal(container.value.history, [1, 2, 3, 4, 5])


def test_sum():
    """Check that .sum() returns the sum and has the right history"""
    container = rollstats.Container()
    assert container.sum == 0

    container.push(1, 2, -2, 2, 3, 4)
    assert container.sum == 10
    check_lists_approx_equal(container.sum.history, [1, 3, 1, 3, 6, 10])


def test_means():
    """Check that .mean() and .harmonic_mean() return the right means and have the right history"""
    container = rollstats.Container(window_size=2)
    container.subscribe_mean()
    container.subscribe_harmonic_mean()

    container.push(1, 2)
    assert container.mean == approx(3 / 2)
    assert container.harmonic_mean == approx(4 / 3)

    container.push(3)
    assert container.mean == approx(5 / 2)
    assert container.harmonic_mean == approx(12 / 5)

    container.push(4, 6)
    assert container.mean == approx(5)
    assert container.harmonic_mean == approx(24 / 5)

    check_lists_approx_equal(container.mean.history, [1, 3 / 2, 5 / 2, 7 / 2, 5])
    check_lists_approx_equal(
        container.harmonic_mean.history, [1, 4 / 3, 12 / 5, 24 / 7, 24 / 5]
    )


def test_var_and_std_and_zscore():
    """Check that the var, std, and zscore calculations are correct and have the right history"""
    container = rollstats.Container(window_size=3)
    container.subscribe_var()
    container.subscribe_std()
    container.subscribe_pop_var()
    container.subscribe_pop_std()
    container.subscribe_z_score()

    # Variance-based measurements should be those of (1,1,1)
    container.push(1, 1, 1)
    assert container.var == approx(0)
    assert container.std == approx(0)
    assert container.pop_var == approx(0)
    assert container.pop_std == approx(0)
    assert math.isnan(container.zscore)

    # Variance-based measurements should be those of (1,1,0)
    container.push(0)
    assert container.var == approx(1 / 3)
    assert container.std == approx(math.sqrt(1 / 3))
    assert container.pop_var == approx(2 / 9)
    assert container.pop_std == approx(math.sqrt(2 / 9))
    assert container.zscore == approx(-2 * math.sqrt(1 / 3))

    # Check the histories
    check_lists_approx_equal(container.var.history, [nan, 0, 0, 1 / 3])
    check_lists_approx_equal(container.std.history, [nan, 0, 0, math.sqrt(1 / 3)])
    check_lists_approx_equal(container.pop_var.history, [nan, 0, 0, 2 / 9])
    check_lists_approx_equal(container.pop_std.history, [nan, 0, 0, math.sqrt(2 / 9)])
    check_lists_approx_equal(
        container.zscore.history, [nan, nan, nan, -2 * math.sqrt(1 / 3)]
    )


def test_custom_subscription():
    container = rollstats.Container(window_size=3)
    container.subscribe("mean_plus_1", container.M, func=lambda m: m + 1)
    container.push(2, 1, 3)
    assert container.mean_plus_1 == 3
