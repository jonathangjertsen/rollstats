from pytest import approx

import rollstats

nan = rollstats.nan


def check_history(f, data):
    assert f.history == rollstats.MemoryFloat.new_history_container(data)


def test_memfloat():
    """Check that a MemoryFloat can be created"""
    f = rollstats.MemoryFloat(0.0)


def test_operations():
    """Check that we can do the right operations on the MemoryFloat"""
    f = rollstats.MemoryFloat(0.0)
    assert f == 0

    f += 2
    assert f == approx(2)
    assert f > 1
    assert f < 3
    assert f >= 1
    assert f >= 2
    assert f <= 2
    assert f <= 3
    assert f - 1 == approx(1)
    assert 1 - f == approx(-1)
    assert f + 1 == approx(3)
    assert 1 + f == approx(3)
    f -= 1
    assert f == approx(1)


def test_save():
    """Test that history gets saved."""
    f = rollstats.MemoryFloat(0.0)

    check_history(f, [])
    for i in range(1, 10):
        f.save()
        f += 1
        check_history(f, list(range(i)))


def test_copy():
    """Test that (deep) copying works."""
    # Create a float with history
    f1 = rollstats.MemoryFloat(0.0)
    f1.save()
    f1 += 1
    f1.save()

    # Copy it to another one
    f2 = f1.copy()

    # Check that it is equal but not the same in memory
    assert f1 is not f2
    assert f1 == f2
    assert f1.history == f2.history
    assert f1.history is not f2.history

    # As for the value, it IS actually the same in memory
    # initially, but this is OK since it is immutable.
    assert f1.value == f2.value
    assert f1.value is f2.value

    # The identity changes as soon as we do anything.
    f2 += 0
    assert f1.value == f2.value
    assert f1.value is not f2.value

    # And when we change the value, the value changes too.
    f2.value -= 1
    assert f1.value != f2.value


def test_transform():
    f1 = rollstats.MemoryFloat(0.0)
    f2 = rollstats.MemoryFloat(1.0)
    for i in range(1, 10):
        f1 += 1
        f1.save()
        f2.save()

    f_sum = rollstats.MemoryFloat.transform(f1, f2, func=lambda x, y: x + y)
    check_history(f_sum, list(range(2, 11)))


def test_repr():
    f1 = rollstats.MemoryFloat(0.0)
    assert repr(f1) == "MemoryFloat(v=0.00, history=[])"
    for i in range(10):
        f1 += 1
        f1.save()
    assert (
        repr(f1)
        == "MemoryFloat(v=10.00, history=['1.00', '2.00', '3.00', '4.00', '5.00', '6.00', '7.00', '8.00', '9.00', '10.00'])"
    )


def test_follow():
    i = 0
    values = [0, 1, 2, 3]

    def hook(f):
        nonlocal i
        assert f == values[i]
        i += 1

    f1 = rollstats.MemoryFloat()
    f1.add_hook(hook)
    for value in values:
        f1.assign(value)
        f1.save()


def test_connect():
    i1 = rollstats.MemoryFloat(0)
    i2 = rollstats.MemoryFloat(0)
    i3 = rollstats.MemoryFloat(0)
    o = rollstats.MemoryFloat()

    def sum_func(i1, i2, i3):
        return i1.value + i2.value + i3.value

    rollstats.MemoryFloat.connect(i1, i2, i3, output=o, func=sum_func)

    i1_values = [1, 2, 3]
    i2_values = [10, 20, 30]
    i3_values = [100, 200, 300]

    for i1_val, i2_val, i3_val in zip(i1_values, i2_values, i3_values):
        i1.assign(i1_val)
        i1.save()

        i2.assign(i2_val)
        i2.save()

        i3.assign(i3_val)
        i3.save()

    check_history(o, [111, 222, 333])
