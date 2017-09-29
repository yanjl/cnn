from cnn.lstm import add

import pytest


@pytest.fixture
def test_init():
    # assert add(2, 3) == 5
    print('init test mock')


def test_add():
    assert add(2, 10) == 12
