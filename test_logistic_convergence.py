import random
import numpy as np
from logistic import iterate_f
import pytest
from math import isclose

SEED= np.random.randint (0,2**31)
@pytest.fixture
def random_state():
    print (f"Using seed {SEED}")
    rs=np.random.RandomState(SEED)
    return rs

def test_convergence(random_state):
    its = 42
    xrand = random_state.rand()
    result = iterate_f(its, xrand, 1.5)
    expected = 1/3
    assert isclose(result[-1], expected)
    #print(result)
