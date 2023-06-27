import pytest

import numpy as np
from matplotlib import pyplot as plt
from logistic import iterate_f
from math import isclose

def logistic_uno(x, r):
    return r * x * (1 -x )

L = [0.198, 0.544, 0.31875]

@pytest.mark.parametrize("x,r, out", [(0.1, 2.2, L[0]), (0.2, 3.4, L[1]), (0.75, 1.7, L[2])])
def test_logistic(x,r, out):
    output = logistic_uno(x,r)
    assert isclose(output, out)


L0=[0.198]
L1=[0.544, 0.843418, 0.449019, 0.841163]
L2=[0.31875, 0.369152]

@pytest.mark.parametrize("x,r,it, out",
[(1, 0.1, 2.2, L0), (4, 0.2, 3.4, L1), (2, 0.75, 1.7, L2)])
def test_logistic_it(x,r,it, out):
    output = iterate_f(it, x,r)
    assert isclose( output == out ).all()
