"""
Tests for stat_functions
"""

import numpy as np

from pytest        import mark
from pytest        import approx
from pytest        import raises
from flaky         import flaky
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from hypothesis            import given, settings
from hypothesis.strategies import integers
from hypothesis.strategies import floats
from invisible_cities.core.testing_utils       import exactly
from invisible_cities.core.testing_utils       import float_arrays
from invisible_cities.core.testing_utils       import FLOAT_ARRAY
from invisible_cities.core.testing_utils       import random_length_float_arrays

from . stat_functions       import gaussian_experiment
from . stat_functions       import mean_and_std
from . stat_functions       import relative_error_ratio

from . core_functions import  NN


def test_simple_relative_error_ratio():
    a = 10
    sigma_a = 1
    b = 20
    sigma_b = 1

    sigma_a_rel = sigma_a / a
    sigma_b_rel = sigma_b / b

    rer = relative_error_ratio(a, sigma_a, b, sigma_b)
    assert rer**2   == approx(sigma_a_rel**2 +  sigma_b_rel**2 , rel=1e-3, abs=1e-3)



@given(floats(min_value = 100, max_value = +200),
       floats(min_value = 100, max_value = +200),
       floats(min_value = + 0.1, max_value = + 1),
       floats(min_value = + 0.1,max_value = + 1) )
@settings(max_examples=50)

def test_relative_error_ratio(a, b, sigma_a, sigma_b):
    rer = relative_error_ratio(a, sigma_a, b, sigma_b)
    sigma_a_rel = sigma_a / a
    sigma_b_rel = sigma_b / b

    assert rer**2   == approx(sigma_a_rel**2 +  sigma_b_rel**2 , rel=1e-3, abs=1e-3)


def test_simple_mean_and_std():
    Nevt  = 1e6
    mean = 100
    sigma = 10
    e = gaussian_experiment(nevt=Nevt, mean=mean, std=sigma)
    mu, std = mean_and_std(e, (0,200))
    assert mu   == approx(100  , rel=1e-2)
    assert std  == approx(10, rel=1e-2)


@given(floats(min_value = 100,
              max_value = +1000),
       floats(min_value = + 1,
              max_value = + 20))
@settings(max_examples=50)

def test_mean_and_std_positive(mean, sigma):
    Nevt  = int(1e5)
    e = gaussian_experiment(nevt=Nevt, mean=mean, std=sigma)

    mu, std = mean_and_std(e, (mean- 5 * sigma,mean + 5 * sigma))
    assert mu   == approx(mean  , rel=1e-2)
    assert std  == approx(sigma,  rel=1e-2)


@given(floats(min_value = -100,
              max_value = +100),
       floats(min_value = + 0.1,
              max_value = + 10))
@settings(max_examples=10)

def test_mean_and_std_zero(mean, sigma):
    Nevt  = int(1e5)
    e = gaussian_experiment(nevt=Nevt, mean=mean, std=sigma)

    mu, std = mean_and_std(e, (mean- 5 * sigma,mean + 5 * sigma))
    assert mu   == approx(mean  , abs=1e-1)
    assert std  == approx(sigma,  abs=1e-1)

def test_mean_and_std_nan():
    x = [NN,NN,NN,NN]
    y = mean_and_std(x, (0,10))
    assert np.count_nonzero(np.isnan(y)) == 2 # returns (nan, nan)

    y = [1,2,3,4, NN]
    z = [1,2,3,4]
    np.allclose(mean_and_std(y, (0,10)), mean_and_std(z, (0,10)), rtol=1e-05, atol=1e-05)
    
