# From built-in
import unittest
# From third-party
import numpy as np
# From worm-rod-engine
from worm_rod_engine.parameter.dimensionless_parameter import default_dimensionless_parameter
from worm_rod_engine.parameter.util import convert_to_dimensionless


class TestParameter(unittest.TestCase):

    def test_default_dimensionless_paramter(self):
        # Expected values taken for dissertation
        self.assertAlmostEqual(default_dimensionless_parameter.e, 0.028, places = 2)
        self.assertAlmostEqual(default_dimensionless_parameter.rho, 1.0/3.0)
        self.assertAlmostEqual(default_dimensionless_parameter.K_c, 1.51, places = 2)
        self.assertAlmostEqual(default_dimensionless_parameter.K_y, 4.0)
        self.assertAlmostEqual(default_dimensionless_parameter.K_n, 0.00123, places=4)

if __name__ == '__main__':
    unittest.main()







