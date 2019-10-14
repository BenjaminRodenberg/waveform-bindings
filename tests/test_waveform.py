# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html
# first install package python setup.py install
# then run tests with python setup.py test -s tests.test_waveform

from unittest.mock import MagicMock, patch
from unittest import TestCase
import warnings
import numpy as np
import numpy.testing as npt
import tests.MockedPrecice

fake_dolfin = MagicMock()


@patch.dict('sys.modules', **{'precice_future': tests.MockedPrecice})
class TestWaveform(TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)
        warnings.simplefilter('ignore', category=DeprecationWarning)
        self.window_start = 2
        self.window_size = 3
        self.n_samples = 10
        self.global_time_grid = self.window_start + self.window_size * np.linspace(0, 1, self.n_samples)
        self.n_datapoints = 3
        self.dimensions = 2
        self.scalar_input_data = np.array(range(self.n_datapoints))
        self.vector_input_data = np.array([[1, 1], [2, 2], [3, 3]])

    def test_initialize_scalar_data(self):
        from waveformbindings.waveformbindings import Waveform
        self.waveform = Waveform(self.window_start, self.window_size, self.n_datapoints, 1)
        self.waveform.initialize_constant(self.scalar_input_data)

    def test_initialize_vector_data(self):
        from waveformbindings.waveformbindings import Waveform
        self.waveform = Waveform(self.window_start, self.window_size, self.n_datapoints, self.dimensions)
        self.waveform.initialize_constant(self.vector_input_data)

    def test_sample_scalar_data(self):
        from waveformbindings.waveformbindings import Waveform
        self.waveform = Waveform(self.window_start, self.window_size, self.n_datapoints, 1)
        from waveformbindings.waveformbindings import OutOfLocalWindowError, NoDataError

        with self.assertRaises(NoDataError):
            out = self.waveform.sample(0)

        self.waveform.initialize_constant(self.scalar_input_data)

        for t in np.linspace(self.global_time_grid[0], self.global_time_grid[-1]):
            out = self.waveform.sample(t)
            npt.assert_almost_equal(out, self.scalar_input_data)

        with self.assertRaises(OutOfLocalWindowError):
            self.waveform.sample(self.global_time_grid[-1] + .1)
        with self.assertRaises(OutOfLocalWindowError):
            self.waveform.sample(self.global_time_grid[0] - .1)

    def test_sample_vector_data(self):
        from waveformbindings.waveformbindings import Waveform
        self.waveform = Waveform(self.window_start, self.window_size, self.n_datapoints, self.dimensions)
        from waveformbindings.waveformbindings import OutOfLocalWindowError, NoDataError

        with self.assertRaises(NoDataError):
            out = self.waveform.sample(0)

        self.waveform.initialize_constant(self.vector_input_data)

        for t in np.linspace(self.global_time_grid[0], self.global_time_grid[-1]):
            out = self.waveform.sample(t)
            npt.assert_almost_equal(out, self.vector_input_data)

        with self.assertRaises(OutOfLocalWindowError):
            self.waveform.sample(self.global_time_grid[-1] + .1)
        with self.assertRaises(OutOfLocalWindowError):
            self.waveform.sample(self.global_time_grid[0] - .1)
