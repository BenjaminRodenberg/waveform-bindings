# comments on test layout: https://docs.pytest.org/en/latest/goodpractices.html
# first install package python setup.py install
# then run tests with python setup.py test -s tests.test_waveform_bindings

from unittest.mock import MagicMock, patch
from unittest import TestCase
import warnings
import numpy as np
import tests.MockedPrecice

fake_dolfin = MagicMock()


@patch.dict('sys.modules', **{'precice_future': tests.MockedPrecice})
class TestWaveformBindings(TestCase):

    dt = 1
    t = 0
    n = 0
    dummy_config = "tests/precice-adapter-config-WR10.json"
    n_vertices = 5
    dimensions = 2
    dummy_data_id = 6576

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

    def test_import(self):
        pass

    def test_init(self):
        with patch("precice_future.Interface") as tests.MockedPrecice.Interface:
            from waveformbindings import WaveformBindings
            WaveformBindings("Dummy", 0, 1)

    def test_read_scalar(self):
        from waveformbindings import WaveformBindings
        from precice_future import Interface

        Interface.get_data_id = MagicMock()
        dummy_mesh_id = MagicMock()
        dummy_vertex_ids = np.random.rand(10)
        bindings = WaveformBindings("Dummy", 0, 1)
        bindings.configure_waveform_relaxation(1, 10)
        bindings._precice_tau = self.dt
        old_data = np.random.rand(self.n_vertices)
        write_data = np.random.rand(self.n_vertices)
        read_data = old_data
        to_be_read = old_data + 1
        write_info = {"mesh_id": dummy_mesh_id, "n_vertices": self.n_vertices, "vertex_ids": dummy_vertex_ids, "data_name": "Dummy-Write", "data_dimension": 1}
        read_info = {"mesh_id": dummy_mesh_id, "n_vertices": self.n_vertices, "vertex_ids": dummy_vertex_ids, "data_name": "Dummy-Read", "data_dimension": 1}
        bindings.initialize_waveforms(write_info, read_info)
        bindings._read_data_buffer.append(to_be_read, 0)
        bindings._read_data_buffer.append(to_be_read, 1)
        Interface.read_block_scalar_data = MagicMock(return_value=to_be_read)
        Interface.get_dimensions = MagicMock(return_value=self.dimensions)
        read_data = bindings.read_block_scalar_data("Dummy-Read", dummy_mesh_id, dummy_vertex_ids, 0)
        self.assertTrue(np.isclose(read_data, to_be_read).all())

    def test_read_vector(self):
        from waveformbindings import WaveformBindings
        from precice_future import Interface

        Interface.get_data_id = MagicMock()
        dummy_mesh_id = MagicMock()
        dummy_vertex_ids = np.random.rand(10)
        bindings = WaveformBindings("Dummy", 0, 1)
        bindings.configure_waveform_relaxation(1, 10)
        bindings._precice_tau = self.dt
        old_data = np.random.rand(self.n_vertices, self.dimensions)
        to_be_read = old_data + 1
        write_info = {"mesh_id": dummy_mesh_id, "n_vertices": self.n_vertices, "vertex_ids": dummy_vertex_ids, "data_name": "Dummy-Write", "data_dimension": self.dimensions}
        read_info = {"mesh_id": dummy_mesh_id, "n_vertices": self.n_vertices, "vertex_ids": dummy_vertex_ids, "data_name": "Dummy-Read", "data_dimension": self.dimensions}
        bindings.initialize_waveforms(write_info, read_info)
        bindings._read_data_buffer.append(to_be_read, 0)
        bindings._read_data_buffer.append(to_be_read, 1)
        Interface.read_block_vector_data = MagicMock(return_value=to_be_read)
        Interface.get_dimensions = MagicMock(return_value=self.dimensions)
        read_data = bindings.read_block_vector_data("Dummy-Read", dummy_mesh_id, dummy_vertex_ids, 0)
        self.assertTrue(np.isclose(read_data, to_be_read).all())

    def test_write_scalar(self):
        from waveformbindings import WaveformBindings
        from precice_future import Interface

        Interface.get_data_id = MagicMock()
        Interface.write_block_scalar_data = MagicMock()
        Interface.get_dimensions = MagicMock(return_value=self.dimensions)
        bindings = WaveformBindings("Dummy", 0, 1)
        bindings.configure_waveform_relaxation(10, 1)
        bindings._precice_tau = self.dt
        dummy_mesh_id = MagicMock()
        dummy_vertex_ids = np.random.rand(10)
        old_data = np.random.rand(self.n_vertices)
        to_be_written = old_data + np.random.rand(self.n_vertices)
        write_data = to_be_written
        write_info = {"mesh_id": dummy_mesh_id, "n_vertices": self.n_vertices, "vertex_ids": dummy_vertex_ids, "data_name": "Dummy-Write", "data_dimension": 1}
        read_info = {"mesh_id": dummy_mesh_id, "n_vertices": self.n_vertices, "vertex_ids": dummy_vertex_ids, "data_name": "Dummy-Read", "data_dimension": 1}
        bindings.initialize_waveforms(write_info, read_info)
        bindings._write_data_buffer.append(old_data, 0)
        bindings._write_data_buffer.append(old_data, 1)
        bindings._write_data_buffer.empty_data()
        bindings.write_block_scalar_data("Dummy-Write", dummy_mesh_id, dummy_vertex_ids, write_data, 0)
        bindings.write_block_scalar_data("Dummy-Write", dummy_mesh_id, dummy_vertex_ids, write_data, 1)
        self.assertTrue(np.isclose(to_be_written, bindings._write_data_buffer.sample(0)).all())

    def test_write_vector(self):
        from waveformbindings import WaveformBindings
        from precice_future import Interface

        Interface.get_data_id = MagicMock()
        Interface.write_block_vector_data = MagicMock()
        Interface.get_dimensions = MagicMock(return_value=self.dimensions)
        bindings = WaveformBindings("Dummy", 0, 1)
        bindings.configure_waveform_relaxation(10, 1)
        bindings._precice_tau = self.dt
        dummy_mesh_id = MagicMock()
        dummy_vertex_ids = np.random.rand(10)
        old_data = np.random.rand(self.n_vertices, self.dimensions)
        to_be_written = old_data + np.random.rand(self.n_vertices, self.dimensions)
        write_data = to_be_written
        write_info = {"mesh_id": dummy_mesh_id, "n_vertices": self.n_vertices, "vertex_ids": dummy_vertex_ids, "data_name": "Dummy-Write", "data_dimension": self.dimensions}
        read_info = {"mesh_id": dummy_mesh_id, "n_vertices": self.n_vertices, "vertex_ids": dummy_vertex_ids, "data_name": "Dummy-Read", "data_dimension": self.dimensions}
        bindings.initialize_waveforms(write_info, read_info)
        bindings._write_data_buffer.append(old_data, 0)
        bindings._write_data_buffer.append(old_data, 1)
        bindings._write_data_buffer.empty_data()
        bindings.write_block_vector_data("Dummy-Write", dummy_mesh_id, dummy_vertex_ids, write_data, 0)
        bindings.write_block_vector_data("Dummy-Write", dummy_mesh_id, dummy_vertex_ids, write_data, 1)
        self.assertTrue(np.isclose(to_be_written, bindings._write_data_buffer.sample(0)).all())

    def test_write_vector_data_to_precice(self):
        from waveformbindings import WaveformBindings
        from precice_future import Interface
        import numpy.testing as npt
        Interface.advance = MagicMock()
        Interface.get_data_id = MagicMock(return_value=self.dummy_data_id)
        Interface.get_dimensions = MagicMock(return_value=self.dimensions)
        Interface.write_block_vector_data = MagicMock()
        bindings = WaveformBindings("Dummy", 0, 1)
        bindings.read_slope = 0
        bindings.write_slope = 0
        bindings.configure_waveform_relaxation(1, 1)
        bindings._precice_tau = self.dt
        dummy_mesh_id = MagicMock()
        dummy_vertex_ids = np.random.rand(self.n_vertices)
        dummy_write_data = np.random.rand(self.n_vertices, self.dimensions)
        write_info = {"mesh_id": dummy_mesh_id, "n_vertices": self.n_vertices, "vertex_ids": dummy_vertex_ids,
                      "data_name": "Dummy-Write", "data_dimension": self.dimensions}
        read_info = {"mesh_id": dummy_mesh_id, "n_vertices": self.n_vertices, "vertex_ids": dummy_vertex_ids,
                     "data_name": "Dummy-Read", "data_dimension": self.dimensions}
        bindings.initialize_waveforms(write_info, read_info)
        bindings._write_data_buffer.append(np.zeros((self.n_vertices, self.dimensions)), 0)
        bindings._read_data_buffer.append(np.zeros((self.n_vertices, self.dimensions)), 0)
        bindings._read_data_buffer.append(np.zeros((self.n_vertices, self.dimensions)), 1)
        bindings._precice_tau = self.dt
        Interface.is_action_required = MagicMock(return_value=False)
        self.assertEqual(bindings._current_window_start, 0.0)
        bindings.write_block_vector_data("Dummy-Write", dummy_mesh_id, dummy_vertex_ids, dummy_write_data,
                                         bindings._window_time + self.dt)
        self.assertEqual(bindings._window_time, 0)
        self.assertEqual(bindings._current_window_start, 0)
        self.assertEqual(bindings._window_size(), self.dt)
        bindings.advance(self.dt)
        # replaces the following call, since we cannot compare np.arrays
        # Interface.write_block_vector_data.assert_called_with(self.dummy_data_id, dummy_vertex_ids, dummy_write_data)
        self.assertEqual(Interface.write_block_vector_data.call_args[0][0], self.dummy_data_id)
        npt.assert_array_equal(Interface.write_block_vector_data.call_args[0][1], dummy_vertex_ids)
        npt.assert_array_equal(Interface.write_block_vector_data.call_args[0][2], dummy_write_data)
        self.assertEqual(bindings._window_time, 0)
        self.assertEqual(bindings._current_window_start, self.dt)

    def test_do_some_steps(self):
        """
        Performs two complete coupling iterations. The coupling window has a size of 1, while the timestep size is 0.1.
        Therefore we have to perform 10 steps in order to complete the window. The test is designed in such a way that
        in the first coupling iteration a checkpoint has to be read and the window has to be repeated. In the second
        iteration, the coupling is successful and no checkpoint is read. Therefore, in the end of the test the window is
        complete.
        """
        from waveformbindings import WaveformBindings
        from precice_future import Interface, action_read_iteration_checkpoint, action_write_iteration_checkpoint

        Interface.advance = MagicMock()
        Interface.get_data_id = MagicMock()
        Interface.read_block_scalar_data = MagicMock(return_value=np.zeros(self.n_vertices))
        Interface.write_block_scalar_data = MagicMock()
        Interface.is_timestep_complete = MagicMock(return_value=False)
        bindings = WaveformBindings("Dummy", 0, 1)
        bindings.read_slope = 0
        bindings.write_slope = 0
        bindings.configure_waveform_relaxation(1, 10)
        bindings._precice_tau = self.dt
        dummy_mesh_id = MagicMock()
        dummy_vertex_ids = np.random.rand(self.n_vertices)
        write_info = {"mesh_id": dummy_mesh_id, "n_vertices": self.n_vertices, "vertex_ids": dummy_vertex_ids, "data_name": "Dummy-Write", "data_dimension": 1}
        read_info = {"mesh_id": dummy_mesh_id, "n_vertices": self.n_vertices, "vertex_ids": dummy_vertex_ids, "data_name": "Dummy-Read", "data_dimension": 1}
        bindings.initialize_waveforms(write_info, read_info)
        bindings._write_data_buffer.append(np.zeros(self.n_vertices), 0)
        bindings._read_data_buffer.append(np.zeros(self.n_vertices), 0)
        bindings._read_data_buffer.append(np.zeros(self.n_vertices), 1)
        bindings._precice_tau = self.dt
        Interface.is_action_required = MagicMock(return_value=False)
        self.assertEqual(bindings._current_window_start, 0.0)

        ## do 10 timesteps in first coupling iteration
        for i in range(9):
            self.assertTrue(np.isclose(bindings._window_time, i * .1))
            bindings.write_block_scalar_data("Dummy-Write", dummy_mesh_id, dummy_vertex_ids, (i + 1) * np.ones(self.n_vertices), bindings._window_time + .1)
            bindings.advance(.1)
            self.assertTrue(np.isclose(bindings._window_time, (i+1) * .1))
            self.assertTrue(np.isclose(bindings._current_window_start, 0.0))
        bindings.write_block_scalar_data("Dummy-Write", dummy_mesh_id, dummy_vertex_ids, (i + 1) * np.ones(self.n_vertices), bindings._window_time + .1)

        # mock steering methods, such that window is repeated: read checkpoint and return False for is_timestep_complete
        Interface.is_timestep_complete = MagicMock(return_value=False)
        def is_action_required_behavior(py_action):
            if py_action == action_read_iteration_checkpoint():
                return True
            elif py_action == action_write_iteration_checkpoint():
                return False
        Interface.is_action_required = MagicMock(side_effect=is_action_required_behavior)
        Interface.is_timestep_complete = MagicMock(return_value=False)

        bindings.advance(.1)  # window is repeated

        self.assertTrue(np.isclose(bindings._current_window_start, 0.0))

        # do 10 timesteps in second coupling iteration
        for i in range(9):
            self.assertTrue(np.isclose(bindings._window_time, i * .1))
            bindings.write_block_scalar_data("Dummy-Write", dummy_mesh_id, dummy_vertex_ids, np.random.rand(self.n_vertices), bindings._current_window_start + bindings._window_time + .1)
            bindings.advance(.1)
            self.assertTrue(np.isclose(bindings._window_time, (i+1) * .1))
            self.assertTrue(np.isclose(bindings._current_window_start, 0.0))
        bindings.write_block_scalar_data("Dummy-Write", dummy_mesh_id, dummy_vertex_ids, (i + 1) * np.ones(self.n_vertices), bindings._current_window_start + bindings._window_time + .1)

        # mock steering methods, such that window is complete: don't require reading checkpoint and return True for is_timestep_complete
        def is_action_required_behavior(py_action):
            if py_action == action_read_iteration_checkpoint():
                return False
            elif py_action == action_write_iteration_checkpoint():
                return True
        Interface.is_action_required = MagicMock(side_effect=is_action_required_behavior)
        Interface.is_timestep_complete = MagicMock(return_value=True)

        bindings.advance(.1)  # window is completed

        self.assertTrue(np.isclose(bindings._current_window_start, 1.0))
