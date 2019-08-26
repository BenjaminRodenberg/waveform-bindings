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

    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)

    def test_import(self):
        pass

    def test_init(self):
        with patch("precice_future.Interface") as tests.MockedPrecice.Interface:
            from waveformbindings import WaveformBindings
            WaveformBindings("Dummy", 0, 1)

    def test_read(self):
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
        bindings.initialize_waveforms(dummy_mesh_id, self.n_vertices, dummy_vertex_ids, "Dummy-Write", "Dummy-Read", 1, 1)
        bindings._read_data_buffer.append(to_be_read, 0)
        bindings._read_data_buffer.append(to_be_read, 1)
        Interface.read_block_scalar_data = MagicMock(return_value=to_be_read)
        bindings.read_block_scalar_data("Dummy-Read", dummy_mesh_id, self.n_vertices, dummy_vertex_ids, read_data, 0)
        self.assertTrue(np.isclose(read_data, to_be_read).all())

    def test_write(self):
        from waveformbindings import WaveformBindings
        from precice_future import Interface

        Interface.get_data_id = MagicMock()
        Interface.write_block_scalar_data = MagicMock()
        bindings = WaveformBindings("Dummy", 0, 1)
        bindings.configure_waveform_relaxation(10, 1)
        bindings._precice_tau = self.dt
        dummy_mesh_id = MagicMock()
        dummy_vertex_ids = np.random.rand(10)
        old_data = np.random.rand(self.n_vertices)
        to_be_written = old_data + np.random.rand(self.n_vertices)
        write_data = to_be_written
        bindings.initialize_waveforms(dummy_mesh_id, self.n_vertices, dummy_vertex_ids, "Dummy-Write", "Dummy-Read", 1, 1)
        bindings._write_data_buffer.append(old_data, 0)
        bindings._write_data_buffer.append(old_data, 1)
        bindings._write_data_buffer.empty_data()
        bindings.write_block_scalar_data("Dummy-Write", dummy_mesh_id, self.n_vertices, dummy_vertex_ids, write_data, 0)
        bindings.write_block_scalar_data("Dummy-Write", dummy_mesh_id, self.n_vertices, dummy_vertex_ids, write_data, 1)
        self.assertTrue(np.isclose(to_be_written, bindings._write_data_buffer.sample(0)).all())

    def test_do_some_steps(self):
        from waveformbindings import WaveformBindings
        from precice_future import Interface, action_read_iteration_checkpoint, action_write_iteration_checkpoint

        Interface.advance = MagicMock()
        Interface.get_data_id = MagicMock()
        Interface.read_block_scalar_data = MagicMock(return_value=np.zeros(self.n_vertices))
        Interface.write_block_scalar_data = MagicMock()
        bindings = WaveformBindings("Dummy", 0, 1)
        bindings.read_slope = 0
        bindings.write_slope = 0
        bindings.configure_waveform_relaxation(1, 10)
        bindings._precice_tau = self.dt
        dummy_mesh_id = MagicMock()
        dummy_vertex_ids = np.random.rand(self.n_vertices)
        bindings.initialize_waveforms(dummy_mesh_id, self.n_vertices, dummy_vertex_ids, "Dummy-Write", "Dummy-Read", 1, 1)
        bindings._write_data_buffer.append(np.zeros(self.n_vertices), 0)
        bindings._read_data_buffer.append(np.zeros(self.n_vertices), 0)
        bindings._read_data_buffer.append(np.zeros(self.n_vertices), 1)
        bindings._precice_tau = self.dt
        Interface.is_action_required = MagicMock(return_value=False)
        self.assertEqual(bindings._current_window_start, 0.0)
        for i in range(9):
            self.assertTrue(np.isclose(bindings._window_time, i * .1))
            bindings.write_block_scalar_data("Dummy-Write", dummy_mesh_id, self.n_vertices, dummy_vertex_ids, (i + 1) * np.ones(self.n_vertices), bindings._window_time + .1)
            bindings.advance(.1)
            self.assertTrue(np.isclose(bindings._window_time, (i+1) * .1))
            self.assertTrue(np.isclose(bindings._current_window_start, 0.0))
        bindings.write_block_scalar_data("Dummy-Write", dummy_mesh_id, self.n_vertices, dummy_vertex_ids, (i + 1) * np.ones(self.n_vertices), bindings._window_time + .1)
        bindings.advance(.1)
        self.assertTrue(np.isclose(bindings._current_window_start, 1.0))

        def is_action_required_behavior(py_action):
            if py_action == action_read_iteration_checkpoint():
                return True
            elif py_action == action_write_iteration_checkpoint():
                return False
        Interface.is_action_required = MagicMock(side_effect=is_action_required_behavior)

        for i in range(9):
            self.assertTrue(np.isclose(bindings._window_time, i * .1))
            bindings.write_block_scalar_data("Dummy-Write", dummy_mesh_id, self.n_vertices, dummy_vertex_ids, np.random.rand(self.n_vertices), bindings._current_window_start + bindings._window_time + .1)
            bindings.advance(.1)
            self.assertTrue(np.isclose(bindings._window_time, (i+1) * .1))
            self.assertTrue(np.isclose(bindings._current_window_start, 1.0))
        bindings.write_block_scalar_data("Dummy-Write", dummy_mesh_id, self.n_vertices, dummy_vertex_ids, (i + 1) * np.ones(self.n_vertices), bindings._current_window_start + bindings._window_time + .1)
        bindings.advance(.1)
        self.assertTrue(np.isclose(bindings._current_window_start, 1.0))