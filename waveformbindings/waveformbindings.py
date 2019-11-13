import numpy as np
import logging
import itertools

# create logger
module_logger = logging.getLogger('waveformbindings')
module_logger.setLevel(logging.WARNING)

try:
    import precice_future
    from precice_future import action_read_iteration_checkpoint, action_write_initial_data, action_write_iteration_checkpoint
except ImportError:
    import os
    import sys
    # check if PRECICE_ROOT is defined
    if not os.getenv('PRECICE_ROOT'):
       raise Exception("ERROR: PRECICE_ROOT not defined!")

    precice_root = os.getenv('PRECICE_ROOT')
    precice_python_adapter_root = precice_root+"/src/precice/bindings/python"
    sys.path.insert(0, precice_python_adapter_root)
    import precice_future
    from precice_future import action_read_iteration_checkpoint, action_write_initial_data, action_write_iteration_checkpoint


class WaveformBindings(precice_future.Interface):

    def configure_waveform_relaxation(self, n_this, n_other, interpolation_strategy='linear'):
        self._sample_counter_this = 0
        self._sample_counter_other = 0
        self._precice_tau = None
        self._n_this = n_this  # number of timesteps in this window, by default: no WR
        self._n_other = n_other  # number of timesteps in other window, todo: in the end we don't want to worry about the other solver's resolution!
        # multirate time stepping
        self._current_window_start = 0  # defines start of window
        self._window_time = self._current_window_start  # keeps track of window time
        self._interpolation_strategy = interpolation_strategy

    def initialize_waveforms(self, write_info, read_info):
        module_logger.debug("Calling initialize_waveforms")
        module_logger.debug("Initializing waveforms.")

        self._write_info = write_info
        module_logger.debug("Creating write_data_buffer: data_dimension = {}".format(self._write_info["data_dimension"]))
        self._write_data_buffer = Waveform(self._current_window_start, self._precice_tau, self._write_info["n_vertices"], self._write_info["data_dimension"], interpolation_strategy=self._interpolation_strategy)

        self._read_info = read_info
        module_logger.debug("Creating read_data_buffer: data_dimension = {}".format(self._read_info["data_dimension"]))
        self._read_data_buffer = Waveform(self._current_window_start, self._precice_tau, self._read_info["n_vertices"], self._read_info["data_dimension"], interpolation_strategy=self._interpolation_strategy)

    def perform_write_checks_and_append(self, write_data_name, mesh_id, vertex_ids, write_data, time):
        assert(self._is_inside_current_window(time))
        # we put the data into a buffer. Data will be send to other participant via preCICE in advance
        module_logger.debug("write data name is {write_data_name}".format(write_data_name=write_data_name))
        module_logger.debug("write data is {write_data}".format(write_data=write_data))
        self._write_data_buffer.append(write_data[:], time)
        # we assert that the preCICE specific write parameters did not change since configure_waveform_relaxation
        assert (self._write_info["mesh_id"] == mesh_id)
        assert ((self._write_info["vertex_ids"] == vertex_ids).all())
        assert (self._write_info["data_name"] == write_data_name)

    def write_block_scalar_data(self, write_data_name, mesh_id, vertex_ids, write_data, time):
        module_logger.debug("calling write_block_scalar_data for time {time}".format(time=time))
        assert (self._write_info["data_dimension"] == 1)
        self.perform_write_checks_and_append(write_data_name, mesh_id, vertex_ids, write_data, time)

    def write_block_vector_data(self, write_data_name, mesh_id, vertex_ids, write_data, time):
        module_logger.debug("calling write_block_vector_data for time {time}".format(time=time))
        assert (self._write_info["data_dimension"] == self.get_dimensions())
        self.perform_write_checks_and_append(write_data_name, mesh_id, vertex_ids, write_data, time)

    def perform_read_checks_and_sample(self, read_data_name, mesh_id, vertex_ids, time):
        assert (self._is_inside_current_window(time))
        # we get the data from the interpolant. New data will be obtained from the other participant via preCICE in advance
        module_logger.debug("read at time {time}".format(time=time))
        read_data = self._read_data_buffer.sample(time)[:].copy()
        module_logger.debug("read_data is {read_data}".format(read_data=read_data))
        # we assert that the preCICE specific write parameters did not change since configure_waveform_relaxation
        assert (self._read_info["mesh_id"] == mesh_id)
        assert ((self._read_info["vertex_ids"] == vertex_ids).all())
        assert (self._read_info["data_name"] == read_data_name)
        return read_data

    def read_block_scalar_data(self, read_data_name, mesh_id, vertex_ids, time):
        module_logger.debug("calling read_block_scalar_data for time {time}".format(time=time))
        assert(self._read_info["data_dimension"] == 1)
        return self.perform_read_checks_and_sample(read_data_name, mesh_id, vertex_ids, time)

    def read_block_vector_data(self, read_data_name, mesh_id, vertex_ids, time):
        module_logger.debug("calling read_block_vector_data for time {time}".format(time=time))
        assert (self._read_info["data_dimension"] == self.get_dimensions())
        return self.perform_read_checks_and_sample(read_data_name, mesh_id, vertex_ids, time)

    def _write_all_window_data_to_precice(self):
        module_logger.debug("Calling _write_all_window_data_to_precice")
        write_data_name_prefix = self._write_info["data_name"]
        write_waveform = self._write_data_buffer
        module_logger.debug("write_waveform._temporal_grid = {grid}".format(grid=write_waveform._temporal_grid))
        for substep in range(1, self._n_this + 1):
            module_logger.debug("writing substep {substep} of {n_this}".format(substep=substep, n_this=self._n_this))
            write_data_name = write_data_name_prefix + str(substep)
            write_data_id = self.get_data_id(write_data_name, self._write_info["mesh_id"])
            substep_time = write_waveform._temporal_grid[substep]

            write_data = write_waveform.sample(substep_time)
            if self._write_info["data_dimension"] == 1:
                super().write_block_scalar_data(write_data_id, self._write_info["vertex_ids"], write_data)
            elif self._write_info["data_dimension"] == self.get_dimensions():
                super().write_block_vector_data(write_data_id, self._write_info["vertex_ids"], write_data)
      
            module_logger.debug("write data called {name}:{write_data} @ time = {time}".format(name=write_data_name,
                                                                                         write_data=write_data,
                                                                                         time=substep_time))

    def _rollback_write_data_buffer(self):
        self._write_data_buffer.empty_data(keep_first_sample=True)

    def _read_all_window_data_from_precice(self):
        module_logger.debug("Calling _read_all_window_data_from_precice")
        read_data_name_prefix = self._read_info["data_name"]
        read_waveform = self._read_data_buffer
        read_waveform.empty_data(keep_first_sample=True)
        read_times = np.linspace(self._current_window_start, self._current_window_end(), self._n_other + 1)  # todo THIS IS HARDCODED! FOR ADAPTIVE GRIDS THIS IS NOT FITTING.

        for substep in range(1, self._n_other + 1):
            read_data_name = read_data_name_prefix + str(substep)
            read_data_id = self.get_data_id(read_data_name, self._read_info["mesh_id"])
            substep_time = read_times[substep]
            if self._read_info["data_dimension"] == 1:
                read_data = super().read_block_scalar_data(read_data_id, self._read_info["vertex_ids"])
            elif self._read_info["data_dimension"] == self.get_dimensions():
                read_data = super().read_block_vector_data(read_data_id, self._read_info["vertex_ids"])
            module_logger.debug("reading at time {time}".format(time=substep_time))
            module_logger.debug("read_data called {name}:{read_data} @ time = {time}".format(name=read_data_name,
                                                                                       read_data=read_data,
                                                                                       time=substep_time))
            read_waveform.append(read_data, substep_time)

    def is_timestep_complete(self):
        return self._timestep_is_complete

    def advance(self, dt):
        self._window_time += dt
        self._timestep_is_complete = False

        if self._window_is_completed():
            module_logger.debug("Window is complete.")

            module_logger.debug("print write waveform")
            module_logger.debug(self._write_info["data_name"])
            self._write_data_buffer.print_waveform()
            self._write_all_window_data_to_precice()
            read_data_last = self._read_data_buffer.sample(self._current_window_end()).copy()  # store last read data before advance, otherwise it might be lost if window is finished
            module_logger.debug("calling precice_future.advance")

            write_data_last = self._write_data_buffer.sample(self._current_window_end()).copy()  # store last write data before advance, otherwise it might be lost if window is finished
            max_dt = super().advance(self._window_time)  # = time given by preCICE

            if self.reading_checkpoint_is_required():  # repeat window
                # repeat window
                module_logger.info("Repeat window.")
                self._rollback_write_data_buffer()
                self._window_time = 0
                self._read_all_window_data_from_precice()

            if super().is_timestep_complete():  # window is finished
                assert (not self.reading_checkpoint_is_required())
                module_logger.info("Next window.")
                # go to next window
                read_data_init = read_data_last
                write_data_init = write_data_last
                module_logger.debug("write_data_init with {write_data} from t = {time}".format(write_data=write_data_init,
                                                                                         time=self._current_window_end()))
                module_logger.debug("read_data_init with {read_data} from t = {time}".format(read_data=read_data_init,
                                                                                       time=self._current_window_end()))
                self._current_window_start += self._window_size()
                self._window_time = 0
                # initialize window start of new window with data from window end of old window
                module_logger.debug("create new write data buffer")
                self._write_data_buffer = Waveform(self._current_window_start, self._precice_tau, self._write_info["n_vertices"],
                                                   self._write_info["data_dimension"], interpolation_strategy=self._interpolation_strategy)
                self._write_data_buffer.append(write_data_init, self._current_window_start)
                # use constant extrapolation as initial guess for read data
                module_logger.debug("create new read data buffer with initial guess")
                self._read_data_buffer = Waveform(self._current_window_start, self._precice_tau, self._read_info["n_vertices"],
                                                  self._read_info["data_dimension"], interpolation_strategy=self._interpolation_strategy)
                self._read_data_buffer.append(read_data_init,
                                              self._current_window_start)
                self._read_all_window_data_from_precice()  # this read buffer will be overwritten anyway. todo: initial guess is currently not treated properly!
                self._print_window_status()
                self._timestep_is_complete = True

            module_logger.debug("print read waveform")
            module_logger.debug(self._read_info["data_name"])
            self._read_data_buffer.print_waveform()

        else:
            module_logger.debug("remaining time: {remain}".format(remain=self._remaining_window_time()))
            max_dt = self._remaining_window_time()  # = window time remaining
            assert(max_dt > 0)

        return max_dt

    def _print_window_status(self):
        module_logger.debug("## window status:")
        module_logger.debug(self._current_window_start)
        module_logger.debug(self._window_size())
        module_logger.debug(self._window_time)
        module_logger.debug("##")

    def _window_is_completed(self):
        if np.isclose(self._window_size(), self._window_time):
            module_logger.debug("COMPLETE!")
            return True
        else:
            return False

    def _remaining_window_time(self):
        return self._window_size() - self._window_time

    def _current_window_end(self):
        return self._current_window_start + self._window_size()

    def _is_inside_current_window(self, time):
        tol = self._window_size() * 10**-5
        return 0-tol <= time - self._current_window_start <= self._window_size() + tol

    def _window_size(self):
        return self._precice_tau

    def initialize(self):
        self._precice_tau = super().initialize()
        return np.max([self._precice_tau, self._remaining_window_time()])

    def writing_checkpoint_is_required(self):
        return self._is_action_required(action_write_iteration_checkpoint())

    def reading_checkpoint_is_required(self):
        return self._is_action_required(action_read_iteration_checkpoint())

    def writing_initial_data_is_required(self):
        return self._is_action_required(action_write_initial_data())

    def is_action_required(self, action):
        raise Exception("Don't use is_action_required({action}). Use the corresponding function call "
                        "writing_checkpoint_is_required(), reading_checkpoint_is_required() or "
                        "writing_initial_data_is_required() instead.".format(action=action))

    def _is_action_required(self, action):
        if action == action_write_initial_data():
            return True  # if we use waveform relaxation, we require initial data for both participants to be able to fill the write buffers correctly
        elif action == action_write_iteration_checkpoint() or action == action_read_iteration_checkpoint():
            return super().is_action_required(action)
        else:
            raise Exception("unexpected action. %s", str(action))

    def fulfilled_action(self, action):
        if action == action_write_initial_data():
            return None  # do not forward to precice. We have to check for this condition again in initialize_data
        elif action == action_write_iteration_checkpoint() or action == action_read_iteration_checkpoint():
            return super().fulfilled_action(action)  # forward to precice
        else:
            raise Exception("unexpected action. %s", str(action))

    def initialize_data(self, time=0, read_zero=None, write_zero=None):
        """

        :param time:
        :param read_zero: read data that should be used at the very beginning
        :param write_zero: write data that should be used at the very beginning
        :return:
        """
        module_logger.debug("Calling initialize_data")
        if self.writing_initial_data_is_required():
            module_logger.info("writing in initialize_data()")
            for substep in range(1, self._n_this + 1):
                time = substep * self._precice_tau / self._n_this
                module_logger.debug("initialize with: {data} @ time = {time}".format(time=time, data=write_zero))
                self._write_data_buffer.append(write_zero, time)
            self._write_all_window_data_to_precice()
            self._rollback_write_data_buffer()
            super().fulfilled_action(action_write_initial_data())

        return_value = super().initialize_data()

        if self.is_read_data_available():
            module_logger.info("reading in initialize_data()")
            self._read_data_buffer.empty_data(keep_first_sample=False)
            module_logger.debug("initialize with: {data} @ time = {time}".format(time=self._current_window_start, data=read_zero))
            if isinstance(read_zero, np.ndarray):
                self._read_data_buffer.append(read_zero, self._current_window_start)
            else:
                self._read_data_buffer.append(self._read_data_buffer.get_empty_ndarray(), self._current_window_start)
            self._read_all_window_data_from_precice()
            if not isinstance(read_zero, np.ndarray):
                self._read_data_buffer.copy_second_to_first()

        return return_value


class OutOfLocalWindowError(Exception):
    """Raised when the time is not inside the window; i.e. t not inside [t_start, t_end]"""
    pass


class NotOnTemporalGridError(Exception):
    """Raised when the point in time is not on the temporal grid. """
    pass


class NoDataError(Exception):
    """Raised if not data exists in waveform"""
    pass


class Waveform:
    def __init__(self, window_start, window_size, n_datapoints, dimension, interpolation_strategy='linear'):
        """
        :param window_start: starting time of the window
        :param window_size: size of window
        :param n_samples: number of samples on window
        """
        assert (n_datapoints >= 1)
        assert (window_size > 0)

        self._window_size = window_size
        self._window_start = window_start
        self._n_datapoints = n_datapoints
        self._data_dimension = dimension
        self._samples_in_time = None
        self._temporal_grid = None
        self._interpolation_strategy = interpolation_strategy
        self.empty_data()

    def _window_end(self):
        return self._window_start + self._window_size

    def _append_sample(self, data, time):
        """
        appends a new piece of data for given time to the datastructures
        :param data: new dataset
        :param time: associated time
        :return:
        """
        data = np.expand_dims(data, axis=data.ndim)
        self._samples_in_time = np.append(self._samples_in_time, data, axis=data.ndim-1)
        self._temporal_grid.append(time)

    def initialize_constant(self, data):
        assert (not self._temporal_grid)  # list self._temporal_grid is empty
        assert (self._samples_in_time.size == 0)  # numpy.array self._samples_in_time is empty

        self._append_sample(data, self._window_start)
        self._append_sample(data, self._window_end())

    def sample(self, time):
        from scipy.interpolate import interp1d, splrep, splev
        module_logger.debug("sample Waveform at %f" % time)

        if not self._temporal_grid:
            module_logger.error("Waveform does not hold any data. self_temporal_grid = {}".format(self._temporal_grid))
            raise NoDataError

        atol = 1e-08  # todo: this is equal to atol used by default in np.isclose. Is there a nicer way to implement the check below?
        if not (np.min(self._temporal_grid) - atol <= time <= np.max(self._temporal_grid) + atol):
            msg = "\ntime: {time} on temporal grid {grid}\n".format(
                time=time,
                grid=self._temporal_grid)
            raise OutOfLocalWindowError(msg)

        if self._data_dimension > 1:
            return_value = np.zeros((self._n_datapoints, self._data_dimension))
        else:
            return_value = np.zeros(self._n_datapoints)

        for i, d in itertools.product(range(self._n_datapoints), range(self._data_dimension)):
            values_along_time = dict()
            for j in range(len(self._temporal_grid)):
                t = self._temporal_grid[j]
                if self._data_dimension > 1:
                    values_along_time[t] = self._samples_in_time[i, d, j]
                else:
                    values_along_time[t] = self._samples_in_time[i, j]
            if self._interpolation_strategy in ['linear', 'quadratic', 'cubic']:
                interpolant = interp1d(list(values_along_time.keys()), list(values_along_time.values()), kind=self._interpolation_strategy)
            elif self._interpolation_strategy in ['quartic']:
                tck = splrep(list(values_along_time.keys()), list(values_along_time.values()), k=4)
                interpolant = lambda t: splev(t, tck)
            try:
                if self._data_dimension > 1:
                    return_value[i, d] = interpolant(time)
                else:
                    return_value[i] = interpolant(time)
            except ValueError:
                time_min = np.min(self._temporal_grid)
                time_max = np.max(self._temporal_grid)

                if not time_min <= time <= time_max:  # time is not in valid range [time_min,time_max]
                    atol = 10**-8
                    if time_min-atol <= time <= time_min:  # time < time_min within within tolerance atol -> truncuate
                        time = time_min
                    elif time_max <= time <= time_max+atol:  # time > time_max within within tolerance atol -> truncuate
                        time = time_max
                    else:
                        raise Exception("Invalid time {time} computed!".format(time=time))

                if self._data_dimension > 1:
                    return_value[i, d] = interpolant(time)
                else:
                    return_value[i] = interpolant(time)



        module_logger.debug("result is {result}.".format(result=return_value))
        return return_value

    def append(self, data, time):
        try:
            assert (data.shape[0] == self._n_datapoints)
            if self._data_dimension > 1:
                assert (data.shape[1] == self._data_dimension)
        except AssertionError:
            raise Exception("Data shape does NOT fit: shape expected = {shape_expected}, shape retreived = {shape_retrieved}".format(shape_expected=(self._n_datapoints, self._data_dimension), shape_retrieved=data.shape))

        if time in self._temporal_grid or (self._temporal_grid and time <= self._temporal_grid[-1]):
            raise Exception("It is only allowed to append data associated with time that is larger than the already existing time. Trying to append invalid time = {time} to temporal grid = {temporal_grid}".format(time=time, temporal_grid=self._temporal_grid))
        module_logger.debug("Append data of shape {}".format(data.shape))
        module_logger.debug("n_datapoints = {}, data_dimensions = {}".format(self._n_datapoints, self._data_dimension))
        self._append_sample(data, time)

    def empty_data(self, keep_first_sample=False):
        if keep_first_sample:
            first_sample, first_time = self.get_init()
            assert(first_time == self._window_start)
        if self._data_dimension > 1:
            self._samples_in_time = np.empty(shape=(self._n_datapoints, self._data_dimension, 0))  # store samples in time in this data structure. Number of rows = number of gridpoints per sample; number of columns = number of sampls in time
        else:
            self._samples_in_time = np.empty(shape=(self._n_datapoints, 0))  # store samples in time in this data structure. Number of rows = number of gridpoints per sample; number of columns = number of sampls in time
        self._temporal_grid = list()  # store time associated to samples in this datastructure
        if keep_first_sample:
            self._append_sample(first_sample, first_time)

    def get_init(self):
        if self._data_dimension > 1:
            return self._samples_in_time[:, :, 0], self._temporal_grid[0]
        else:
            return self._samples_in_time[:, 0], self._temporal_grid[0]

    def get_empty_ndarray(self):
        if self._data_dimension > 1:
            return np.empty(self._n_datapoints, self._data_dimension)
        else:
            return np.empty(self._n_datapoints)

    def copy_second_to_first(self):
        if self._data_dimension > 1:
            self._samples_in_time[:, :, 0] = self._samples_in_time[:, :, 1]
        else:
            self._samples_in_time[:, 0] = self._samples_in_time[:, 1]

    def print_waveform(self):
        module_logger.debug("time: {time}".format(time=self._temporal_grid))
        module_logger.debug("data: {data}".format(data=self._samples_in_time))

