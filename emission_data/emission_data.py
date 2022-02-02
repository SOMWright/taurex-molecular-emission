import numpy as np
import pathlib
from taurex.mpi import allocate_as_shared
from taurex.log import Logger
from taurex.util.math import intepr_bilin, interp_exp_and_lin, \
    interp_lin_only, interp_exp_only


class EmissionData(Logger):

    @classmethod
    def priority(cls):
        return 2

    @classmethod
    def discover(cls):
        import os
        import glob
        from taurex.cache import GlobalCache

        path = GlobalCache()['emission_data_path']
        if path is None:
            return []
        path = [os.path.join(path, '*.h5'), os.path.join(path, '*.hdf5')]
        file_list = [f for glist in path for f in glob.glob(glist)]

        discovery = []

        interp = GlobalCache()['xsec_interpolation'] or 'linear'
        mem = GlobalCache()['xsec_in_memory'] or True

        for f in file_list:
            op = EmissionData(f, interpolation_mode='linear', in_memory=False)
            mol_name = op.molecule_name
            discovery.append((mol_name, [f, interp, mem]))
            del op

        return discovery

    def __init__(self, filename, interpolation_mode='linear', in_memory=True):
        super().__init__('EmissionData:{}'.format(pathlib.Path(filename).stem[0:10]))

        self._filename = filename
        self._molecule_name = None
        self._spec_dict = None
        self.in_memory = in_memory
        self._load_hdf_file(filename)
        self._interp_mode = interpolation_mode

    @property
    def molecule_name(self):
        return self._molecule_name

    @property
    def emiss_grid(self):
        return self._emiss_grid

    def _load_hdf_file(self, filename):
        import h5py
        import astropy.units as u

        self.debug('Loading emission data from {}'.format(filename))

        self._spec_dict = h5py.File(filename, 'r')

        self._wavenumber_grid = self._spec_dict['bin_edges'][:]
        self._temperature_grid = self._spec_dict['t'][:]  # *t_conversion

        pressure_units = self._spec_dict['p'].attrs['units']
        try:
            p_conversion = u.Unit(pressure_units).to(u.Pa)
        except:
            p_conversion = u.Unit(pressure_units, format="cds").to(u.Pa)

        self._pressure_grid = self._spec_dict['p'][:] * p_conversion

        if self.in_memory:
            self._emiss_grid = allocate_as_shared(self._spec_dict['xsecarr'][...], logger=self)
        else:
            self._emiss_grid = self._spec_dict['xsecarr']

        self._resolution = np.average(np.diff(self._wavenumber_grid))
        self._molecule_name = self._spec_dict['mol_name'][()]

        if isinstance(self._molecule_name, np.ndarray):
            self._molecule_name = self._molecule_name[0]

        try:
            self._molecule_name = self._molecule_name.decode()
        except (UnicodeDecodeError, AttributeError,):
            pass

        from taurex.util.util import ensure_string_utf8

        self._molecule_name = ensure_string_utf8(self._molecule_name)

        self._min_pressure = self._pressure_grid.min()
        self._max_pressure = self._pressure_grid.max()
        self._min_temperature = self._temperature_grid.min()
        self._max_temperature = self._temperature_grid.max()

        if self.in_memory:
            self._spec_dict.close()

    @property
    def wavenumber_grid(self):
        return self._wavenumber_grid

    @property
    def temperature_grid(self):
        return self._temperature_grid

    @property
    def pressure_grid(self):
        return self._pressure_grid

    @property
    def resolution(self):
        return self._resolution

    def emission(self, temperature, pressure, wngrid=None):
        if wngrid is None:
            wngrid_filter = slice(None)
        else:
            wngrid_filter = np.where((self.wavenumber_grid >= wngrid.min()) & (
                    self.wavenumber_grid <= wngrid.max()))[0]

        orig = self.compute_molecular_emission(temperature, pressure, wngrid_filter)

        if wngrid is None or np.array_equal(self.wavenumber_grid.take(wngrid_filter), wngrid):
            return orig
        else:
            return np.interp(wngrid, self.wavenumber_grid[wngrid_filter], orig)

    def compute_molecular_emission(self, temperature, pressure, wngrid=None):
        import math
        logpressure = math.log10(pressure)
        return self.interp_bilinear_grid(temperature, logpressure, *self.find_closest_index(temperature, logpressure),
                                         wngrid)

    @property
    def pressure_max(self):
        return self.pressure_grid[-1]

    @property
    def pressure_min(self):
        return self.pressure_grid[0]

    @property
    def temperature_max(self):
        return self.temperature_grid[-1]

    @property
    def temperature_min(self):
        return self.temperature_grid[0]

    @property
    def log_pressure(self):
        return np.log10(self.pressure_grid)

    @property
    def pressure_bounds(self):
        return self.log_pressure.min(), self.log_pressure.max()

    @property
    def temperature_bounds(self):
        return self.temperature_grid.min(), self.temperature_grid.max()

    def find_closest_index(self, T, P):
        from taurex.util.util import find_closest_pair

        t_min, t_max = find_closest_pair(self.temperature_grid, T)
        p_min, p_max = find_closest_pair(self.log_pressure, P)

        return t_min, t_max, p_min, p_max

    def set_interpolation_mode(self, interp_mode):
        self._interp_mode = interp_mode.strip()

    def interp_temp_only(self, T, t_idx_min, t_idx_max, P, filt):
        Tmax = self.temperature_grid[t_idx_max]
        Tmin = self.temperature_grid[t_idx_min]
        fx0 = self.emiss_grid[P, t_idx_min, filt]
        fx1 = self.emiss_grid[P, t_idx_max, filt]

        if self._interp_mode == 'linear':
            return interp_lin_only(fx0, fx1, T, Tmin, Tmax)
        elif self._interp_mode == 'exp':
            return interp_exp_only(fx0, fx1, T, Tmin, Tmax)
        else:
            raise ValueError(
                'Unknown interpolation mode {}'.format(self._interp_mode))

    def interp_pressure_only(self, P, p_idx_min, p_idx_max, T, filt):
        Pmax = self.log_pressure[p_idx_max]
        Pmin = self.log_pressure[p_idx_min]
        fx0 = self.emiss_grid[p_idx_min, T, filt]
        fx1 = self.emiss_grid[p_idx_max, T, filt]

        return interp_lin_only(fx0, fx1, P, Pmin, Pmax)

    def interp_bilinear_grid(self, T, P, t_idx_min, t_idx_max, p_idx_min,
                             p_idx_max, wngrid_filter=None):

        self.debug('Interpolating %s %s %s %s %s %s', T, P,
                   t_idx_min, t_idx_max, p_idx_min, p_idx_max)

        min_pressure, max_pressure = self.pressure_bounds
        min_temperature, max_temperature = self.temperature_bounds

        check_pressure_max = P >= max_pressure
        check_temperature_max = T >= max_temperature

        check_pressure_min = P < min_pressure
        check_temperature_min = T < min_temperature

        self.debug('Check pressure min/max %s/%s',
                   check_pressure_min, check_pressure_max)
        self.debug('Check temperature min/max %s/%s',
                   check_temperature_min, check_temperature_max)
        # Are we both max?
        if check_pressure_max and check_temperature_max:
            self.debug('Maximum Temperature pressure reached. Using last')
            return self.emiss_grid[-1, -1, wngrid_filter].ravel()

        if check_pressure_min and check_temperature_min:
            return np.zeros_like(self.emiss_grid[0, 0, wngrid_filter]).ravel()

        # Max pressure
        if check_pressure_max:
            self.debug('Max pressure reached. Interpolating temperature only')
            return self.interp_temp_only(T, t_idx_min, t_idx_max, -1, wngrid_filter)

        # Max temperature
        if check_temperature_max:
            self.debug('Max temperature reached. Interpolating pressure only')
            return self.interp_pressure_only(P, p_idx_min, p_idx_max, -1, wngrid_filter)

        if check_pressure_min:
            self.debug('Min pressure reached. Interpolating temperature only')
            return self.interp_temp_only(T, t_idx_min, t_idx_max, 0, wngrid_filter).ravel()

        if check_temperature_min:
            self.debug('Min temeprature reached. Interpolating pressure only')
            return self.interp_pressure_only(P, p_idx_min, p_idx_max, 0, wngrid_filter).ravel()

        q_11 = self.emiss_grid[p_idx_min, t_idx_min][wngrid_filter].ravel()
        q_12 = self.emiss_grid[p_idx_min, t_idx_max][wngrid_filter].ravel()
        q_21 = self.emiss_grid[p_idx_max, t_idx_min][wngrid_filter].ravel()
        q_22 = self.emiss_grid[p_idx_max, t_idx_max][wngrid_filter].ravel()

        Tmax = self.temperature_grid[t_idx_max]
        Tmin = self.temperature_grid[t_idx_min]
        Pmax = self.log_pressure[p_idx_max]
        Pmin = self.log_pressure[p_idx_min]

        if self._interp_mode == 'linear':
            return intepr_bilin(q_11, q_12, q_21, q_22, T, Tmin, Tmax, P, Pmin, Pmax)
        elif self._interp_mode == 'exp':
            return interp_exp_and_lin(q_11, q_12, q_21, q_22, T, Tmin, Tmax, P, Pmin, Pmax)
        else:
            raise ValueError(
                'Unknown interpolation mode {}'.format(self._interp_mode))
