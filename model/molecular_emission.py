import numpy as np
from taurex.model.simplemodel import SimpleForwardModel
from taurex.constants import PI
from taurex.util.emission import black_body
from taurex.core import derivedparam


class MolecularEmissionModel(SimpleForwardModel):
    """
    A forward model for eclipses including specific molecular emission data
    """

    def __init__(self,
                 planet=None,
                 star=None,
                 pressure_profile=None,
                 temperature_profile=None,
                 chemistry=None,
                 nlayers=100,
                 atm_min_pressure=1e-4,
                 atm_max_pressure=1e6,
                 ngauss=4,
                 ):
        super().__init__(self.__class__.__name__,
                         planet,
                         star,
                         pressure_profile,
                         temperature_profile,
                         chemistry,
                         nlayers,
                         atm_min_pressure,
                         atm_max_pressure)

        self.set_num_gauss(ngauss)
        self._clamp = 10

    def set_num_gauss(self, value, coeffs=None):
        self._ngauss = int(value)

        mu, weight = np.polynomial.legendre.leggauss(self._ngauss)
        self._mu_quads = (mu + 1) / 2
        self._wi_quads = weight / 2
        self._coeffs = coeffs
        if coeffs is None:
            self._coeffs = np.ones(self._ngauss)

    def set_quadratures(self, mu, weight, coeffs=None):
        self._mu_quads = (mu + 1) / 2
        self._wi_quads = weight / 2
        self._coeffs = coeffs
        if coeffs is None:
            self._coeffs = np.ones(self._ngauss)

    def compute_final_flux(self, f_total):
        star_sed = self._star.spectralEmissionDensity

        self.debug('Star SED: %s', star_sed)
        # quit()
        star_radius = self._star.radius
        planet_radius = self._planet.fullRadius
        self.debug('star_radius %s', self._star.radius)
        self.debug('planet_radius %s', self._star.radius)
        last_flux = (f_total / star_sed) * (planet_radius / star_radius) ** 2

        self.debug('last_flux %s', last_flux)

        return last_flux

    def partial_model(self, wngrid=None, cutoff_grid=True):
        from taurex.util.util import clip_native_to_wngrid
        self.initialize_profiles()

        native_grid = self.nativeWavenumberGrid
        if wngrid is not None and cutoff_grid:
            native_grid = clip_native_to_wngrid(native_grid, wngrid)
        self._star.initialize(native_grid)

        for contrib in self.contribution_list:
            contrib.prepare(self, native_grid)

        return self.evaluate_emission(native_grid, False)

    def evaluate_emission(self, wngrid, return_contrib):

        dz = self.deltaz

        total_layers = self.nLayers

        density = self.densityProfile

        wngrid_size = wngrid.shape[0]

        temperature = self.temperatureProfile
        tau = np.zeros(shape=(self.nLayers, wngrid_size))
        surface_tau = np.zeros(shape=(1, wngrid_size))

        layer_tau = np.zeros(shape=(1, wngrid_size))

        dtau = np.zeros(shape=(1, wngrid_size))

        # Do surface first
        # for layer in range(total_layers):
        for contrib in self.contribution_list:
            contrib.contribute(self, 0, total_layers, 0, 0,
                               density, surface_tau, path_length=dz)
        self.debug('density = %s', density[0])
        self.debug('surface_tau = %s', surface_tau)

        BB = black_body(wngrid, temperature[0]) / PI

        _mu = 1.0 / self._mu_quads[:, None]
        _w = self._wi_quads[:, None]
        I = BB * (np.exp(-surface_tau * _mu))

        self.debug('I1_pre %s', I)
        # Loop upwards
        for layer in range(total_layers):
            layer_tau[...] = 0.0
            dtau[...] = 0.0
            for contrib in self.contribution_list:
                contrib.contribute(self, layer + 1, total_layers,
                                   0, 0, density, layer_tau, path_length=dz)
                contrib.contribute(self, layer, layer + 1, 0,
                                   0, density, dtau, path_length=dz)
            # for contrib in self.contribution_list:

            self.debug('Layer_tau[%s]=%s', layer, layer_tau)

            dtau += layer_tau

            dtau_calc = 0.0
            if dtau.min() < self._clamp:
                dtau_calc = np.exp(-dtau)
            layer_tau_calc = 0.0
            if layer_tau.min() < self._clamp:
                layer_tau_calc = np.exp(-layer_tau)

            _tau = layer_tau_calc - dtau_calc

            if isinstance(_tau, float):
                tau[layer] += _tau
            else:
                tau[layer] += _tau[0]

            self.debug('dtau[%s]=%s', layer, dtau)
            BB = black_body(wngrid, temperature[layer]) / PI
            self.debug('BB[%s]=%s,%s', layer, temperature[layer], BB)

            dtau_calc = 0.0
            if dtau.min() < self._clamp:
                dtau_calc = np.exp(-dtau * _mu)
            layer_tau_calc = 0.0
            if layer_tau.min() < self._clamp:
                layer_tau_calc = np.exp(-layer_tau * _mu)

            I += BB * (layer_tau_calc - dtau_calc)

        self.debug('I: %s', I)

        return I, _mu, _w, tau

    def path_integral(self, wngrid, return_contrib):

        I, _mu, _w, tau = self.evaluate_emission(wngrid, return_contrib)
        self.debug('I: %s', I)

        flux_total = 2.0 * np.pi * sum(I * (_w / _mu))
        self.debug('flux_total %s', flux_total)

        return self.compute_final_flux(flux_total).ravel(), tau

    def write(self, output):
        model = super().write(output)
        model.write_scalar('ngauss', self._ngauss)
        return model

    @classmethod
    def input_keywords(self):
        return ['molecular_emission', ]

    @derivedparam(param_name='log_F_bol', param_latex='log(F$_\{bol\}$)', compute=False)
    def logBolometricFlux(self):
        """
        log10 Flux integrated over all wavelengths (W m-2)

        """
        from scipy.integrate import simps
        import math
        I, _mu, _w, tau = self.partial_model()

        flux_total = 2.0 * np.pi * sum(I * (_w / _mu))

        flux_wl = flux_total[::-1]
        wlgrid = 10000 / self.nativeWavenumberGrid[::-1]

        return math.log10(simps(flux_wl, wlgrid))
