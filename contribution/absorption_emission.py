import numpy as np

from taurex.cache import OpacityCache
from taurex.contributions import Contribution


class AbsorptionEmissionContribution(Contribution):

    def contribute(self, model, start_horz_layer, end_horz_layer,
                   density_offset, layer, density, tau, path_length=None):

        super().contribute(model, start_horz_layer, end_horz_layer,
                           density_offset, layer, density, tau, path_length)

    def __init__(self):
        super().__init__('Absorption_Emission')
        self._opacity_cache = OpacityCache()

    def build(self, model):
        pass

    def prepare_each(self, model, wngrid):
        self.debug('Preparing model with %s', wngrid.shape)
        self._ngrid = wngrid.shape[0]
        self._opacity_cache = OpacityCache()
        sigma_xsec = None

        for gas in model.chemistry.activeGases:

            # self._total_contrib[...] =0.0
            gas_mix = model.chemistry.get_gas_mix_profile(gas)
            self.info('Recomputing active gas %s opacity', gas)

            xsec = self._opacity_cache[gas]

            if sigma_xsec is None:

                sigma_xsec = np.zeros(shape=(self._nlayers, self._ngrid))
            else:
                sigma_xsec[...] = 0.0

            for idx_layer, tp in enumerate(zip(model.temperatureProfile, model.pressureProfile)):
                self.debug('Got index,tp %s %s', idx_layer, tp)

                temperature, pressure = tp
                # print(gas,self._opacity_cache[gas].opacity(temperature,pressure,wngrid),gas_mix[idx_layer])
                sigma_xsec[idx_layer] += xsec.opacity(temperature, pressure, wngrid) * gas_mix[idx_layer]

            self.sigma_xsec = sigma_xsec

            self.debug('SIGMAXSEC %s', self.sigma_xsec)

            yield gas, sigma_xsec

    def prepare(self, model, wngrid):
        """

        Used to prepare the contribution for the calculation.
        Called before the forward model performs the main optical depth
        calculation. Default behaviour is to loop through :func:`prepare_each`
        and sum all results into a single cross-section.

        Parameters
        ----------
        model: :class:`~taurex.model.model.ForwardModel`
            Forward model

        wngrid: :obj:`array`
            Wavenumber grid
        """

        self._ngrid = wngrid.shape[0]
        self._nlayers = model.nLayers

        sigma_xsec = None
        self.debug('ABSORPTION VERSION')
        for gas, sigma in self.prepare_each(model, wngrid):
            self.debug('Gas %s', gas)
            self.debug('Sigma %s', sigma)
            if sigma_xsec is None:
                sigma_xsec = np.zeros_like(sigma)
            sigma_xsec += sigma

        self.sigma_xsec = sigma_xsec
        self.debug('Final sigma is %s', self.sigma_xsec)
        self.info('Done')

    def finalize(self, model):
        raise NotImplementedError

    @property
    def sigma(self):
        return self.sigma_xsec

    @classmethod
    def input_keywords(self):
        return ['AbsorptionEmission', 'MolecularAbsorptionAndEmission', 'AbsorptionAndEmission']
