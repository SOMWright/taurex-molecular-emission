"""
Contains caching class for Molecular Emission Data
"""

from taurex.core import Singleton
from taurex.log import Logger
from taurex.cache.globalcache import GlobalCache

from taurex.core import Singleton


class EmissionCache(Singleton):
    def init(self):
        self.emission_dict = {}
        self._emission_data_path = None
        self.log = Logger('EmissionCache')
        self._force_active = []

    def set_emission_path(self, emission_data_path):
        import os

        GlobalCache()['emission_data_path'] = emission_data_path

        if not os.path.isdir(emission_data_path):
            self.log.error('PATH: %s does not exist!!!', emission_data_path)
            raise NotADirectoryError
        self.log.debug('Path set to %s', emission_data_path)

    def set_memory_mode(self, in_memory):
        GlobalCache()['emiss_in_memory'] = in_memory
        self.clear_cache()

    def set_interpolation(self, interpolation_mode):
        GlobalCache()['emiss_interpolation'] = interpolation_mode
        self.clear_cache()

    def __getitem__(self, key):
        if key in self.emission_dict:
            return self.emission_dict[key]
        else:
            self.load_emission_data(molecule_filter=[key])
            if key in self.opacity_dict:
                return self.opacity_dict[key]
            else:
                # Otherwise throw an error
                self.log.error('Emission data for molecule %s could not be loaded', key)
                self.log.error('It could not be found in the local dictionary %s', list(self.opacity_dict.keys()))
                self.log.error('Or paths %s', GlobalCache()['xsec_path'])
                self.log.error('Try loading it manually/ putting it in a path')
                raise Exception('Opacity could not be loaded')

    def add_emission_data(self, opacity, molecule_filter=None):
        """

        Adds a :class:`~taurex.opacity.opacity.Opacity` object to the cache to then be
        used by Taurex 3

        Parameters
        ----------
        opacity : :class:`~taurex.opacity.opacity.Opacity`
            Opacity object to add to the cache

        molecule_filter : :obj:`list` of str , optional
            If provided, the opacity object will only be included
            if its molecule is in the list. Mostly used by the
            :func:`__getitem__` for filtering

        """
        self.log.info('Reading Emission Data %s', opacity.moleculeName)
        if opacity.moleculeName in self.opacity_dict:
            self.log.warning('Emission data with name %s already in emission dictionary %s skipping',
                             opacity.moleculeName,
                             self.opacity_dict.keys())
            return
        if molecule_filter is not None:
            if opacity.moleculeName in molecule_filter:
                self.log.info('Loading emission data %s into model', opacity.moleculeName)
                self.opacity_dict[opacity.moleculeName] = opacity
        else:
            self.log.info('Loading emission data %s into model', opacity.moleculeName)
            self.opacity_dict[opacity.moleculeName] = opacity

    def find_list_of_molecules(self):
        from glob import glob
        import os
        from taurex.parameter.classfactory import ClassFactory
        opacity_klasses = ClassFactory().opacityKlasses

        molecules = []

        for c in opacity_klasses:
            molecules.extend([x[0] for x in c.discover()])

        forced = self._force_active or []
        return set(molecules + forced + list(self.opacity_dict.keys()))

    def load_emission_data_from_path(self, path, molecule_filter=None):

        from taurex.parameter.classfactory import ClassFactory

        cf = ClassFactory()

        opacity_klass_list = sorted(cf.opacityKlasses,
                                    key=lambda x: x.priority())

        for c in opacity_klass_list:

            for mol, args in c.discover():
                self.log.debug('Klass: %s %s', mol, args)
                op = None
                if mol in molecule_filter and mol not in self.opacity_dict:
                    if not isinstance(args, (list, tuple,)):
                        args = [args]
                    op = c(*args)

                if op is not None and op.moleculeName not in self.opacity_dict:
                    self.add_emission_data(op, molecule_filter=molecule_filter)
                op = None  # Ensure garbage collection when run once

    def load_emission_data(self, opacities=None, opacity_path=None, molecule_filter=None):
        """
        Main function to use when loading molecular opacities. Handles both
        cross sections and paths. Handles lists of either so lists of
        :class:`~taurex.opacity.opacity.Opacity` objects or lists of paths can be used
        to load multiple files/objects


        Parameters
        ----------
        opacities : :class:`~taurex.opacity.opacity.Opacity` or :obj:`list` of :class:`~taurex.opacity.opacity.Opacity` , optional
            Object(s) to include in cache

        opacity_path : str or :obj:`list` of str, optional
            search path(s) to look for molecular opacities

        molecule_filter : :obj:`list` of str , optional
            If provided, the opacity will only be loaded
            if its molecule is in this list. Mostly used by the
            :func:`__getitem__` for filtering

        """
        from taurex.opacity import Opacity

        if opacity_path is None:
            opacity_path = GlobalCache()['xsec_path']

        if opacities is not None:
            if isinstance(opacities, (list,)):
                self.log.debug('Opacity passed is list')
                for opacity in opacities:
                    self.add_emission_data(opacity, molecule_filter=molecule_filter)
            elif isinstance(opacities, Opacity):
                self.add_emission_data(opacities, molecule_filter=molecule_filter)
            else:
                self.log.error('Unknown type %s passed into opacities, should be a list, single \
                     opacity or None if reading a path', type(opacities))
                raise Exception('Unknown type passed into opacities')
        else:
            self.load_emission_data_from_path(opacity_path, molecule_filter=molecule_filter)
            # if isinstance(opacity_path, str):
            #     self.load_opacity_from_path(opacity_path, molecule_filter=molecule_filter)
            # elif isinstance(opacity_path, (list,)):
            #     for path in opacity_path:
            #         self.load_opacity_from_path(path, molecule_filter=molecule_filter)

    def clear_cache(self):
        """
        Clears all currently loaded cross-sections
        """
        self.opacity_dict = {}
