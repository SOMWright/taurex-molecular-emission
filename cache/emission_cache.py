"""
Contains caching class for Molecular Emission Data
"""

from taurex.log import Logger
from taurex.cache.globalcache import GlobalCache
from taurex.core import Singleton
from emission_data import EmissionData


class EmissionCache(Singleton):
    def init(self):
        self.emission_dict = {}
        self._emission_data_path = None
        self.log = Logger('EmissionCache')
        self._force_active = []
        self.emission_data_classes = [EmissionData]

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
            if key in self.emission_dict:
                return self.emission_dict[key]
            else:
                # Otherwise throw an error
                self.log.error('Emission data for molecule %s could not be loaded', key)
                self.log.error('It could not be found in the local dictionary %s', list(self.emission_dict.keys()))
                self.log.error('Or paths %s', GlobalCache()['emission_data_path'])
                self.log.error('Try loading it manually/ putting it in a path')
                raise Exception('Opacity could not be loaded')

    def add_emission_data(self, emission_data, molecule_filter=None):

        self.log.info('Reading Emission Data %s', emission_data.molecule_name)
        if emission_data.molecule_name in self.emission_dict:
            self.log.warning('Emission data with name %s already in emission dictionary %s skipping',
                             emission_data.molecule_name,
                             self.emission_dict.keys())
            return
        if molecule_filter is not None:
            if emission_data.molecule_name in molecule_filter:
                self.log.info('Loading emission data %s into model', emission_data.molecule_name)
                self.emission_dict[emission_data.molecule_name] = emission_data
        else:
            self.log.info('Loading emission data %s into model', emission_data.molecule_name)
            self.emission_dict[emission_data.molecule_name] = emission_data

    def find_list_of_molecules(self):

        molecules = []

        for c in self.emission_data_classes:
            molecules.extend([x[0] for x in c.discover()])

        forced = self._force_active or []
        return set(molecules + forced + list(self.emission_dict.keys()))

    def load_emission_data_from_path(self, path, molecule_filter=None):

        for c in self.emission_data_classes:

            for mol, args in c.discover():
                self.log.debug('Klass: %s %s', mol, args)
                op = None
                if mol in molecule_filter and mol not in self.emission_dict:
                    if not isinstance(args, (list, tuple,)):
                        args = [args]
                    op = c(*args)

                if op is not None and op.molecule_name not in self.emission_dict:
                    self.add_emission_data(op, molecule_filter=molecule_filter)
                op = None  # Ensure garbage collection when run once

    def load_emission_data(self, emission_data=None, emiss_path=None, molecule_filter=None):

        if emiss_path is None:
            emiss_path = GlobalCache()['emission_data_path']

        if emission_data is not None:
            if isinstance(emission_data, (list,)):
                self.log.debug('Emission passed is list')
                for emission in emission_data:
                    self.add_emission_data(emission, molecule_filter=molecule_filter)
            elif isinstance(emission_data, EmissionData):
                self.add_emission_data(emission_data, molecule_filter=molecule_filter)
            else:
                self.log.error('Unknown type %s passed into opacities, should be a list, single \
                     opacity or None if reading a path', type(emission_data))
                raise Exception('Unknown type passed into opacities')
        else:
            self.load_emission_data_from_path(emiss_path, molecule_filter=molecule_filter)

    def clear_cache(self):
        """
        Clears all currently loaded cross-sections
        """
        self.emission_dict = {}
