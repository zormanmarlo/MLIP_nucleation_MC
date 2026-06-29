import yaml
from utils import Bias, logger


class Config:
    AVOGADRO = 6.022e23

    DEFAULTS = {
        'kT': 0.592,
        'ratio': '1:1',
        'input_file': None,
        'lower_energy_cutoff': 1.5,
        'lower_bonded_cutoff': 2.5,
        'energy_cutoff': 20,
        'n_rosenbluth_trials': 32,
        'bias_type': None,
        'min_target': 0,
        'output_rcut': False,
        'output_rcut_traj': False,
    }

    REQUIRED = [
        'box_length', 'num_particles', 'equil_steps', 'prod_steps', 'output_interval',
        'output_rcut', 'output_rcut_traj', 'internal_interval', 'seed', 'bias_type',
        'avbmc_rate', 'nvt_rate', 'translation_rate', 'swap_rate', 'max_displacement',
        'upper_bonded_cutoff', 'lower_bonded_cutoff', 'clust_cutoff', 'model_path',
        'input_path', 'kT', 'ratio', 'input_file', 'lower_energy_cutoff', 'energy_cutoff',
        'concentration', 'n_rosenbluth_trials',
    ]

    def __init__(self, config_file):
        '''Parse config file, fill defaults, resolve system size, then set up bias and move types'''
        self.parameters = self._parse_config_file(config_file)
        self._apply_defaults()
        self._parse_ratio()
        self._resolve_system_size()
        self._validate_required()
        self._set_bias()
        self._set_attributes()
        self._setup_move_types()
        self._log_summary()

    def _parse_config_file(self, config_file):
        '''Parse YAML configuration file and return parameters dict'''
        with open(config_file, 'r') as f:
            parameters = yaml.safe_load(f) or {}
        if not isinstance(parameters, dict):
            raise ValueError("Config file must contain YAML mapping (key: value format)")
        return parameters

    def _apply_defaults(self):
        '''Fill in default values for any parameters not provided'''
        for key, value in self.DEFAULTS.items():
            self.parameters.setdefault(key, value)
        if 'upper_bonded_cutoff' not in self.parameters:
            self.parameters['upper_bonded_cutoff'] = self.parameters['clust_cutoff']

    def _validate_required(self):
        '''Raise an error if any required parameter is still missing after defaults/derivation'''
        for param in self.REQUIRED:
            if param not in self.parameters:
                logger.error(f"Parameter '{param}' not set in configuration file.")
                raise ValueError(f"Missing required parameter: {param}")

    def _log_summary(self):
        '''Log a short summary of simulation configuration'''
        logger.info(f"Concentration: {self.parameters['concentration']} M")
        logger.info(f"Box length: {self.parameters['box_length']} Å")
        logger.info(f"Number of particles: {self.parameters['num_particles']}")
        logger.info(f"Bias: {self.parameters.get('bias_type', 'none')}")

    def _resolve_system_size(self):
        '''Validate or derive box_length / num_particles / concentration (need at least 2 of 3)'''
        has_box_length = 'box_length' in self.parameters
        has_num_particles = 'num_particles' in self.parameters
        has_concentration = 'concentration' in self.parameters
        param_count = sum([has_box_length, has_num_particles, has_concentration])

        if param_count == 3:
            calculated_concentration = self._calculate_concentration(
                self.parameters['box_length'], self.parameters['num_particles']
            )
            if abs(calculated_concentration - self.parameters['concentration']) > 0.01:
                logger.warning(f"Provided concentration ({self.parameters['concentration']} M) "
                               f"inconsistent with calculated ({calculated_concentration:.2f} M)")
        elif param_count == 2:
            if not has_concentration:
                self.parameters['concentration'] = self._calculate_concentration(
                    self.parameters['box_length'], self.parameters['num_particles']
                )
            elif not has_num_particles:
                self.parameters['num_particles'] = self._calculate_num_particles(
                    self.parameters['box_length'], self.parameters['concentration']
                )
            else:
                self.parameters['box_length'] = self._calculate_box_length(
                    self.parameters['num_particles'], self.parameters['concentration']
                )
        else:
            raise ValueError("Must specify at least 2 of: box_length, num_particles, concentration")

    def _parse_ratio(self):
        '''Parse ratio string like '1:1' or '2:1' into tuple of integers'''
        ratio_str = self.parameters['ratio']
        try:
            parts = ratio_str.split(':')
            if len(parts) != 2:
                raise ValueError(f"Invalid ratio format: {ratio_str}. Use format like '1:1' or '2:1'")
            self.ratio_type1, self.ratio_type2 = int(parts[0]), int(parts[1])
            self.total_ratio = self.ratio_type1 + self.ratio_type2
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid ratio format: {ratio_str}. Use format like '1:1' or '2:1'")

    def _box_volume_L(self, box_length):
        '''Volume of the cubic box (side length in Å) in liters'''
        return (box_length * 1e-10) ** 3 * 1000

    def _calculate_concentration(self, box_length, num_particles):
        '''Calculate molar concentration from box size and number of particles'''
        formula_units = num_particles // self.total_ratio * min(self.ratio_type1, self.ratio_type2)
        moles_nacl = formula_units / self.AVOGADRO
        return moles_nacl / self._box_volume_L(box_length)

    def _calculate_num_particles(self, box_length, concentration):
        '''Calculate number of particles from box size and concentration'''
        moles_nacl = concentration * self._box_volume_L(box_length)
        formula_units = int(moles_nacl * self.AVOGADRO)
        return formula_units * self.total_ratio

    def _calculate_box_length(self, num_particles, concentration):
        '''Calculate box length from number of particles and concentration'''
        formula_units = num_particles // self.total_ratio
        moles_nacl = formula_units / self.AVOGADRO
        volume_L = moles_nacl / concentration
        volume_A3 = volume_L / 1000 / (1e-10) ** 3
        return volume_A3 ** (1 / 3)

    def _set_bias(self):
        '''Initialize bias potential based on bias_type parameter'''
        bias_type = self.parameters['bias_type']
        if bias_type == 'harmonic':
            if 'bias_center' not in self.parameters:
                logger.warning("Parameter 'bias_center' not set for harmonic bias. Defaulting to 0.0.")
                self.parameters['bias_center'] = 0.0
            if 'bias_k' not in self.parameters:
                logger.warning("Parameter 'bias_k' not set for harmonic bias. Defaulting to 1.0.")
                self.parameters['bias_k'] = 1.0
            self.bias = Bias(type='harmonic', center=self.parameters['bias_center'],
                             force_constant=self.parameters['bias_k'],
                             min_size=self.parameters['min_target'])
        elif bias_type == 'linear':
            if 'bias_file' not in self.parameters:
                logger.warning("Parameter 'bias_file' not set for linear bias. Setting bias to zero")
            self.bias = Bias(type='linear', path=self.parameters.get('bias_file'),
                             max_size=self.parameters.get('max_target', 200),
                             min_size=self.parameters['min_target'])
        else:
            self.bias = None

    def _set_attributes(self):
        '''Set all parameters as instance attributes for easy access'''
        for key, value in self.parameters.items():
            setattr(self, key, value)

    def _setup_move_types(self):
        '''Create list of active move types and their normalized selection probabilities'''
        from moves import TranslationMove, SwapMove, InOutAVBMCMove, OutInAVBMCMove, NVTInOutMove, NVTOutInMove

        self.active_moves = []

        if getattr(self, 'translation_rate', 0) > 0:
            self.active_moves.append(('translation', self.translation_rate, TranslationMove))
        if getattr(self, 'swap_rate', 0) > 0:
            self.active_moves.append(('swap', self.swap_rate, SwapMove))

        if getattr(self, 'avbmc_rate', 0) > 0:
            sub_rate = self.avbmc_rate / 2
            self.active_moves.append(('inout_avbmc', sub_rate, InOutAVBMCMove))
            self.active_moves.append(('outin_avbmc', sub_rate, OutInAVBMCMove))
        if getattr(self, 'nvt_rate', 0) > 0:
            sub_rate = self.nvt_rate / 2
            self.active_moves.append(('nvt_inout', sub_rate, NVTInOutMove))
            self.active_moves.append(('nvt_outin', sub_rate, NVTOutInMove))

        if self.active_moves:
            total_rate = sum(rate for _, rate, _ in self.active_moves)
            self.move_probabilities = [rate / total_rate for _, rate, _ in self.active_moves]
        else:
            self.move_probabilities = []

    def get_parameters(self):
        '''Return copy of all configuration parameters'''
        return self.parameters.copy()
