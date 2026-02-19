import numpy as np
import re
from utils import is_float, Bias, logger

class Config:
    def __init__(self, config_file):
        '''Initialize configuration by parsing file and setting up parameters, bias, and move types'''
        self.parameters = self._parse_config_file(config_file)
        self._missing_parameters()
        self._set_bias()
        self._set_attributes()
        self._setup_move_types()

        logger.info(f"Concentration: {self.parameters['concentration']} M")
        logger.info(f"Box length: {self.parameters['box_length']} Å")
        logger.info(f"Number of particles: {self.parameters['num_particles']}")
        logger.info(f"Bias: {self.parameters.get('bias_type', 'none')}")

        self.default_params = [
            'box_length', 'num_particles', 'equil_steps', 'prod_steps', 'output_interval',
            'internal_interval', 'seed', 'bias_type', 'avbmc_rate', 'nvt_rate',
            'translation_rate', 'swap_rate', 'max_displacement', 'upper_cutoff',
            'lower_cutoff', 'clust_cutoff', 'ff_path', 'input_path', 'kT', 'ratio',
            'input_file', 'lower_energy_cutoff', 'energy_cutoff', 'concentration', 'debug_mode'
        ]
        for param in self.default_params:
            if param not in self.parameters:
                logger.error(f"Parameter '{param}' not set in configuration file.")
                raise ValueError(f"Missing required parameter: {param}")
    
    def _parse_config_file(self, config_file):
        '''Parse configuration file and convert values to appropriate types'''
        with open(config_file, 'r') as f:
            parameters = {}
            for line in f:
                if "#" not in line[0]:
                    match = re.match(r'^([^#=]+?)\s*=\s*([^\s#]+)', line)
                    if not match:
                        continue
                    key, value = match.group(1).strip(), match.group(2).strip()
                    if value.isdigit():
                        parameters[key] = int(value)
                    elif is_float(value):
                        parameters[key] = float(value)
                    elif value.lower() in ['true', 'false']:
                        parameters[key] = value.lower() == 'true'
                    else:
                        if value.lower() == "none":
                            parameters[key] = None
                        else:
                            parameters[key] = value
        return parameters

    def _set_bias(self):
        '''Initialize bias potential based on bias_type parameter'''
        if 'bias_type' not in self.parameters:
            self.parameters['bias_type'] = None
        
        if self.parameters['bias_type'] == 'harmonic':
            if 'bias_center' not in self.parameters:
                logger.warning("Parameter 'bias_center' not set for harmonic bias. Defaulting to 0.0.")
                self.parameters['bias_center'] = 0.0
            if 'bias_k' not in self.parameters:
                logger.warning("Parameter 'bias_k' not set for harmonic bias. Defaulting to 1.0.")
                self.parameters['bias_k'] = 1.0
            self.bias = Bias(center=self.parameters['bias_center'], type='harmonic', force_constant=self.parameters['bias_k'])
        elif self.parameters['bias_type'] == 'linear':
            if 'bias_file' not in self.parameters:
                logger.warning("Parameter 'bias_file' not set for linear bias. Setting bias to zero")
                self.bias = Bias(type='linear', max_size=self.parameters.get('max_target', 200))
            else:
                self.bias = Bias(path=self.parameters['bias_file'], type='linear', max_size=self.parameters.get('max_target', 200))
        else:
            self.bias = None
    
    def _missing_parameters(self):
        '''Set default values for missing parameters and validate system size consistency'''
        # if no kT is provided, set it to 0.592
        if 'kT' not in self.parameters:
            self.parameters['kT'] = 0.592
        
        # if no ratio is provided, default to 1:1
        if 'ratio' not in self.parameters:
            self.parameters['ratio'] = '1:1'

        # if no input file is provided, default to None
        if 'input_file' not in self.parameters:
            self.parameters['input_file'] = None

        # if no lower_cutoff is provided, default to 1.5
        if 'lower_energy_cutoff' not in self.parameters:
            self.parameters['lower_energy_cutoff'] = 1.5

        # if no energy_cutoff is provided, default to 20.0
        if 'energy_cutoff' not in self.parameters:
            self.parameters['energy_cutoff'] = 20

        # if no debug_mode is provided, default to False
        if 'debug_mode' not in self.parameters:
            self.parameters['debug_mode'] = False

        # Parse the ratio
        self._parse_ratio()

        # Ensure at least two parameters are provided (box_length, num_particles, concentration)
        # Check which combination of parameters we have
        has_box_length = 'box_length' in self.parameters
        has_num_particles = 'num_particles' in self.parameters
        has_concentration = 'concentration' in self.parameters
        param_count = sum([has_box_length, has_num_particles, has_concentration])
        
        if param_count == 3:
            # All three provided - validate consistency
            calculated_concentration = self._calculate_concentration(
                self.parameters['box_length'], self.parameters['num_particles']
            )
            if abs(calculated_concentration - self.parameters['concentration']) > 0.01:
                logger.warning(f"Provided concentration ({self.parameters['concentration']} M) "
                              f"inconsistent with calculated ({calculated_concentration:.2f} M)")
        
        elif param_count == 2:
            # Calculate the missing parameter
            if not has_concentration:
                self.parameters['concentration'] = self._calculate_concentration(
                    self.parameters['box_length'], self.parameters['num_particles']
                )
            elif not has_num_particles:
                self.parameters['num_particles'] = self._calculate_num_particles(
                    self.parameters['box_length'], self.parameters['concentration']
                )
            elif not has_box_length:
                self.parameters['box_length'] = self._calculate_box_length(
                    self.parameters['num_particles'], self.parameters['concentration']
                )
        
        elif param_count < 2:
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
    
    def _calculate_concentration(self, box_length, num_particles):
        '''Calculate molar concentration from box size and number of particles'''
        # Calculate based on ion ratio
        volume_L = (box_length * 1e-10) ** 3 * 1000  # Convert Å³ to L
        # For NaCl, the number of formula units is determined by the limiting ion
        formula_units = num_particles // self.total_ratio * min(self.ratio_type1, self.ratio_type2)
        moles_nacl = formula_units / 6.022e23
        return moles_nacl / volume_L
    
    def _calculate_num_particles(self, box_length, concentration):
        '''Calculate number of particles from box size and concentration'''
        volume_L = (box_length * 1e-10) ** 3 * 1000  # Convert Å³ to L
        moles_nacl = concentration * volume_L
        formula_units = int(moles_nacl * 6.022e23)
        return formula_units * self.total_ratio  # Total ions based on ratio
    
    def _calculate_box_length(self, num_particles, concentration):
        '''Calculate box length from number of particles and concentration'''
        formula_units = num_particles // self.total_ratio
        moles_nacl = formula_units / 6.022e23
        volume_L = moles_nacl / concentration
        volume_A3 = volume_L / 1000 / (1e-10)**3  # Convert L to Å³
        return volume_A3 ** (1/3)
    
    def _set_attributes(self):
        '''Set all parameters as instance attributes for easy access'''
        # Set all parameters as instance attributes for easy access
        for key, value in self.parameters.items():
            setattr(self, key, value)
    
    def _setup_move_types(self):
        '''Create list of active move types and their rates'''
        # Import moves here to avoid circular imports and keep system.py clean
        from moves import TranslationMove, SwapMove, InOutAVBMCMove, OutInAVBMCMove, NVTInOutMove, NVTOutInMove
        
        move_classes = {
            'translation': TranslationMove,
            'swap': SwapMove,
            'inout_avbmc': InOutAVBMCMove,
            'outin_avbmc': OutInAVBMCMove,
            'nvt_inout': NVTInOutMove,
            'nvt_outin': NVTOutInMove
        }
        
        self.active_moves = []  # List of (move_name, rate, move_class) tuples
        
        # Add moves that have non-zero rates
        if getattr(self, 'translation_rate', 0) > 0:
            self.active_moves.append(('translation', self.translation_rate, move_classes['translation']))
        
        if getattr(self, 'swap_rate', 0) > 0:
            self.active_moves.append(('swap', self.swap_rate, move_classes['swap']))
        
        # AVBMC moves (split rate between in->out and out->in)
        if getattr(self, 'avbmc_rate', 0) > 0:
            avbmc_sub_rate = self.avbmc_rate / 2
            self.active_moves.append(('inout_avbmc', avbmc_sub_rate, move_classes['inout_avbmc']))
            self.active_moves.append(('outin_avbmc', avbmc_sub_rate, move_classes['outin_avbmc']))
        
        # NVT moves (split rate between in->out and out->in)
        if getattr(self, 'nvt_rate', 0) > 0:
            nvt_sub_rate = self.nvt_rate / 2
            self.active_moves.append(('nvt_inout', nvt_sub_rate, move_classes['nvt_inout']))
            self.active_moves.append(('nvt_outin', nvt_sub_rate, move_classes['nvt_outin']))
        
        # Calculate normalized probabilities for move selection
        if self.active_moves:
            total_rate = sum(rate for _, rate, _ in self.active_moves)
            self.move_probabilities = [rate/total_rate for _, rate, _ in self.active_moves]
        else:
            self.move_probabilities = []
    
    def get_parameters(self):
        '''Return copy of all configuration parameters'''
        return self.parameters.copy()
