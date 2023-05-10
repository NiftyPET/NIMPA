# initialise the module folder
__all__ = ['get_params', 'get_paths', 'extract_reso_part', 'sampling_masks'
           'create_mumap_core', 'create_nac_core', 'create_reso',
           'create_sampl_reso', 'create_sampl',
           'standard_analysis', 'estimate_fwhm', 'preproc']

from .params import get_params
from .ioaux import get_paths, extract_reso_part, sampling_masks
from .templates import create_mumap_core, create_nac_core, create_reso, create_sampl_reso, create_sampl
from .analysis import standard_analysis, estimate_fwhm
from .proc import preproc