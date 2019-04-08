# init the package folder

import resources as rs
import os


# check if the CUDA can run
if rs.CC_ARCH != '':
    
    from dinf import dev_info

    def gpuinfo(extended=False):
        ''' Run the CUDA dev_info shared library to get info about the installed GPU devices. 
        '''

        if extended:
            info = dev_info(1)
            print info
        else:
            info = dev_info(0)
        

        return info


# cdir = os.path.dirname(os.path.abspath(__file__))
# if len([f for f in os.listdir(cdir) if 'dinf.' in f])>0: