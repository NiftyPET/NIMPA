''' 
install tools for NiftyPET including:
* NiftyReg
* dcm2niix
'''
__author__      = "Pawel Markiewicz"
__copyright__   = "Copyright 2018"


import os
import sys
import multiprocessing
import platform
import shutil
import glob
from subprocess import call, Popen, PIPE
import re
import cudasetup as cs

if 'DISPLAY' in os.environ:
    from Tkinter import Tk
    from tkFileDialog import askdirectory

# NiftyReg git repository
repo_reg = 'https://cmiclab.cs.ucl.ac.uk/mmodat/niftyreg.git' #'git://git.code.sf.net/p/niftyreg/git'
# git SHA-1 checksum for NiftyReg version (16 Nov 2017) used for PET/MR image registration and resampling
sha1_reg = 'f673b7837c0824f55dedb1534b32b55bf68a2823'
#'6bf84b492050a4b9a93431209babeab9bc8f14da' 
#'62af1ca6777379316669b6934889c19863eaa708'
reg_ver = '1.5.58'

# dcm2niix git repository
repo_dcm = 'https://github.com/rordenlab/dcm2niix'
http_dcm_lin =  'https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20180622/dcm2niix_27-Jun-2018_lnx.zip'
http_dcm_win = 'https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20180622/dcm2niix_27-Jun-2018_win.zip'
# git SHA-1 checksum for the version used for PET/MR
sha1_dcm = '4b641113273d86ad73123816993092fc643ac62f'
dcm_ver = '1.0.20180622'

# source and build folder names
dirsrc = '_src'
dirbld = '_bld'

# number of threads
ncpu = multiprocessing.cpu_count()

def query_yesno(question):
    valid = {'yes': True, 'y': True, 'ye': True,
             'no': False, 'n': False}
    prompt = ' [Y/n]: '
    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if choice == '':
            return True
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def input_path(question, default=os.path.expanduser('~')):
    while True:
        question += '['+default+']:'
        path = raw_input(question)
        if os.path.isdir(path):
            return path
        else:
            print 'e> the provided path is not valid: '+str(path)
            
#-----------------------------------------------------------------------------------------------------
def check_depends():
    print 'i> checking if [CUDA], [git] and [cmake] are installed...'

    outdct = {'cuda':True, 'git':True, 'cmake':True}

    #-check if CUDA is installed
    try:
        out = call(['nvcc', '--version'])
    except OSError:
        print 'e> CUDA (nvcc) does not seem to be installed;'
        outdct['cuda'] = False

    #-check if git is installed
    try:
        out = call(['git', '--version'])
    except OSError:
        print 'e> git does not seem to be installed;  for help, visit: https://git-scm.com/download/'
        outdct['git'] = False

    #-check if cmake is installed
    try:
        out = call(['cmake', '--version'])
    except OSError:
        print 'e> git does not seem to be installed;  for help, visit: https://git-scm.com/download/'
        outdct['cmake'] = False

    return outdct

#--------------------------------------------------------------------
def check_version(Cnt, chcklst=['RESPATH','REGPATH','DCM2NIIX','HMUDIR']):
    ''' Check version and existence of all third-party software and input data.
        Output a dictionary with bool type of the requested bits in 'chcklst'
    '''

    # at start, assume that nothing is present yet
    output = {}
    for itm in chcklst:
        output[itm] = False

    # niftyreg reg_resample first
    if 'RESPATH' in chcklst and 'RESPATH' in Cnt:
        try:
            proc = Popen([Cnt['RESPATH'], '--version'], stdout=PIPE)
            out = proc.stdout.read()
            if reg_ver in out:
                output['RESPATH'] = True
        except OSError:
            print 'e> NiftyReg (reg_resample) either is NOT installed or is corrupt.'
    
    # niftyreg reg_aladin
    if 'REGPATH' in chcklst and 'REGPATH' in Cnt:
        try:
            proc = Popen([Cnt['REGPATH'], '--version'], stdout=PIPE)
            out = proc.stdout.read()
            if reg_ver in out:
                output['REGPATH'] = True
        except OSError:
            print 'e> NiftyReg (reg_aladin) either is NOT installed or is corrupt.'

    # dcm2niix
    if 'DCM2NIIX' in chcklst and 'DCM2NIIX' in Cnt:
        try:
            proc = Popen([Cnt['DCM2NIIX'], '-h'], stdout=PIPE)
            out = proc.stdout.read()
            if dcm_ver in re.search('(?<=dcm2niiX version v)\d{1,2}.\d{1,2}.\d*', out).group(0):
                output['DCM2NIIX'] = True
        except OSError:
            print 'e> dcm2niix either is NOT installed or is corrupt.'

    # hdw mu-map list
    if 'HMUDIR' in chcklst and 'HMUDIR' in Cnt:
        for hi in Cnt['HMULIST']:
            if os.path.isfile(os.path.join(Cnt['HMUDIR'],hi)):
                output['HMUDIR'] = True
            else:
                output['HMUDIR'] = False
                break

    return output
#--------------------------------------------------------------------
def download_dcm2niix(Cnt, path):
    print '================================================='
    print 'i> dcm2niix will be installed directly from:'
    print '   https://github.com/rordenlab/dcm2niix/releases'
    print '================================================='

    #-create the installation folder
    if not os.path.isdir(path):
        os.mkdir(path)
    binpath = os.path.join(path, 'bin')
    if not os.path.isdir(binpath):
        os.mkdir(binpath)

    import urllib, zipfile
    if platform.system()=='Windows':
        urllib.urlretrieve(
            http_dcm_win,
            os.path.join(path, 'dcm2niix.zip')
        )
    elif platform.system()=='Linux':
        urllib.urlretrieve(
            http_dcm_lin,
            os.path.join(path, 'dcm2niix.zip')
        )
    else:
        raise OSError('Unrecognised operating system.')

    zipf = zipfile.ZipFile(os.path.join(path, 'dcm2niix.zip'), 'r')
    zipf.extractall(os.path.join(path, 'bin'))
    zipf.close()
    Cnt['DCM2NIIX'] = glob.glob(os.path.join(os.path.join(path,'bin'), 'dcm2niix*'))[0]
    # ensure the permissions are given to the executable
    os.chmod(Cnt['DCM2NIIX'], 755)
    # update the resources.py file in ~/.niftypet
    Cnt = update_resources(Cnt)
    return Cnt


def install_tool(app, Cnt):
    ''' Install the requested software from the git 'repo'
        and check out the version given by 'sha1'.
    '''

    # get the current working directory
    cwd = os.getcwd()

    # pick the target installation folder
    if 'PATHTOOLS' in Cnt and Cnt['PATHTOOLS']!='':
        path_tools = Cnt['PATHTOOLS']
    elif ('PATHTOOLS' not in Cnt or Cnt['PATHTOOLS']!='') and 'DISPLAY' in os.environ:
        print '>>>>> DISPLAY', os.environ['DISPLAY']
        Tk().withdraw()
        dircore = askdirectory(title='choose a place for NiftyPET tools', initialdir=os.path.expanduser('~'))
        # get the full (combined path)
        path_tools = os.path.join(dircore, Cnt['DIRTOOLS'])
        Cnt['PATHTOOLS'] = path_tools
    else:
        if platform.system() == 'Linux' :
            path_tools = os.path.join( os.path.expanduser('~'), Cnt['DIRTOOLS'] )
        elif platform.system() == 'Windows' :
            path_tools = os.path.join( os.getenv('LOCALAPPDATA'), Cnt['DIRTOOLS'] )
        else:
            print 'e> only Linux and Windows operating systems are supported!'
            raise SystemError('OS not supported!')      
        Cnt['PATHTOOLS'] = path_tools

    #create the main tools folder
    if not os.path.isdir(path_tools):
        os.mkdir(path_tools)
    # identify the specific path for the requested app
    if app=='niftyreg':
        repo = repo_reg
        sha1 = sha1_reg
        path = os.path.join(path_tools, 'niftyreg')
    elif app=='dcm2niix':
        repo = repo_dcm
        sha1 = sha1_dcm
        path = os.path.join(path_tools, 'dcm2niix')
        # avoid installing from source, instead download the full version:
        Cnt = download_dcm2niix(Cnt, path)
        return Cnt

    # Check if the source folder exists and delete it, if it does
    if os.path.isdir(path): shutil.rmtree(path)
    # Create an empty folder and enter it
    os.mkdir(path)
    os.chdir(path)

    # clone the git repository
    call(['git', 'clone', repo, dirsrc])
    os.chdir(dirsrc)
    print 'i> checking out the specific git version of the software...'
    call(['git', 'checkout', sha1])
    os.chdir('../')

    # create the building folder
    if not os.path.isdir(dirbld):
        os.mkdir(dirbld)
    # go inside the build folder
    os.chdir(dirbld)

    # run cmake with arguments
    if platform.system()=='Windows':
        cmd = ['cmake', '../'+dirsrc,
            '-DBUILD_ALL_DEP=ON',
            '-DCMAKE_INSTALL_PREFIX='+path,
            '-G', Cnt['MSVC_VRSN']]
        call(cmd)
        call(['cmake', '--build', './', '--config', 'Release', '--target', 'install'])
    elif platform.system()=='Linux':
        cmd = ['cmake', '../'+dirsrc,
            '-DBUILD_ALL_DEP=ON',
            '-DCMAKE_INSTALL_PREFIX='+path]
        if Cnt['CMAKE_TLS_PAR']!='': cmd.append(Cnt['CMAKE_TLS_PAR'])
        call(cmd)
        call(
            ['cmake', '--build', './',
            '--config', 'Release',
            '--target', 'install',
            '--','-j', str(ncpu)]
        )

    # restore the current working directory
    os.chdir(cwd)

    if app=='niftyreg':
        try:
            Cnt['RESPATH'] = glob.glob(os.path.join(os.path.join(path,'bin'), 'reg_resample*'))[0]
            Cnt['REGPATH'] = glob.glob(os.path.join(os.path.join(path,'bin'), 'reg_aladin*'))[0]
        except IndexError:
            print 'e> NiftyReg has NOT been successfully installed.'
            raise SystemError('Failed Installation (NiftyReg)')
        # updated the file resources.py
        Cnt = update_resources(Cnt)
        # check the installation:
        chck_niftyreg = check_version(Cnt, chcklst=['RESPATH','REGPATH'])
        if not all([chck_niftyreg[k] for k in chck_niftyreg.keys()]):
            print 'e> NiftyReg has NOT been successfully installed.'
            raise SystemError('Failed Installation (NiftyReg)')

    elif app=='dcm2niix':
        try:
            Cnt['DCM2NIIX'] = glob.glob(os.path.join(os.path.join(path,'bin'), 'dcm2niix*'))[0]
        except IndexError:
            print 'e> dcm2niix has NOT been successfully installed.'
            Cnt = download_dcm2niix(Cnt, path)
        # check the installation:
        if not check_version(Cnt, chcklst=['DCM2NIIX']):
            print 'e> dcm2niix has NOT been successfully compiled from github.'
            Cnt = download_dcm2niix(Cnt, path)
    return Cnt
    

def update_resources(Cnt):
    '''Update resources.py with the paths to the new installed apps.
    '''

    # list of path names which will be saved
    key_list = ['PATHTOOLS', 'RESPATH', 'REGPATH', 'DCM2NIIX', 'HMUDIR']

    # get the local path to NiftyPET resources.py
    path_resources = cs.path_niftypet_local()
    resources_file = os.path.join(path_resources,'resources.py')

    # update resources.py
    if os.path.isfile(resources_file):
        f = open(resources_file, 'r')
        rsrc = f.read()
        f.close()
        # get the region of keeping in synch with Python
        i0 = rsrc.find('### start NiftyPET tools ###')
        i1 = rsrc.find('### end NiftyPET tools ###')
        pth_list = []
        for k in key_list:
            if k in Cnt:
                pth_list.append('\'' + Cnt[k].replace("\\","/") + '\'')
            else:
                pth_list.append('\'\'')

        # modify resources.py with the new paths
        strNew = '### start NiftyPET tools ###\n'
        for i in range(len(key_list)):
            if pth_list[i] != '\'\'':
                strNew += key_list[i]+' = '+pth_list[i] + '\n'
        rsrcNew = rsrc[:i0] + strNew + rsrc[i1:]
        f = open(resources_file, 'w')
        f.write(rsrcNew)
        f.close()

    return Cnt

#---------