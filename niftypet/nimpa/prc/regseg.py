""" NIMPA: functions for neuro image processing and analysis.
	Includes functions relating to image registration/segmentation.
    
"""
__author__    = "Pawel Markiewicz"
__copyright__ = "Copyright 2019"
#-------------------------------------------------------------------------------

import sys
import os
import shutil
from subprocess import call

import numpy as np
import scipy.ndimage as ndi

import imio
import prc



def imfill(immsk):
    '''fill the empty patches of image mask 'immsk' '''

    for iz in range(immsk.shape[0]):
        for iy in range(immsk.shape[1]):
            ix0 = np.argmax(immsk[iz,iy,:]>0)
            ix1 = immsk.shape[2] - np.argmax(immsk[iz,iy,::-1]>0)
            if (ix1-ix0) > immsk.shape[2]-10: continue
            immsk[iz,iy,ix0:ix1] = 1
    return immsk



#-------------------------------------------------------------------------------
# Create object mask for the input image 
#-------------------------------------------------------------------------------
def create_mask(
        fnii,
        fimout = '',
        outpath = '',
        fill = 1,
        dtype_fill = np.uint8,
        
        thrsh = 0.,
        fwhm = 0.,):

    ''' create mask over the whole image or over the threshold area'''
    

    #> output path
    if outpath=='' and fimout!='':
        opth = os.path.dirname(fimout)
        if opth=='':
            opth = os.path.dirname(fnii)
            fimout = os.path.join(opth, fimout)

    elif outpath=='':
        opth = os.path.dirname(fnii)

    else:
        opth = outpath

    #> output file name if not given
    if fimout=='':
        fniis = os.path.split(fnii)
        fimout = os.path.join(opth, fniis[1].split('.nii')[0]+'_mask.nii.gz')

    niidct = imio.getnii(fnii, output='all')
    im = niidct['im']
    hdr = niidct['hdr']

    if im.ndim>3:
        raise ValueError('The masking function only accepts 3-D images.')

    #> generate output image
    if thrsh>0.:
        smoim = ndi.filters.gaussian_filter(
                    im,
                    imio.fwhm2sig(fwhm, voxsize=abs(hdr['pixdim'][1])), 
                    mode='mirror')
        thrsh = thrsh*smoim.max()
        immsk = np.int8(smoim>thrsh)
        immsk = imfill(immsk)

        #> output image
        imo = fill * immsk.astype(dtype_fill)

    else:

        imo = fill * np.ones(im.shape, dtype = dtype_fill)

    #> save output image
    imio.array2nii( 
                imo,
                niidct['affine'],
                fimout,
                trnsp = (niidct['transpose'].index(0),
                         niidct['transpose'].index(1),
                         niidct['transpose'].index(2)),
                flip = niidct['flip'])

    return {'fim':fimout, 'im':imo}
#-------------------------------------------------------------------------------



# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# I M A G E   R E G I S T R A T I O N
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>


#-------------------------------------------------------------------------------
def affine_niftyreg(
    fref,
    fflo,
    outpath='',
    fname_aff='',
    pickname='ref',
    fcomment='',
    executable = '',
    omp=1,
    rigOnly = False,
    affDirect = False,
    maxit=5,
    speed=True,
    pi=50, pv=50,
    smof=0, smor=0,
    rmsk=True,
    fmsk=True,
    rfwhm=15.,
    rthrsh=0.05,
    ffwhm = 15.,
    fthrsh=0.05,
    verbose=True):

    # check if the executable exists:
    if not os.path.isfile(executable):
        raise IOError('Incorrect path to executable file for registration.')

    #create a folder for images registered to ref
    if outpath!='':
        odir = os.path.join(outpath,'affine-niftyreg')
        fimdir = os.path.join(outpath, os.path.join('affine-niftyreg','mask'))
    else:
        odir = os.path.join(os.path.dirname(fflo),'affine-niftyreg')
        fimdir = os.path.join(os.path.dirname(fflo), 'affine-niftyreg', 'mask')
    imio.create_dir(odir)
    imio.create_dir(fimdir)

    if rmsk:
        f_rmsk = os.path.join(fimdir, 'rmask_'+os.path.basename(fref).split('.nii')[0]+'.nii.gz')
        create_mask(fref, fimout = f_rmsk, thrsh = rthrsh, fwhm = rfwhm)
    
    if fmsk:
        f_fmsk = os.path.join(fimdir, 'fmask_'+os.path.basename(fflo).split('.nii')[0]+'.nii.gz')
        create_mask(fflo, fimout = f_fmsk, thrsh = fthrsh, fwhm = ffwhm)

    # output in register with ref and text file for the affine transform
    if fname_aff!='':
        fout = os.path.join(odir, fname_aff.split('.')[0]+'.nii.gz')
        faff = os.path.join(odir, fname_aff)
    else:
        if pickname=='ref':
            fout = os.path.join(odir, 'affine_ref-' \
                +os.path.basename(fref).split('.nii')[0]+fcomment+'.nii.gz')
            faff = os.path.join(odir, 'affine_ref-' \
                +os.path.basename(fref).split('.nii')[0]+fcomment+'.txt')
        elif pickname=='flo':
            fout = os.path.join(odir, 'affine_flo-' \
                +os.path.basename(fflo).split('.nii')[0]+fcomment+'.nii.gz')
            faff = os.path.join(odir, 'affine_flo-' \
                +os.path.basename(fflo).split('.nii')[0]+fcomment+'.txt')
    
    # call the registration routine
    cmd = [executable,
         '-ref', fref,
         '-flo', fflo,
         '-aff', faff,
         '-pi', str(pi),
         '-pv', str(pv),
         '-smooF', str(smof),
         '-smooR', str(smor),
         '-maxit', '10',
         '-omp', str(omp),
         '-res', fout]
    if speed:
        cmd.append('-speeeeed')
    if rigOnly:
        cmd.append('-rigOnly')
    if affDirect:
        cmd.append('affDirect')
    if rmsk: 
        cmd.append('-rmask')
        cmd.append(f_rmsk)
    if fmsk:
        cmd.append('-fmask')
        cmd.append(f_fmsk)
    if not verbose:
        cmd.append('-voff')

    call(cmd)

    #> affine to Numpy array
    aff = np.loadtxt(faff)
       
    return {'affine':aff, 'faff':faff, 'fim':fout} #faff, fout
#-------------------------------------------------------------------------------------

#-------------------------------------------------------------------------------------
def resample_niftyreg(
        fref,
        fflo,
        faff,
        outpath = '',
        fimout = '',
        fcomment = '',
        pickname = 'ref',
        intrp = 1,
        executable = '',
        verbose = True):

    # check if the executable exists:
    # if executable=='' and 'RESPATH' in Cnt and os.path.isfile(Cnt['RESPATH']):
    #     executable = Cnt['RESPATH']
    
    if not os.path.isfile(executable):
        raise IOError('Incorrect path to executable file for registration.')

    #> output path
    if outpath=='' and fimout!='':
        opth = os.path.dirname(fimout)
        if opth=='':
            opth = os.path.dirname(fflo)

    elif outpath=='':
        opth = os.path.dirname(fflo)

    else:
        opth = outpath

    imio.create_dir(opth)

    #> the output naming
    if '/' in fimout:
        fout = fimout
    elif fimout!='' and not os.path.isfile(fimout):
        fout = os.path.join(opth, fimout)
    elif pickname=='ref':
        fout = os.path.join(opth, 'affine_ref-' \
                + os.path.basename(fref).split('.nii')[0]+fcomment+'.nii.gz')
    elif pickname=='flo':        
        fout = os.path.join(opth, 'affine_flo-' \
                + os.path.basename(fflo).split('.nii')[0]+fcomment+'.nii.gz')

    if isinstance(intrp, (int, long)): intrp = str(intrp)
    
    cmd = [executable,
       '-ref', fref,
       '-flo', fflo,
       '-trans', faff,
       '-res', fout,
       '-inter', intrp
       ]
    if not verbose:
        cmd.append('-voff')
    call(cmd)

    return fout
#-------------------------------------------------------------------------------------



#===============================================================================
# S P M registration
#===============================================================================
def coreg_spm(
        imref,
        imflo,
        matlab_eng='',
        outpath='',
        fname_aff='',
        fcomment = '',
        pickname='ref',
        costfun='nmi',
        sep = [4,2],
        tol = [ 0.0200,0.0200,0.0200,0.0010,0.0010,0.0010,
                0.0100,0.0100,0.0100,0.0010,0.0010,0.0010],
        fwhm = [7,7],
        params = [0,0,0,0,0,0],
        graphics = 1,
        visual = 0,
        del_uncmpr=True,
        save_arr = True,
        save_txt = True,
    ):

    import matlab
    from pkg_resources import resource_filename

    #-start Matlab engine if not given
    if matlab_eng=='':
        import matlab.engine
        eng = matlab.engine.start_matlab()
    else:
        eng = matlab_eng

    # add path to SPM matlab file
    spmpth = resource_filename(__name__, 'spm')
    print 'PATH: ' + spmpth
    eng.addpath(spmpth, nargout=0)

    #> output path
    if outpath=='' and fname_aff!='' and '/' in fname_aff:
        opth = os.path.dirname(fname_aff)
        if opth=='':
            opth = os.path.dirname(imflo)
        fname_aff = os.path.basename(fname_aff)
    elif outpath=='':
        opth = os.path.dirname(imflo)
    else:
        opth = outpath
    imio.create_dir(opth)

    #> decompress ref image as necessary 
    if imref[-3:]=='.gz':
        imrefu = imio.nii_ugzip(imref, outpath=opth)
    else:
        fnm = os.path.basename(imref).split('.nii')[0] + '_copy.nii'
        imrefu = os.path.join(opth, fnm)
        shutil.copyfile(imref, imrefu)
    
    #> floating
    if imflo[-3:]=='.gz': 
        imflou = imio.nii_ugzip(imflo, outpath=opth)
    else:
        fnm = os.path.basename(imflo).split('.nii')[0] + '_copy.nii'
        imflou = os.path.join(opth, fnm)
        shutil.copyfile(imflo, imflou)

    # run the matlab SPM coregistration
    Mm, xm = eng.coreg_spm_m(
        imrefu,
        imflou,
        costfun,
        matlab.double(sep),
        matlab.double(tol),
        matlab.double(fwhm),
        matlab.double(params),
        graphics,
        visual,
        nargout=2
    )

    # get the affine matrix
    M = np.array(Mm._data.tolist())
    M = M.reshape(4,4).T

    # get the translation and rotation parameters in a vector
    x = np.array(xm._data.tolist())

    # delete the uncompressed files
    if del_uncmpr:
        if imref[-3:]=='.gz': os.remove(imrefu)
        if imflo[-3:]=='.gz': os.remove(imflou)

    imio.create_dir( os.path.join(opth, 'affine-spm') )

    #---------------------------------------------------------------------------
    if fname_aff == '':
        if pickname=='ref':
            faff = os.path.join(
                    opth,
                    'affine-spm',
                    'affine-ref-'+os.path.basename(imref).split('.nii')[0]+fcomment+'.npy')
        else:
            faff = os.path.join(
                    opth,
                    'affine-spm',
                    'affine-flo-'+os.path.basename(imflo).split('.nii')[0]+fcomment+'.npy')
    
    else:

        #> add '.npy' extension if not in the affine output file name
        if not fname_aff.endswith('.npy'):
            fname_aff += '.npy'

        faff = os.path.join(
                opth,
                'affine-spm',
                fname_aff)
    #---------------------------------------------------------------------------

    #> safe the affine transformation
    if save_arr:
        np.save(faff, M)
    if save_txt:
        np.savetxt(faff.split('.npy')[0]+'.txt', M)
    
    return {'affine':M, 'faff':faff,
            'rotations':x[3:], 'translations':x[:3],
            'matlab_eng':eng}
#===============================================================================





#===============================================================================
# S P M resampling
#===============================================================================
def resample_spm(
        imref,
        imflo,
        M,
        matlab_eng='',
        intrp=1.,
        which=1,
        mask=0,
        mean=0,
        outpath='',
        fimout='',
        fcomment='',
        prefix='r_',
        pickname='ref',
        del_ref_uncmpr=False,
        del_flo_uncmpr=False,
        del_out_uncmpr=False
    ):


    print '====================================================================='
    print ' S P M  inputs:'
    print '> ref:', imref
    print '> flo:', imflo
    print '====================================================================='

    import matlab
    from pkg_resources import resource_filename

    #-start Matlab engine if not given
    if matlab_eng=='':
        import matlab.engine
        eng = matlab.engine.start_matlab()
    else:
        eng = matlab_eng

    # add path to SPM matlab file
    spmpth = resource_filename(__name__, 'spm')
    eng.addpath(spmpth, nargout=0)

    #> output path
    if outpath=='' and fimout!='':
        opth = os.path.dirname(fimout)
        if opth=='':
            opth = os.path.dirname(imflo)

    elif outpath=='':
        opth = os.path.dirname(imflo)

    else:
        opth = outpath

    imio.create_dir(opth)

    #> decompress if necessary 
    if imref[-3:]=='.gz':
        imrefu = imio.nii_ugzip(imref, outpath=opth)
    else:
        fnm = os.path.basename(imref).split('.nii')[0] + '_copy.nii'
        imrefu = os.path.join(opth, fnm)
        shutil.copyfile(imref, imrefu)

    #> floating
    if imflo[-3:]=='.gz': 
        imflou = imio.nii_ugzip(imflo, outpath=opth)
    else:
        fnm = os.path.basename(imflo).split('.nii')[0] + '_copy.nii'
        imflou = os.path.join(opth, fnm)
        shutil.copyfile(imflo, imflou)

    if isinstance(M, basestring):
        M = np.load(M)
        print 'i> matrix M given in the form of numpy file'
    elif isinstance(M, (np.ndarray, np.generic)):
        print 'i> matrix M given in the form of Numpy array'
    else:
        raise IOError('The form of affine matrix not recognised.')

    # run the Matlab SPM resampling
    r = eng.resample_spm_m(
        imrefu,
        imflou,
        matlab.double(M.tolist()),
        mask,
        mean,
        intrp,
        which,
        prefix)

    
    #-compress the output
    split = os.path.split(imflou)
    fim = os.path.join(split[0], prefix+split[1])
    imio.nii_gzip(fim, outpath=opth)

    # delete the uncompressed
    if del_ref_uncmpr:  os.remove(imrefu)
    if del_flo_uncmpr and os.path.isfile(imflou):  os.remove(imflou)
    if del_out_uncmpr: os.remove(fim)

    #> the compressed output naming
    if fimout!='':
        fout = os.path.join(opth, fimout)
    elif pickname=='ref':
        fout = os.path.join(opth, 'affine_ref-' \
                + os.path.basename(imrefu).split('.nii')[0]+fcomment+'.nii.gz')
    elif pickname=='flo':        
        fout = os.path.join(opth, 'affine_flo-' \
                + os.path.basename(imflo).split('.nii')[0]+fcomment+'.nii.gz')
    # change the file name
    os.rename(fim+'.gz', fout)

    return fout

#===============================================================================



#===============================================================================
# V I N C I  registration and resampling
#===============================================================================

def coreg_vinci(
        fref,
        fflo,
        vc = '',
        con = '',
        vincipy_path = '',
        scheme_xml = '',
        outpath = '',
        fname_aff = '',
        pickname = 'ref',
        fcomment = '',
        flo_colourmap = 'Green',
        close_vinci = False,
        close_buff = True,
        cleanup = True,
        save_res = False,
        ):

    if scheme_xml=='':
        raise IOError('e> the Vinci schema *.xml file is not provided. \n \
                i> please add the schema file in the call: scheme_xml=...')

    #---------------------------------------------------------------------------
    #> output path
    if outpath=='' and fname_aff!='' and '/' in fname_aff:
        opth = os.path.dirname(fname_aff)
        if opth=='':
            opth = os.path.dirname(fflo)
        fname_aff = os.path.basename(fname_aff)

    elif outpath=='':
        opth = os.path.dirname(fflo)
    else:
        opth = outpath
    imio.create_dir(opth)

    imio.create_dir(os.path.join(opth, 'affine-vinci'))

    #> output floating and affine file names
    if fname_aff=='':
        if pickname=='ref':

            faff = os.path.join(
                    opth,
                    'affine-vinci',
                    'affine-ref-' \
                    +os.path.basename(fref).split('.nii')[0]+fcomment+'.xml')

            fout = os.path.join(
                    opth,
                    'affine-vinci',
                    'affine-ref-' \
                    +os.path.basename(fref).split('.nii')[0]+fcomment+'.nii.gz')

        else:
            faff = os.path.join(
                    opth,
                    'affine-vinci',
                    'affine-flo-' \
                    +os.path.basename(fflo).split('.nii')[0]+fcomment+'.xml')

            fout = os.path.join(
                    opth,
                    'affine-vinci',
                    'affine-flo-' \
                    +os.path.basename(fflo).split('.nii')[0]+fcomment+'.nii.gz')

    else:

        #> add '.xml' extension if not in the affine output file name
        if not fname_aff.endswith('.xml'):
            fname_aff += '.xml'

        faff = os.path.join(
                opth,
                'affine-vinci',
                fname_aff)

        fout = os.path.join(
                opth,
                fname_aff.split('.')[0]+'.nii.gz')
    #---------------------------------------------------------------------------

    if vincipy_path=='':
        try:
            import resources
            vincipy_path = resources.VINCIPATH
        except:
            raise NameError('e> could not import resources \
                    or find variable VINCIPATH in resources.py')

    sys.path.append(vincipy_path)

    try:
        from VinciPy import Vinci_Bin, Vinci_Connect, Vinci_Core, Vinci_XML, Vinci_ImageT
    except ImportError:
        raise ImportError('e> could not import Vinci:\n \
                check the variable VINCIPATH (path to Vinci) in resources.py')

    #> start Vinci core engine if it is not running/given
    if vc=='' or con=='':
        #> adopted from the Vinci's example RunMMMJob.py
        bin = Vinci_Bin.Vinci_Bin()
        con = Vinci_Connect.Vinci_Connect(bin)
        vinci_binpath = con.StartMyVinci()

        vc = Vinci_Core.Vinci_CoreCalc(con)
        vc.StdProject()

    #> read the registration schema file 
    f = open(scheme_xml, 'rb')
    reg_scheme = f.read()
    f.close()

    #> check scheme data file and remove possible XML comments at its beginning
    root = Vinci_XML.ElementTree.fromstring(reg_scheme)
    if root.tag != "MMM":
        sys.exit("scheme data file %s does not contain tag MMM\n"%scheme_xml)
    reg_scheme = Vinci_XML.ElementTree.tostring(root)
    # bytes -> str
    reg_scheme = reg_scheme.decode("utf-8")

    #> pick reference and floating images
    ref = Vinci_ImageT.newTemporary(vc, szFileName=fref)
    rsl = Vinci_ImageT.newTemporary(vc, szFileName=fflo)

    #> set the colour map for the floating image
    rsl.setColorSettings(CTable = flo_colourmap)

    #---------------------------------------------------------------------------
    #> perform the registration
    rsl.alignToRef(ref, reg_scheme, szRegistrationSummaryFile=faff)
    #---------------------------------------------------------------------------

    #> find the weird folder name created and to be disposed of.
    if cleanup:
        #> get the folder name (going to rubbish)
        del_dir = os.path.normpath(faff).split(os.sep)[1]
        fflo_dir = os.path.split(fflo)[0]
        del_dir = os.path.join(
            os.path.split(fflo)[0],
            os.path.normpath(faff).split(os.sep)[1])
        if os.path.isdir(del_dir):
            shutil.rmtree(del_dir)



    #> save the registered image
    if save_res:
        rsl.saveYourselfAs(bUseOffsetRotation=True, szFullFileName=fout)

    #> close image buffers for reference and floating
    if close_buff:
        rsl.killYourself()
        ref.killYourself()

    if close_vinci: con.CloseVinci(True)
    #else:          con.CloseSockets()


    return {'faff':faff, 'fim':fout, 'vinci_con':con, 'vinci_vc':vc}




#-------------------------------------------------------------------------------
# VINCI RESAMPLE
#-------------------------------------------------------------------------------

def resample_vinci(
        fref,
        fflo,
        faff,
        intrp = 0,
        fimout = '',
        fcomment = '',
        outpath = '',
        pickname = 'ref',
        vc = '',
        con = '',
        vincipy_path = '',
        atlas_resample = False,
        atlas_ref_make = False,
        atlas_ref_del = False,
        close_vinci = False,
        close_buff = True,
        ):
    ''' Resample the floating image <fflo> into the geometry of <fref>,
        using the Vinci transformation output <faff> (an *.xml file).
        Output the NIfTI file path of the resampled/resliced image.
    '''

    #---------------------------------------------------------------------------
    #> output path
    if outpath=='' and fimout!='' and '/' in fimout:
        opth = os.path.dirname(fimout)
        if opth=='':
            opth = os.path.dirname(fflo)
        fimout = os.path.basename(fimout)

    elif outpath=='':
        opth = os.path.dirname(fflo)
    else:
        opth = outpath
    imio.create_dir(opth)

    #> output floating and affine file names
    if fimout=='':
        if pickname=='ref':
            fout = os.path.join(
                    opth,
                    'affine-ref-' \
                    +os.path.basename(fref).split('.nii')[0]+fcomment+'.nii.gz')
        else:
            fout = os.path.join(
                    opth,
                    'affine-flo-' \
                    +os.path.basename(fflo).split('.nii')[0]+fcomment+'.nii.gz')
    else:
        fout = os.path.join(
                opth,
                fimout.split('.')[0]+'.nii.gz')
    #---------------------------------------------------------------------------


    if vincipy_path=='':
        try:
            import resources
            vincipy_path = resources.VINCIPATH
        except:
            raise NameError('e> could not import resources \
                    or find variable VINCIPATH in resources.py')

    sys.path.append(vincipy_path)

    try:
        from VinciPy import Vinci_Bin, Vinci_Connect, Vinci_Core, Vinci_XML, Vinci_ImageT
    except ImportError:
        raise ImportError('e> could not import Vinci:\n \
                check the variable VINCIPATH (path to Vinci) in resources.py')


    #---------------------------------------------------------------------------
    #> start Vinci core engine if it is not running/given
    if vc=='' or con=='':
        #> adopted from the Vinci's example RunMMMJob.py
        bin = Vinci_Bin.Vinci_Bin()
        con = Vinci_Connect.Vinci_Connect(bin)
        vinci_binpath = con.StartMyVinci()

        vc = Vinci_Core.Vinci_CoreCalc(con)
        vc.StdProject()
    #---------------------------------------------------------------------------

    if int(intrp)==0:
        interp_nn = True
    else:
        interp_nn = False

    #---------------------------------------------------------------------------
    #> change the reference image to be loaded as atlas
    if atlas_resample and atlas_ref_make:

        #> output file name for the extra reference image
        fextref = os.path.join(opth, 'reference-as-atlas.nii.gz')

        prc.nii_modify(fref, fimout=fextref, voxel_range=[0., 255.])
        
        ref = Vinci_ImageT.newTemporary(
                vc,
                szFileName = fextref,
                bLoadAsAtlas = atlas_resample)

        if atlas_ref_del:
            os.remove(fextref)

    else:
        ref = Vinci_ImageT.newTemporary(
                vc,
                szFileName = fref,
                bLoadAsAtlas = atlas_resample)
    #---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    #> reapply rigid transformation to new image


    flo = Vinci_ImageT.newTemporary(
            vc,
            szFileName = fflo,
            bLoadAsAtlas = atlas_resample)

    rsl = flo.reapplyMMMTransform(
            faff,
            refImage = ref,
            IsComputed=True)

    rsl.saveYourselfAs(
            bUseOffsetRotation = True,
            bUseNextNeighbourInterp = interp_nn,
            szFullFileName = fout)
    #---------------------------------------------------------------------------    


    #> close image buffers for reference and floating
    if close_buff:
        ref.killYourself()
        flo.killYourself()
        rsl.killYourself()
        


    if close_vinci: con.CloseVinci(True)

    return fout
    #return {'fim':fout, 'vinci_con':con, 'vinci_vc':vc}





#-------------------------------------------------------------------------------
# FSL-FLIRT REGISTRATION (AFFINE/RIGID)
#-------------------------------------------------------------------------------


def affine_fsl(
        fref,
        fflo,
        outpath='',
        fname_aff='',
        pickname = 'ref',
        fcomment='',
        executable='',
        costfun='normmi',
        dof=6,
        hstbins=256,
        verbose=True):


    if executable=='':
        if 'FSLDIR' not in os.environ:
            raise IOError('e> no FSL executable provided!')
        else:
            executable = os.path.join(os.environ['FSLDIR'], 'bin', 'flirt')

    if not os.path.isfile(executable) or not call([executable, '-version'])==0:
        raise IOError('e> no valid FSL executable provided!')

    #---------------------------------------------------------------------------
    #> output path
    if outpath=='' and fname_aff!='' and os.path.isfile(fname_aff):
        opth = os.path.dirname(fname_aff)
        if opth=='':
            opth = os.path.dirname(fflo)
        fname_aff = os.path.basename(fname_aff) 

    elif outpath=='':
        opth = os.path.dirname(fflo)
    else:
        opth = outpath
    imio.create_dir(opth)

    imio.create_dir(os.path.join(opth, 'affine-fsl'))

    #> output floating and affine file names
    if fname_aff=='':
        
        if pickname=='ref':
            faff = os.path.join(
                    opth,
                    'affine-fsl',
                    'affine-ref-' \
                    +os.path.basename(fref).split('.nii')[0]+fcomment+'.txt')

        else:
            faff = os.path.join(
                    opth,
                    'affine-fsl',
                    'affine-flo-' \
                    +os.path.basename(fflo).split('.nii')[0]+fcomment+'.txt')

    else:

        #> add '.xml' extension if not in the affine output file name
        if not fname_aff.endswith('.txt'):
            fname_aff += '.txt'

        faff = os.path.join(
                opth,
                'affine-vinci',
                fname_aff)
    #---------------------------------------------------------------------------
    
    #> command with options for FSL-FLIRT registration
    cmd = [ executable,
            '-cost', costfun,
            '-dof', str(dof),
            '-omat', faff,
            '-in', fflo,
            '-ref', fref,
            '-bins', str(hstbins)]

    #> add verbose mode if requested
    if isinstance(verbose, bool): verbose = int(verbose)
    if verbose>0: cmd.extend(['-verbose', str(verbose)])

    #> execute FSL-FLIRT command with the above options
    call(cmd)

    # convert to Numpy float
    aff = np.loadtxt(faff)

    return {'affine':aff, 'faff':faff}


#-------------------------------------------------------------------------------
# FSL RESAMPLING
#-------------------------------------------------------------------------------
def resample_fsl(
        imref,
        imflo,
        faff,
        outpath = '',
        fimout = '',
        pickname = 'ref',
        fcomment = '',
        intrp = 1,
        executable = ''):


    if executable=='':
        if 'FSLDIR' not in os.environ:
            raise IOError('e> no FSL executable provided!')
        else:
            executable = os.path.join(os.environ['FSLDIR'], 'bin', 'flirt')

    if not os.path.isfile(executable) or not call([executable, '-version'])==0:
        raise IOError('e> no valid FSL executable provided!')

    print '==================================================================='
    print ' F S L  resampling'

    #> output path
    if outpath=='' and fimout!='':
        opth = os.path.dirname(fimout)
        if opth=='':
            opth = os.path.dirname(imflo)

    elif outpath=='':
        opth = os.path.dirname(imflo)
    elif outpath!='':
        opth = outpath
    
    imio.create_dir(opth)

    #> the output naming
    if '/' in fimout:
        fout = fimout
    elif fimout!='' and not os.path.isfile(fimout):
        fout = os.path.join(opth, fimout)
    elif pickname=='ref':
        fout = os.path.join(opth, 'affine_ref-' \
                + os.path.basename(imref).split('.nii')[0]+fcomment+'.nii.gz')
    elif pickname=='flo':        
        fout = os.path.join(opth, 'affine_flo-' \
                + os.path.basename(imflo).split('.nii')[0]+fcomment+'.nii.gz')

    intrp = int(intrp)
    if isinstance(intrp, (int, long)): intrp = str(intrp)
    if intrp=='1':
        interpolation = 'trilinear'
    elif intrp=='0':
        interpolation = 'nearestneighbour'

    cmd = [ executable,
            '-in', imflo,
            '-ref', imref,
            '-out', fout,
            '-init', faff,
            '-applyxfm',
            '-interp', interpolation]
    call(cmd)

    print 'D O N E'
    print '==================================================================='

    return fout











# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# M O T I O N
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def motion_reg(
        ref,
        flo,
        fcomment = '',
        pickname = 'ref',
        outpath = '',
        fname_aff = '',
        matlab_eng='',
        rot_thresh = 1.,
        trn_thresh = 1.):

    
    if isinstance(flo, basestring):
        flolst = [flo]
    elif isinstance(flo, list) and all([os.path.isfile(f) for f in flo]):
        flolst = flo
    else:
        raise OSError('could not decode the input of floating images.')


    if matlab_eng=='':
        import matlab.engine
        eng = matlab.engine.start_matlab()
    else:
        eng = matlab_eng

    # dctout = {}
    lstout = []

    motion_rot = False
    motion_trn = False

    for i in range(len(flolst)):

        M = coreg_spm(
            ref,
            flolst[i],
            fcomment = fcomment,
            pickname = pickname,
            fname_aff = fname_aff,
            matlab_eng = eng,
            outpath = outpath,
            visual = 0,
            del_uncmpr=True)

        if any((np.abs(M['rotations'])*180/np.pi) > rot_thresh):
            print 'i> at least one rotation is above the threshold of', rot_thresh
            print '   ', M['rotations']*180/np.pi
            motion_rot = True
        if any(np.abs(M['translations']) > trn_thresh):
            print 'i> at least one translation is above the threshold of', rot_thresh
            print '   ', M['translations']
            motion_trn = True

        lstout.append({
                'regout':M, 
                'trans_mo':motion_trn,
                'rotat_mo':motion_rot,
            })

    return lstout

    #     dctout[os.path.basename(flolst[i])] = {
    #         'regout':M, 
    #         'trans_mo':motion_trn,
    #         'rotat_mo':motion_rot,
    #     }
    # return dctout






# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
# I M A G E   S I M I L A R I T Y
# <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>

def dice_coeff(im1, im2, val=1):
    ''' Calculate Dice score for parcellation images <im1> and <im2> and ROI value <val>.
        Input images can be given as:
            1. paths to NIfTI image files or as
            2. Numpy arrays.
        The ROI value can be given as:
            1. a single integer representing one ROI out of many in the parcellation
               images (<im1> and <im2>) or as
            2. a list of integers to form a composite ROI used for the association test.
        Outputs a float number representing the Dice score.
    '''

    if isinstance(im1, basestring) and isinstance(im2, basestring) \
    and os.path.isfile(im1) and os.path.basename(im1).endswith(('nii', 'nii.gz')) \
    and os.path.isfile(im2) and os.path.basename(im2).endswith(('nii', 'nii.gz')):
        imn1 = imio.getnii(im1, output='image')
        imn2 = imio.getnii(im2, output='image')
    elif isinstance(im1, (np.ndarray, np.generic)) and isinstance(im1, (np.ndarray, np.generic)):
        imn1 = im1
        imn2 = im2
    else:
        raise TypeError('Unrecognised or Mismatched Images.')

    # a single value corresponding to one ROI
    if isinstance(val, (int, long)):
        imv1 = (imn1 == val)
        imv2 = (imn2 == val)
    # multiple values in list corresponding to a composite ROI
    elif isinstance(val, list) and all([isinstance(v, (int, long)) for v in val]):
        imv1 = (imn1==val[0])
        imv2 = (imn2==val[0])
        for v in val[1:]:
            # boolean addition to form a composite ROI
            imv1 += (imn1==v)
            imv2 += (imn2==v)
    else:
        raise TypeError('ROI Values have to be integer (single or in a list).')
        

    if imv1.shape != imv2.shape:
        raise ValueError('Shape Mismatch: Input images must have the same shape.')

    #-compute Dice coefficient
    intrsctn = np.logical_and(imv1, imv2)

    return 2. * intrsctn.sum() / (imv1.sum() + imv2.sum())




#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def dice_coeff_multiclass(im1, im2, roi2ind):
    ''' Calculate Dice score for parcellation images <im1> and <im2> and ROI value <val>.
        Input images can be given as:
            1. paths to NIfTI image files or as
            2. Numpy arrays.
        The ROI value must be given as a dictionary of lists of indexes for each ROI
        Outputs a float number representing the Dice score.
    '''

    if isinstance(im1, basestring) and isinstance(im2, basestring) \
    and os.path.isfile(im1) and os.path.basename(im1).endswith(('nii', 'nii.gz')) \
    and os.path.isfile(im2) and os.path.basename(im2).endswith(('nii', 'nii.gz')):
        imn1 = imio.getnii(im1, output='image')
        imn2 = imio.getnii(im2, output='image')
    elif isinstance(im1, (np.ndarray, np.generic)) and isinstance(im1, (np.ndarray, np.generic)):
        imn1 = im1
        imn2 = im2
    else:
        raise TypeError('Unrecognised or Mismatched Images.')

    if imn1.shape != imn2.shape:
        raise ValueError('Shape Mismatch: Input images must have the same shape.')

    out = {}
    for k in roi2ind.keys():

    	# multiple values in list corresponding to a composite ROI
        imv1 = (imn1==roi2ind[k][0])
        imv2 = (imn2==roi2ind[k][0])
        for v in roi2ind[k][1:]:
            # boolean addition to form a composite ROI
            imv1 += (imn1==v)
            imv2 += (imn2==v)

	    #-compute Dice coefficient
    	intrsctn = np.logical_and(imv1, imv2)
    	out[k] = 2. * intrsctn.sum() / (imv1.sum() + imv2.sum())

    return out


