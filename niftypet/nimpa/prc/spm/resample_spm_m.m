function out = resample_spm_m(imref, imflo, M, f_mask, f_mean, f_interp, f_which, f_prefix)

    %-Reslicing parameters
    rflags.mask = f_mask;
    rflags.mean = f_mean;
    rflags.interp = f_interp;
    rflags.which = f_which;
    rflags.wrap = [0 0 0];
    rflags.prefix = f_prefix;

    VG = strcat(imref,',1');
    VF = strcat(imflo,',1');

    disp('Matlab internal reference image:');
    disp(imref);

    disp('Matlab internal floating image:');
    disp(imflo);

    disp(rflags)

    MM = zeros(4,4);
    MM(:,:) = spm_get_space(VF);
    spm_get_space(VF, M\MM(:,:));
    P = {VG; VF};
    spm_reslice(P,rflags);

    out = 0;
end