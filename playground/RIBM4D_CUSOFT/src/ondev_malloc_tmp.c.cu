float* ref_patch (new double[patch_bsize]);
double* sigR (new double[sigpatSig_bsize]);
double* sigI (new double[sigpatSig_bsize]);

// pre-allocated workspace for cmp patch and fftshift of it
float* cmp_patch (new double[patch_bsize]);
double* patR (new double[sigpatSig_bsize]);
double* patI (new double[sigpatSig_bsize]);

// pre-allocated workspace for cusoft
double* so3SigR (new double[so3Sig_bsize]);
double* so3SigI (new double[so3Sig_bsize]);
double* workspace1 (new double[wsp1_bsize]);
double* workspace2 (new double[wsp2_bsize]);
double* sigCoefR (new double[sigpatCoef_bsize]);
double* sigCoefI (new double[sigpatCoef_bsize]);
double* patCoefR (new double[sigpatCoef_bsize]);
double* patCoefI (new double[sigpatCoef_bsize]);
double* so3CoefR (new double[so3Coef_bsize]);
double* so3CoefI (new double[so3Coef_bsize]);
double* seminaive_naive_tablespace (new double[SNTspace_bsize]);
double* cos_even (new double[cos_even_bsize]);



delete ref_patch;
delete sigR;
delete sigI;

// pre-allocated workspace for cmp patch and fftshift of it
delete cmp_patch;
delete patR;
delete patI;

// pre-allocated workspace for cusoft
delete so3SigR;
delete so3SigI;
delete workspace1;
delete workspace2;
delete sigCoefR;
delete sigCoefI;
delete patCoefR;
delete patCoefI;
delete so3CoefR;
delete so3CoefI;
delete seminaive_naive_tablespace;
delete cos_even;


float* ref_patch (new double[patch_bsize]);
double* sigR (new double[sigpatSig_bsize]);
double* sigI (new double[sigpatSig_bsize]);

// pre-allocated workspace for cmp patch and fftshift of it
float* ref_patch = (float*)malloc(sizeof(float) * patch_bsize);
double* sigR = (double*)malloc(sizeof(double) * sigpatSig_bsize);
double* sigI = (double*)malloc(sizeof(double) * sigpatSig_bsize);

float* cmp_patch = (float*)malloc(sizeof(float) * patch_bsize);
double* patR = (double*)malloc(sizeof(double) * sigpatSig_bsize);
double* patI = (double*)malloc(sizeof(double) * sigpatSig_bsize);

// pre-allocated workspace for cusoft
double* so3SigR = (double*)malloc(sizeof(double) * so3Sig_bsize);
double* so3SigI = (double*)malloc(sizeof(double) * so3Sig_bsize);
double* workspace1 = (double*)malloc(sizeof(double) * wsp1_bsize);
double* workspace2 = (double*)malloc(sizeof(double) * wsp2_bsize);
double* sigCoefR = (double*)malloc(sizeof(double) * sigpatCoef_bsize);
double* sigCoefI = (double*)malloc(sizeof(double) * sigpatCoef_bsize);
double* patCoefR = (double*)malloc(sizeof(double) * sigpatCoef_bsize);
double* patCoefI = (double*)malloc(sizeof(double) * sigpatCoef_bsize);
double* so3CoefR = (double*)malloc(sizeof(double) * so3Coef_bsize);
double* so3CoefI = (double*)malloc(sizeof(double) * so3Coef_bsize);
double* seminaive_naive_tablespace (double*)malloc(sizeof(double) * SNTspace_bsize);
double* cos_even = (double*)malloc(sizeof(double) * cos_even_bsize);

free(ref_patch);
free(sigR);
free(sigI);

// pre-allocated workspace for cmp patch and fftshift of it
free(cmp_patch);
free(patR);
free(patI);

// pre-allocated workspace for cusoft
free(so3SigR);
free(so3SigI);
free(workspace1);
free(workspace2);
free(sigCoefR);
free(sigCoefI);
free(patCoefR);
free(patCoefI);
free(so3CoefR);
free(so3CoefI);
free(seminaive_naive_tablespace);
free(cos_even);