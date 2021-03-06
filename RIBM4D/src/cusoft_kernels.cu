#include <cusoft/cusoft_kernels.cuh>
#include <cusoft/CUcospmls.cuh>
#include <cusoft/CUFST_semi_memo.cuh>
#include <cusoft/CUso3_correlate_sym.cuh>
#include <cusoft/CUsoft_sym.cuh>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void sample_run(double *d_sigR, double *d_sigI,
    double *d_so3SigR, double *d_so3SigI,
    double *d_workspace1, double *d_workspace2,
    double *d_sigCoefR, double *d_sigCoefI,
    double *d_patCoefR, double *d_patCoefI,
    double *d_so3CoefR, double *d_so3CoefI,
    double *d_seminaive_naive_tablespace,
    double *d_cos_even,
    double **d_seminaive_naive_table,
    int bwIn, int bwOut, int degLim,
    int sig_patch_size,
    int wsp1_bsize,
    int wsp2_bsize,
    int sigpatCoef_bsize,
    int so3Coef_bsize,
    int so3Sig_bsize,
    int SNTspace_bsize,
    int SNT_bsize,
    int cos_even_bsize)
{
    FILE *fp;
    int i;
    int sig_n = bwIn * 2;
    double *sigCoefR_test, *sigCoefI_test;
    double *patCoefR_test, *patCoefI_test;

    sigCoefR_test = (double *) malloc( sizeof(double) * bwIn * 2 * bwIn * 2 ) ;
    sigCoefI_test = (double *) malloc( sizeof(double) * bwIn * 2 * bwIn * 2 ) ;
    patCoefR_test = (double *) malloc( sizeof(double) * bwIn * 2 * bwIn * 2 ) ;
    patCoefI_test = (double *) malloc( sizeof(double) * bwIn * 2 * bwIn * 2 ) ;

    fp = fopen("./soft_samples/randomS2sig_bw8.dat","r");
    for (i = 0 ; i < sig_patch_size ; i++)
    {
        fscanf(fp,"%lf", sigCoefR_test + i);
        fscanf(fp,"%lf", sigCoefI_test + i);
    }
    fclose( fp );

    /* now the pattern */
    fp = fopen("./soft_samples/randomS2sigA_bw8.dat","r");
    for (i = 0 ; i < sig_patch_size ; i++)
    {
        fscanf(fp,"%lf", patCoefR_test + i);
        fscanf(fp,"%lf", patCoefI_test + i);
    }
    fclose( fp );
    
    for ( i = 0; i < 4; i ++ )
        printf("%f\n", patCoefR_test[i]);
    
    for ( i = 0; i < 4; i ++ )
        printf("%f\n", sigCoefR_test[i]);

    cudaMemcpy( &d_sigR[1 * sig_patch_size], patCoefR_test, sizeof(double) * sig_patch_size, cudaMemcpyHostToDevice ) ;
    cudaMemcpy( &d_sigI[1 * sig_patch_size], patCoefI_test, sizeof(double) * sig_patch_size, cudaMemcpyHostToDevice ) ;
    cudaMemcpy( &d_sigR[0 * sig_patch_size], patCoefR_test, sizeof(double) * sig_patch_size, cudaMemcpyHostToDevice ) ;
    cudaMemcpy( &d_sigI[0 * sig_patch_size], patCoefI_test, sizeof(double) * sig_patch_size, cudaMemcpyHostToDevice ) ;

    printf("memcpy done\n");
    k_sample_run <<< 1, 1 >>> (d_sigR, d_sigI,
        d_so3SigR, d_so3SigI,
        d_workspace1, d_workspace2,
        d_sigCoefR, d_sigCoefI,
        d_patCoefR, d_patCoefI,
        d_so3CoefR, d_so3CoefI,
        d_seminaive_naive_tablespace,
        d_cos_even,
        d_seminaive_naive_table,
        bwIn, bwOut, degLim,
        sig_patch_size,
        wsp1_bsize,
        wsp2_bsize,
        sigpatCoef_bsize,
        so3Coef_bsize,
        so3Sig_bsize,
        SNTspace_bsize,
        SNT_bsize,
        cos_even_bsize);
}

__global__ void k_sample_run(double *d_sigR, double *d_sigI,
    double *d_so3SigR, double *d_so3SigI,
    double *d_workspace1, double *d_workspace2,
    double *d_sigCoefR, double *d_sigCoefI,
    double *d_patCoefR, double *d_patCoefI,
    double *d_so3CoefR, double *d_so3CoefI,
    double *d_seminaive_naive_tablespace,
    double *d_cos_even,
    double **d_seminaive_naive_table,
    int bwIn, int bwOut, int degLim,
    int sig_patch_size,
    int wsp1_bsize,
    int wsp2_bsize,
    int sigpatCoef_bsize,
    int so3Coef_bsize,
    int so3Sig_bsize,
    int SNTspace_bsize,
    int SNT_bsize,
    int cos_even_bsize)
{
    int pat_ind = 0;
    int sig_ind = 1;
    int pat_offset = 0;
    float4 res = soft_corr(&d_sigR[sig_ind * sig_patch_size], &d_sigI[sig_ind * sig_patch_size],
        &d_sigR[pat_ind * sig_patch_size], &d_sigI[pat_ind * sig_patch_size],
        &d_so3SigR[pat_offset * so3Sig_bsize], &d_so3SigI[pat_offset * so3Sig_bsize],
        &d_workspace1[pat_offset * wsp1_bsize], &d_workspace2[pat_offset * wsp2_bsize],
        &d_sigCoefR[pat_offset * sigpatCoef_bsize], &d_sigCoefI[pat_offset * sigpatCoef_bsize],
        &d_patCoefR[pat_offset * sigpatCoef_bsize], &d_patCoefI[pat_offset * sigpatCoef_bsize],
        &d_so3CoefR[pat_offset * so3Coef_bsize], &d_so3CoefI[pat_offset * so3Coef_bsize],
        &d_seminaive_naive_tablespace[pat_offset * SNTspace_bsize],
        &d_cos_even[pat_offset * cos_even_bsize],
        &d_seminaive_naive_table[pat_offset * SNT_bsize],
        bwIn, bwOut, degLim);
}

__device__ float4 soft_corr(double *sigR, double *sigI,
    double *patR, double *patI,
    double *so3SigR, double *so3SigI,
    double *workspace1, double *workspace2,
    double *sigCoefR, double *sigCoefI,
    double *patCoefR, double *patCoefI,
    double *so3CoefR, double *so3CoefI,
    double *seminaive_naive_tablespace,
    double *cos_even,
    double **seminaive_naive_table,
    int bwIn, int bwOut, int degLim)
{
    // printf("softcorr begin\n");
    int i;
    int tmp, ii, jj, kk;
    float maxval = 0.0;
    int maxloc = 9;
    int n = bwIn * 2; // input shape
    
    INPLACE_SemiNaive_Naive_Pml_Table(bwIn,
        bwIn,
        seminaive_naive_table,
        seminaive_naive_tablespace,
        workspace2);
    // printf("softcorr1\n");
    
    PREALLOC_FST_semi_memo(sigR, sigI,
        sigCoefR, sigCoefI,
        n, seminaive_naive_table,
        workspace2, 1, bwIn,
        cos_even);
    // printf("softcorr2\n");
        
    PREALLOC_FST_semi_memo(patR, patI,
            patCoefR, patCoefI,
            n, seminaive_naive_table,
            workspace2, 1, bwIn,
            cos_even);
    // printf("softcorr3\n");

    /* combine coefficients */
    so3CombineCoef( bwIn, bwOut, degLim,
            sigCoefR, sigCoefI,
            patCoefR, patCoefI,
            so3CoefR, so3CoefI ) ;
    // printf("softcorr4\n");

    /* now inverse so(3) */
    Inverse_SO3_Naive_sym( bwOut,
                so3CoefR, so3CoefI,
                so3SigR, so3SigI,
                workspace1, workspace2,
                1 );
    // printf("softcorr5\n");

    maxloc = 0;
    for (i = 0; i < 8*bwOut*bwOut*bwOut; i++) {
        if (so3SigR[i] >= maxval){
            maxval = so3SigR[i];
            maxloc = i ;
        }
    }

    ii = floor(maxloc / (4.0 * bwOut * bwOut));
    tmp = maxloc - (ii * 4.0 * bwOut * bwOut);
    jj = floor(tmp / (2.0 * bwOut));
    tmp = maxloc - (ii * 4 * bwOut * bwOut) - jj * (2 * bwOut);
    kk = tmp;

    // printf("ii = %d\tjj = %d\tkk = %d\n", ii, jj, kk);

    float alpha = M_PI * jj / ((float) bwOut);
    float beta = M_PI * (2 * ii + 1) / (4.0 * bwOut);
    float gamma = M_PI * kk / ((float) bwOut);

    // printf("alpha = %f  beta = %f  gamma = %f\n", alpha, beta, gamma);
    return make_float4(alpha, beta, gamma, 0.0);
}