#include <cusoft/cusoft_kernels.cuh>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void sample_run(double *fftCoefR, double *fftCoefI,
                double *so3CoefR, double *so3CoefI,
                double *rdata, double *idata,
                double *workspace1,
                double *workspace2,
                double *d_cos_even,
                double **d_seminaive_naive_table,
                int bw, int degLim,
                int wsp1_bsize, int wsp2_bsize,
                int ridata_bsize,
                int so3Coef_bsize,
                int SNT_bsize,
                int fft_patch_size)
{
    FILE *fp ;
    int i ;
    double *sigCoefR_test, *sigCoefI_test ;
    double *patCoefR_test, *patCoefI_test ;

    sigCoefR_test = (double *) malloc( sizeof(double) * bw * bw ) ;
    sigCoefI_test = (double *) malloc( sizeof(double) * bw * bw ) ;
    patCoefR_test = (double *) malloc( sizeof(double) * bw * bw ) ;
    patCoefI_test = (double *) malloc( sizeof(double) * bw * bw ) ;

    fp = fopen("./soft_samples/randomS2sigB_bw8.dat","r");
    for ( i = 0 ; i < bw*bw ; i ++ )
    {
        fscanf(fp,"%lf", sigCoefR_test + i);
        fscanf(fp,"%lf", sigCoefI_test + i);
    }
    fclose( fp );

    /* now the pattern */
    fp = fopen("./soft_samples/randomS2sig_bw8.dat","r");
    for ( i = 0 ; i < bw*bw ; i ++ )
    {
        fscanf(fp,"%lf", patCoefR_test + i );
        fscanf(fp,"%lf", patCoefI_test + i );
    }
    fclose( fp );
    
    for ( i = 0; i < 4; i ++ )
        printf("%f\n", patCoefR_test[i]);
    
    for ( i = 0; i < 4; i ++ )
        printf("%f\n", sigCoefR_test[i]);

    cudaMemcpy( &fftCoefR[1 * fft_patch_size], sigCoefR_test, sizeof(double) * bw * bw, cudaMemcpyHostToDevice ) ;
    cudaMemcpy( &fftCoefI[1 * fft_patch_size], sigCoefI_test, sizeof(double) * bw * bw, cudaMemcpyHostToDevice ) ;
    cudaMemcpy( &fftCoefR[0 * fft_patch_size], patCoefR_test, sizeof(double) * bw * bw, cudaMemcpyHostToDevice ) ;
    cudaMemcpy( &fftCoefI[0 * fft_patch_size], patCoefI_test, sizeof(double) * bw * bw, cudaMemcpyHostToDevice ) ;

    printf("memcpy done\n");
    k_sample_run <<< 1, 1 >>> (fftCoefR, fftCoefI,
                               so3CoefR, so3CoefI,
                               rdata, idata,
                               workspace1,
                               workspace2,
                               d_cos_even,
                               d_seminaive_naive_table,
                               bw, degLim,
                               wsp1_bsize, wsp2_bsize,
                               ridata_bsize,
                               so3Coef_bsize,
                               SNT_bsize,
                               fft_patch_size);
}

__global__ void k_sample_run(double *d_fftCoefR, double *d_fftCoefI,
                             double *d_so3CoefR, double *d_so3CoefI,
                             double *d_rdata, double *d_idata,
                             double *d_workspace1,
                             double *d_workspace2,
                             double *d_cos_even,
                             double **d_seminaive_naive_table,
                             int bw, int degLim,
                             int wsp1_bsize, int wsp2_bsize,
                             int ridata_bsize,
                             int so3Coef_bsize,
                             int SNT_bsize,
                             int fft_patch_size)
{
    int pat_ind = 0;
    int sig_ind = 1;
    int pat_offset = 0;
    float4 res = soft_corr(&d_fftCoefR[sig_ind * fft_patch_size], &d_fftCoefI[sig_ind * fft_patch_size],
                           &d_fftCoefR[pat_ind * fft_patch_size], &d_fftCoefI[pat_ind * fft_patch_size],
                           &d_so3CoefR[pat_offset * so3Coef_bsize], &d_so3CoefI[pat_offset * so3Coef_bsize],
                           &d_rdata[pat_offset * ridata_bsize], &d_idata[pat_offset * ridata_bsize],
                           &d_workspace1[pat_offset * wsp1_bsize],
                           &d_workspace2[pat_offset * wsp2_bsize],
                           &d_cos_even[pat_offset * bw],
                           &d_seminaive_naive_table[pat_offset * SNT_bsize],
                           bw, degLim);
}

__device__ float4 soft_corr(double *sigCoefR, double *sigCoefI,
                            double *patCoefR, double *patCoefI,
                            double *so3CoefR, double *so3CoefI,
                            double *rdata, double *idata,
                            double *workspace1,
                            double *workspace2,
                            double *cos_even,
                            double **seminaive_naive_table,
                            int bw, int degLim)
{
    printf("output from device function\n");
    for (int i = 0; i < 4; i++)
        printf("%f\n", patCoefR[i]);

    for (int i = 0; i < 4; i++)
        printf("%f\n", sigCoefR[i]);

    int tmp, maxloc, ii, jj, kk ;
    float4 ret;
    return ret;
}