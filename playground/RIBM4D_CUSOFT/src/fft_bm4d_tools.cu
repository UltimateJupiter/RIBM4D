#include <bm4d-gpu/fft_bm4d_tools.cuh>

// int fft_bm4d_tools::N = 4;
// int fft_bm4d_tools::xm = 2;
// int fft_bm4d_tools::ym = 2;
// int fft_bm4d_tools::zm = 2;
// int fft_bm4d_tools::B = 8;

// __device__ void init_patch_n(int N_) {
//     N = N_;
//     xm = N_ / 2;
//     ym = N_ / 2;
//     zm = N_ / 2;
//     return;
// }

// __device__ void init_bandwidth(int B_) {
//     B = B_;
// }

__device__ __constant__ int fftshift[64] = {42, 43, 40, 41, 46, 47, 44, 45, 34, 35, 32, 33, 38, 39, 36, 37, 58, 59, 56, 57, 62, 63, 60, 61, 50, 51, 48, 49, 54, 55, 52, 53, 10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5, 26, 27, 24, 25, 30, 31, 28, 29, 18, 19, 16, 17, 22, 23, 20, 21};

//In-place 3-d fftshift
__device__ void fft_bm4d_tools::fftshift_3d_in(float* data_re, float* data_im) {
    float tmp_re, tmp_im;

    for (int i = 0; i < N * N * N / 2; i++) {
        tmp_re = data_re[i];
        tmp_im = data_im[i];
        data_re[i] = data_re[fftshift[i]];
        data_re[i] = data_re[fftshift[i]];
        data_re[fftshift[i]] = tmp_re;
        data_im[fftshift[i]] = tmp_im;
     }
     return;

    // int target_x, target_y, target_z;
    // float temp_re, temp_im;
    // //(*NEED TO ADD*) cudaMalloc needed?

    // for (int x = 0; x < N; x++)
    //     for (int y = 0; y < N; y++)
    //         for (int z = 0; z < N; z++) {
    //             temp_re = data_re[x * N * N + y * N + z];
    //             temp_im = data_im[x * N * N + y * N + z];

    //             if (x < xm) {
    //                 if (y < ym) {
    //                     if (z < zm) {
    //                         // Section 1
    //                         target_x = x + xm;
    //                         target_y = y + ym;
    //                         target_z = z + zm;

    //                         // Equivalent to:
    //                         // data[x][y][z] = data[target_x][target_y][target_z];
    //                         // data[target_x][target_y][target_z] = temp;
    //                         data_re[x * N * N + y * N + z] = 
    //                             data_re[target_x * N * N + target_y * N + target_z];
    //                         data_re[target_x * N * N + target_y * N + target_z] = temp_re;
    //                         data_im[x * N * N + y * N + z] = 
    //                             data_im[target_x * N * N + target_y * N + target_z];
    //                         data_im[target_x * N * N + target_y * N + target_z] = temp_im;
    //                     }
    //                     else {
    //                         //section 2
    //                         target_x = x + xm;
    //                         target_y = y + ym;
    //                         target_z = z - zm;
    //                         data_re[x * N * N + y * N + z] = 
    //                             data_re[target_x * N * N + target_y * N + target_z];
    //                         data_re[target_x * N * N + target_y * N + target_z] = temp_re;
    //                         data_im[x * N * N + y * N + z] = 
    //                             data_im[target_x * N * N + target_y * N + target_z];
    //                         data_im[target_x * N * N + target_y * N + target_z] = temp_im;
    //                     }
    //                 }
    //                 else {
    //                     if (z < zm) {
    //                         // Section 3
    //                         target_x = x + xm;
    //                         target_y = y + ym;
    //                         target_z = z + zm;
    //                         data_re[x * N * N + y * N + z] = 
    //                             data_re[target_x * N * N + target_y * N + target_z];
    //                         data_re[target_x * N * N + target_y * N + target_z] = temp_re;
    //                         data_im[x * N * N + y * N + z] = 
    //                             data_im[target_x * N * N + target_y * N + target_z];
    //                         data_im[target_x * N * N + target_y * N + target_z] = temp_im;
    //                     }
    //                     else {
    //                         //section 4
    //                         target_x = x + xm;
    //                         target_y = y - ym;
    //                         target_z = z - zm;
    //                         data_re[x * N * N + y * N + z] = 
    //                             data_re[target_x * N * N + target_y * N + target_z];
    //                         data_re[target_x * N * N + target_y * N + target_z] = temp_re;
    //                         data_im[x * N * N + y * N + z] = 
    //                             data_im[target_x * N * N + target_y * N + target_z];
    //                         data_im[target_x * N * N + target_y * N + target_z] = temp_im;
    //                     }
    //                 }
    //             }
    //         }

    // //(*NEED TO ADD*) Free temp if cudaMalloc is required above
    // return;
}

//In-place 3-d ifftshift
__device__ void fft_bm4d_tools::ifftshift_3d_in(float* data_re, float* data_im) {
    if (N % 2 != 0 || N % 2 != 0 || N % 2 != 0)
        printf("Error: UNIMPLEMENTED\n");
    fftshift_3d_in(data_re, data_im);
}

// __device__ void stockham(float x_re[], float x_im[], int n, int flag, int n2, float y_re[], float y_im[])
__device__ void stockham(float x_re[], float x_im[], int n, int flag, int n2, float y_re[], float y_im[])
{
    float  *y_orig_re, *y_orig_im, *tmp_re, *tmp_im;
    int  i, j, k, k2, Ls, r, jrs;
    int  half, m, m2;
    float  wr, wi, tr, ti;

    y_orig_re = y_re;
    y_orig_im = y_im;
    r = half = n >> 1;
    Ls = 1;                                         /* Ls=L/2 is the L star */


    while(r >= n2) {                              /* loops log2(n/n2) times */
        tmp_re = x_re;                           /* swap pointers, y is always old */
        tmp_im = x_im;
        x_re = y_re;                                   /* x is always for new data */
        x_im = y_im;
        y_re = tmp_re;
        y_im = tmp_im;
        m = 0;                        /* m runs over first half of the array */
        m2 = half;                             /* m2 for second half, n2=n/2 */
        for(j = 0; j < Ls; ++j) {
            wr = cos(M_PI*j/Ls);                   /* real and imaginary part */
            wi = -flag * sin(M_PI*j/Ls);                      /* of the omega */
            jrs = j*(r+r);
            for(k = jrs; k < jrs+r; ++k) {           /* "butterfly" operation */
            k2 = k + r;
            tr =  wr*y_re[k2] - wi*y_im[k2];      /* complex multiply, w*y */
            ti =  wr*y_im[k2] + wi*y_re[k2];
            x_re[m] = y_re[k] + tr;
            x_im[m] = y_im[k] + ti;
            x_re[m2] = y_re[k] - tr;
            x_im[m2] = y_im[k] - ti;
            ++m;
            ++m2;
            }
        }
        r  >>= 1;
        Ls <<= 1;
    };


    if (y_re != y_orig_re) {                     /* copy back to permanent memory */
        for(i = 0; i < n; ++i) {               /* if it is not already there */
            y_re[i] = x_re[i];               /* performed only if log2(n/n2) is odd */
        }
    }

    if (y_im != y_orig_im) {                     /* copy back to permanent memory */
        for(i = 0; i < n; ++i) {               /* if it is not already there */
            y_im[i] = x_im[i];               /* performed only if log2(n/n2) is odd */
        }
    }


    //assert(Ls == n/n2);                        /* ensure n is a power of 2  */
    //assert(1 == n || m2 == n);           /* check array index within bound  */
}




/* The Cooley-Tukey multiple column algorithm, see page 124 of Loan.
   x[] is input data, overwritten by output, viewed as n/n2 by n2
   array. flag = 1 for forward and -1 for backward transform.
*/
__device__ void cooley_tukey(float x_re[], float x_im[], int n, int flag, int n2)
{
    float c_re, c_im;
    int i, j, k, m, p, n1;
    int Ls, ks, ms, jm, dk;
    float wr, wi, tr, ti;


    n1 = n/n2;                               /* do bit reversal permutation */
    for(k = 0; k < n1; ++k) {        /* This is algorithms 1.5.1 and 1.5.2. */
        j = 0;
        m = k;
        p = 1;                               /* p = 2^q,  q used in the book */
        while(p < n1) {
            j = 2*j + (m&1);
            m >>= 1;
            p <<= 1;
        }
        //assert(p == n1);                   /* make sure n1 is a power of two */
        if(j > k) {
            for(i = 0; i < n2; ++i) {                     /* swap k <-> j row */
                // c = x[k*n2+i];                              /* for all columns */
                // x[k*n2+i] = x[j*n2+i];
                // x[j*n2+i] = c;
                c_re = x_re[k * n2 + i];
                c_im = x_im[k * n2 + i];
                x_re[k * n2 + i] = x_re[j * n2 + i];
                x_im[k * n2 + i] = x_im[j * n2 + i];
                x_re[j * n2 + i] = c_re;
                x_im[j * n2 + i] = c_im;
            }
        }
    }
                                                /* This is (3.1.7), page 124 */
    p = 1;
    while(p < n1) {
        Ls = p;
        p <<= 1;
        jm = 0;                                                /* jm is j*n2 */
        dk = p*n2;
        for(j = 0; j < Ls; ++j) {
            wr = cos(M_PI*j/Ls);                   /* real and imaginary part */
            wi = -flag * sin(M_PI*j/Ls);                      /* of the omega */
            for(k = jm; k < n; k += dk) {                      /* "butterfly" */
                ks = k + Ls*n2;
                for(i = 0; i < n2; ++i) {                      /* for each row */
                    m = k + i;
                    ms = ks + i;
                    // tr =  wr*x[ms].Re - wi*x[ms].Im;
                    // ti =  wr*x[ms].Im + wi*x[ms].Re;
                    // x[ms].Re = x[m].Re - tr;
                    // x[ms].Im = x[m].Im - ti;
                    // x[m].Re += tr;
                    // x[m].Im += ti;
                    tr = wr * x_re[ms] - wi * x_im[ms];
                    ti = wr * x_im[ms] + wi * x_re[ms];
                    x_re[ms] = x_re[m] - tr;
                    x_im[ms] = x_im[m] - ti;
                    x_re[m] += tr;
                    x_im[m] += ti;
                }
            }
            jm += n2;
        }
    }
}

__device__ void clear_buffer(float* buf, int size) {
    for (int i = 0 ; i < size; i++) {
        buf[i] = 0.0;
    }
}

__device__ void fft3D_helper(float x_re[], float x_im[], float y_re[], float y_im[], int n1, int n2, int n3, int flag)
{
    // float *y_re, *y_im;
    int i, n, n23;

    //assert(1 == flag || -1 == flag);
    n23 = n2*n3;
    n = n1*n23;
    // y_re = (float*) malloc( n23*sizeof(float) );
    // y_im = (float*) malloc( n23*sizeof(float) );
    //assert(NULL != y_re);
    //assert(NULL != y_im);


    for(i=0; i < n; i += n3) {                                  /* FFT in z */
        stockham(x_re+i, x_im+i, n3, flag, 1, y_re, y_im);
    }
    for(i=0; i < n; i += n23) {                                 /* FFT in y */
        stockham(x_re+i, x_im+i, n23, flag, n3, y_re, y_im);
    }
    clear_buffer(y_re, n23);
    clear_buffer(y_im, n23);
    cooley_tukey(x_re, x_im, n, flag, n23);                              /* FFT in x */
}

//Apply 3-d fft on cufftComplex data in place
//buf_re and buf_im are used to buffer data in stockam: each of size n23 = N^2
__device__ void fft_bm4d_tools::fft_3d(float* data_re, float* data_im, float* buf_re, float* buf_im) {
    fft3D_helper(data_re, data_im, buf_re, buf_im, N, N, N, FORWARD);
}

//Calculate the absolute value of complex array
//Data_im will not be needed after this function
__device__ void fft_bm4d_tools::complex_abs(float* data_re, float* data_im) {
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
         for (int k = 0; k < N; k++) {
            data_re[N * N * i + N * j + k] = sqrt(
               pow(data_re[N * N * i + N * j + k], 2) + 
               pow(data_im[N * N * i + N * j + k], 2)
               );
            data_im[N * N * i + N * j + k] = 0.0;
         }
}

// calculate x from theta, phi
__device__ float x_spherical(float theta, float phi) {
   return cos(phi) * cos(theta);
}

// calculate y from theta, phi
__device__ float y_spherical(float theta, float phi) {
   return sin(phi) * sin(phi);
}

// calculate z from theta, phi
__device__ float z_spherical(float theta) {
   return cos(theta);
}

// calculate the trilinear interpolation at coordinate x, y, z
// the complex part of data should all be zero
__device__ float trilinear_interpolation(float* data_re, float x, float y, float z, int N) {
   int x0 = int(x);
   int y0 = int(y);
   int z0 = int(z);

   float xd = x - ceil(x);
   float yd = y - ceil(y);
   float zd = z - ceil(z);

   int x1 = int(ceil(x));
   int y1 = int(ceil(y));
   int z1 = int(ceil(z));

   // c000, c001
   float c00 = data_re[x0 * N * N + y0 * N + z0] * (1 - xd) +
               data_re[x1 * N * N + y0 * N + z0] * xd;

   // c001, c101
   float c01 = data_re[x0 * N * N + y0 * N + z1] * (1 - xd) +
               data_re[x1 * N * N + y0 * N + z1] * xd;

   // c010, c110
   float c10 = data_re[x0 * N * N + y1 * N + z0] * (1 - xd) +
               data_re[x1 * N * N + y1 * N + z0] * xd;

   // c011, c111
   float c11 = data_re[x0 * N * N + y1 * N + z1] * (1 - xd) +
               data_re[x1 * N * N + y1 * N + z1] * xd;

   float c0 = c00 * (1 - yd) + c10 * yd;
   float c1 = c01 * (1 - yd) + c11 * yd;

   // result of trilinear interpolation
   float c = c0 * (1 - zd) + c1 * zd;

   return c;
}

//Map volume data to spherical coordinates and do the integration
__device__ void fft_bm4d_tools::spherical_mapping(double* maga_re, double* maga_im, float* data_re) {
    float theta, phi;
    float x, y, z;
    float sum;
    int maga_idx;
    float interpol_val;

    for (int theta_idx = 0; theta_idx < 2 * B; theta_idx++) {
        for (int phi_idx = 0; phi_idx < 2 * B; phi_idx++) {
            theta = M_PI * (2 * theta_idx + 1) / (4 * B);
            phi = M_PI * phi_idx / B;

            maga_idx = theta_idx * 2 * B + phi_idx;

            x = x_spherical(theta, phi);
            y = y_spherical(theta, phi);
            z = z_spherical(theta);

            sum = 0;

            for (float rho = 0.5; rho < 2; rho+=1.0) {
            interpol_val = trilinear_interpolation(data_re,
                                                    x * rho + xm,
                                                    y * rho + ym,
                                                    z * rho + zm,
                                                    N
                                                    );
            sum += fabs(interpol_val);
            }

            maga_re[maga_idx] = sum;
            maga_im[maga_idx] = 0.0;
        }
    }
}