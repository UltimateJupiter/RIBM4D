/***************************************************************************
  **************************************************************************
  
  S2kit 1.0
  A lite version of Spherical Harmonic Transform Kit

  Copyright (c) 2004 Peter Kostelec, Dan Rockmore

  This file is part of S2kit.

  S2kit is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  S2kit is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
  
  See the accompanying LICENSE file for details.
  
  ************************************************************************
  ************************************************************************/

#include <math.h>
#include <string.h>  /* to declare memcpy */
#include <stdlib.h>

#ifndef PI
#define PI 3.14159265358979
#endif

#define compmult(a,b,c,d,e,f) (e) = ((a)*(c))-((b)*(d)); (f) = ((a)*(d))+((b)*(c))


/************************************************************************/
/* Recurrence coefficients */
/************************************************************************/
/* Recurrence coefficents for L2-normed associated Legendre
   recurrence.  When using these coeffs, make sure that
   inital Pmm function is also L2-normed */
/* l represents degree, m is the order */

__device__ double L2_an(int m,
	     int l)
{
  return (sqrt((((double) (2*l+3))/((double) (2*l+1))) *
	       (((double) (l-m+1))/((double) (l+m+1)))) *
	  (((double) (2*l+1))/((double) (l-m+1))));

}

/* note - if input l is zero, need to return 0 */
__device__ double L2_cn(int m,
	     int l) 
{
  if (l != 0) {
    return (-1.0 *
	  sqrt((((double) (2*l+3))/((double) (2*l-1))) *
	       (((double) (l-m+1))/((double) (l+m+1))) *
	       (((double) (l-m))/((double) (l+m)))) *
	  (((double) (l+m))/((double) (l-m+1))));
  }
  else
    return 0.0;

}

/************************************************************************/
/* vector arithmetic operations */
/************************************************************************/
/* does result = data1 + data2 */
/* result and data are vectors of length n */

__device__ void vec_add(double *data1,
	     double *data2,
	     double *result,
	     int n)
{
  int k;

  for (k = 0; k < n % 4; ++k)
    result[k] = data1[k] + data2[k];

  for ( ; k < n ; k += 4)
    {
      result[k] = data1[k] + data2[k];
      result[k + 1] = data1[k + 1] + data2[k + 1];
      result[k + 2] = data1[k + 2] + data2[k + 2];
      result[k + 3] = data1[k + 3] + data2[k + 3];
    }
}
/************************************************************************/
/************************************************************************/
/*
   vec_mul(scalar,data1,result,n) multiplies the vector 'data1' by
   'scalar' and returns in result 
*/
__device__ void vec_mul(double scalar,
	     double *data1,
	     double *result,
	     int n)
{
   int k;


   for( k = 0; k < n % 4; ++k)
     result[k] = scalar * data1[k];

   for( ; k < n; k +=4)
     {
       result[k] = scalar * data1[k];
       result[k + 1] = scalar * data1[k + 1];
       result[k + 2] = scalar * data1[k + 2];
       result[k + 3] = scalar * data1[k + 3];
     }

}
/************************************************************************/
/* point-by-point multiplication of vectors */

__device__ void vec_pt_mul(double *data1,
		double *data2,
		double *result,
		int n)
{
   int k;
  
  for(k = 0; k < n % 4; ++k)
    result[k] = data1[k] * data2[k];
  
  for( ; k < n; k +=4)
    {
      result[k] = data1[k] * data2[k];
      result[k + 1] = data1[k + 1] * data2[k + 1];
      result[k + 2] = data1[k + 2] * data2[k + 2];
      result[k + 3] = data1[k + 3] * data2[k + 3];
    }
 
}


/************************************************************************/
/* returns an array of the angular arguments of n Chebyshev nodes */
/* eval_pts points to a double array of length n */

__device__ void ArcCosEvalPts(int n,
		   double *eval_pts)
{
    int i;
    double twoN;

    twoN = (double) (2 * n);

   for (i=0; i<n; i++)
     eval_pts[i] = (( 2.0*((double)i)+1.0 ) * PI) / twoN;

}
/************************************************************************/
/* returns an array of n Chebyshev nodes */

__device__ void EvalPts( int n,
	      double *eval_pts)
{
    int i;
    double twoN;

    twoN = (double) (2*n);

   for (i=0; i<n; i++)
     eval_pts[i] = cos((( 2.0*((double)i)+1.0 ) * PI) / twoN);

}

/************************************************************************/
/* L2 normed Pmm.  Expects input to be the order m, an array of
 evaluation points arguments of length n, and a result vector of length n */
/* The norming constant can be found in Sean's PhD thesis */
/* This has been tested and stably computes Pmm functions thru bw=512 */

__device__ void Pmm_L2( int m,
	     double *eval_pts,
	     int n,
	     double *result)
{
  int i;
  double md, id, mcons;

  id = (double) 0.0;
  md = (double) m;
  mcons = sqrt(md + 0.5);

  for (i=0; i<m; i++) {
    mcons *= sqrt((md-(id/2.0))/(md-id));
    id += 1.0;
  }
  if (m != 0 )
    mcons *= pow(2.0,-md/2.0);
  if ((m % 2) != 0) mcons *= -1.0;

  for (i=0; i<n; i++) 
    result[i] = mcons * pow(sin(eval_pts[i]),md);

}

