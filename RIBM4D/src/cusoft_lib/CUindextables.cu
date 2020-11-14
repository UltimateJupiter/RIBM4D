/***************************************************************************
  **************************************************************************
    
  Spherical Harmonic Transform Kit 2.7
    
  Copyright 1997-2003  Sean Moore, Dennis Healy,
                       Dan Rockmore, Peter Kostelec
  Copyright 2004  Peter Kostelec, Dan Rockmore

  This file is part of SpharmonicKit.

  SpharmonicKit is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  (at your option) any later version.

  SpharmonicKit is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
  
  See the accompanying LICENSE file for details.

  ************************************************************************
  ************************************************************************/

/* indextables.c - source code to hard code bit reverse permutations */

#include <string.h> /* for memcpy */

__device__ __constant__ int p2[2] = {0,1};

__device__ __constant__ int p4[4] = {0,2,1,3};

__device__ __constant__ int p8[8] = {0, 4, 2, 6, 1, 5, 3, 7};

__device__ __constant__ int p16[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};

__device__ __constant__ int  p32[32] =
{0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30, 1, 
 17, 9, 25, 5, 21, 13, 29, 3, 19, 11, 27, 7, 23, 15, 31};

__device__ __constant__ int  p64[64] =
{0, 32, 16, 48, 8, 40, 24, 56, 4, 36, 20, 52, 12, 44, 28,60,2, 
 34, 18, 50, 10, 42, 26, 58, 6, 38, 22, 54, 14, 46, 30, 62, 1, 33, 17, 49, 9,41,
 25, 57, 5, 37, 21, 53, 13, 45, 29, 61, 3, 35, 19, 51, 11, 43, 27, 59, 7, 39, 
 23, 55, 15, 47, 31, 63};

__device__ __constant__ int  p128[128] =
{0, 64, 32, 96, 16, 80, 48, 112, 8, 72, 40, 104, 24, 
 88, 56, 120, 4, 68, 36, 100, 20, 84, 52, 116, 12, 76, 44, 108, 28, 92, 60, 
 124, 2, 66, 34, 98, 18, 82, 50, 114, 10, 74, 42, 106, 26, 90, 58, 122, 6, 
 70, 38, 102, 22, 86, 54, 118, 14, 78, 46, 110, 30, 94, 62, 126, 1, 65, 33, 
 97, 17, 81, 49, 113, 9, 73, 41, 105, 25, 89, 57, 121, 5, 69, 37, 101, 21, 
 85, 53, 117, 13, 77, 45, 109, 29, 93, 61, 125, 3, 67, 35, 99, 19, 83, 51, 
 115, 11, 75, 43, 107, 27, 91, 59, 123, 7, 71, 39, 103, 23, 87, 55, 119, 15, 
 79, 47, 111, 31, 95, 63, 127};

__device__ int  p256[256] =
{0, 128, 64, 192, 32, 160, 96, 224, 16, 144, 80, 208, 
 48, 176, 112, 240, 8, 136, 72, 200, 40, 168, 104, 232, 24, 152, 88, 216, 56, 
 184, 120, 248, 4, 132, 68, 196, 36, 164, 100, 228, 20, 148, 84, 212, 52, 180, 
 116, 244, 12, 140, 76, 204, 44, 172, 108, 236, 28, 156, 92, 220, 60, 188, 124,
 252, 2, 130, 66, 194, 34, 162, 98, 226, 18, 146, 82, 210, 50, 178, 114, 242, 
 10, 138, 74, 202, 42, 170, 106, 234, 26, 154, 90, 218, 58, 186, 122, 250, 6, 
 134, 70, 198, 38, 166, 102, 230, 22, 150, 86, 214, 54, 182, 118, 246, 14, 142,
 78, 206, 46, 174, 110, 238, 30, 158, 94, 222, 62, 190, 126, 254, 1, 129, 65, 
 193, 33, 161, 97, 225, 17, 145, 81, 209, 49, 177, 113, 241, 9, 137, 73, 201, 
 41, 169, 105, 233, 25, 153, 89, 217, 57, 185, 121, 249, 5, 133, 69, 197, 37, 
 165, 101, 229, 21, 149, 85, 213, 53, 181, 117, 245, 13, 141, 77, 205, 45, 173,
 109, 237, 29, 157, 93, 221, 61, 189, 125, 253, 3, 131, 67, 195, 35, 163, 99, 
 227, 19, 147, 83, 211, 51, 179, 115, 243, 11, 139, 75, 203, 43, 171, 107, 235,
 27, 155, 91, 219, 59, 187, 123, 251, 7, 135, 71, 199, 39, 167, 103, 231, 23, 
 151, 87, 215, 55, 183, 119, 247, 15, 143, 79, 207, 47, 175, 111, 239, 31, 159,
 95, 223, 63, 191, 127, 255};


/************************************************************************/
__device__ int *ptable(int n)
{
  switch (n)
    {
    case 2: return p2;
    case 4: return p4;
    case 8: return p8;
    case 16: return p16;
    case 32: return p32;
    case 64: return p64;
    case 128: return p128;
    case 256: return p256;
    default: return 0;
    }
}
/************************************************************************/
/* performs a bit-reversal on the data, a double array of size n */
/* Needs workspace of size n */

__device__ void bitreverse(double *data,
		int n,
		double *workspace)
{
  
  int i;
  int *perm;

  perm = ptable(n);
  
  for (i=0; i<n; i++)
    workspace[i] = data[perm[i]];

  memcpy(data, workspace, sizeof(double) * n);

}

