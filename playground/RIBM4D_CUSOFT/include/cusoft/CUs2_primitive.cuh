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

#ifndef PI
#define PI 3.141592653589793
#endif

#ifndef _S2_PRIMITIVE_H
#define _S2_PRIMITIVE_H 1

__device__ double L2_an( int ,
		     int ) ;

__device__ double L2_cn( int ,
		     int ) ;

__device__ void vec_add( double * ,
		     double * ,
		     double * ,
		     int ) ;

__device__ void vec_mul( double ,
		     double * ,
		     double * ,
		     int ) ;

__device__ void vec_pt_mul( double * ,
			double * ,
			double * ,
			int ) ;

__device__ void ArcCosEvalPts( int ,
			   double * ) ;

__device__ void EvalPts( int ,
		     double * ) ;

__device__ void Pmm_L2( int ,
		    double * ,
		    int ,
		    double * ) ;
#endif /* _S2_PRIMITIVE_H */