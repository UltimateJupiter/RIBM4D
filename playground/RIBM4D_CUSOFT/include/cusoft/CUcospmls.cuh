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

#ifndef _COSPMLS_H
#define _COSPMLS_H

__device__ int Power2Ceiling( int ) ;

__device__ int TableSize( int , int ) ;

__device__ int Spharmonic_TableSize( int ) ;

__device__ int Reduced_SpharmonicTableSize( int ,
					int ) ;

__device__ int Reduced_Naive_TableSize( int ,
				    int ) ;

__device__ int NewTableOffset( int ,
			   int ) ;

__device__ int TableOffset( int ,
			int ) ;

__device__ void CosPmlTableGen( int ,
			    int ,
			    double * ,
			    double * ) ;

__device__ void CosPmlTableGenLim( int , 
			       int ,
			       int ,
			       double * ,
			       double * ) ;

__device__ int RowSize( int ,
		    int ) ;

__device__ int Transpose_RowSize( int ,
			      int ,
			      int ) ;

__device__ void Transpose_CosPmlTableGen( int ,
				      int ,
				      double * ,
				      double * ) ;

__device__ double **Spharmonic_Pml_Table( int ,
				      double * ,
				      double * ) ;

__device__ double **Reduced_Spharmonic_Pml_Table( int ,
					      int ,
					      double * ,
					      double * ) ;

__device__ double **Transpose_Spharmonic_Pml_Table( double ** ,
						int ,
						double * ,
						double * ) ;

__device__ double **Reduced_Transpose_Spharmonic_Pml_Table( double ** ,
							int ,
							int ,
							double * ,
							double * ) ;

__device__ double **SemiNaive_Naive_Pml_Table( int ,
					   int ,
					   double * ,
					   double * ) ;

__device__ double **Transpose_SemiNaive_Naive_Pml_Table( double ** , 
						     int ,
						     int ,
						     double * ,
						     double * ) ;

#endif

