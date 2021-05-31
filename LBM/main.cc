/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*############################################################################*/

#include "main.h"
#include "lbm.h"
#include <cuda.h>
#include <driver_types.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
//#include <chrono>
#include <sys/time.h>
#include <sys/stat.h>

/*############################################################################*/
static LBM_Grid CUDA_srcGrid, CUDA_dstGrid;
static LBM_Grid CUDA_srcGridb, CUDA_dstGridb;


/*############################################################################*/

struct pb_TimerSet timers;
int main( int nArgs, char* arg[] ) {
	MAIN_Param param;
	int t,i;

        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1*1024*1024*1024);
	pb_InitializeTimerSet(&timers);
        struct pb_Parameters* params;
        params = pb_ReadParameters(&nArgs, arg);
        

	static LBM_GridPtr TEMP_srcGrid;
	//Setup TEMP datastructures
	LBM_allocateGrid( (float**) &TEMP_srcGrid );
	MAIN_parseCommandLine( nArgs, arg, &param, params );
	MAIN_printInfo( &param );

	MAIN_initialize( &param );
/*
	for( t = 1; t <= param.nTimeSteps; t++ ) {
                pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
		CUDA_LBM_performStreamCollide( CUDA_srcGrid, CUDA_srcGridb, CUDA_dstGrid, CUDA_dstGridb );
                pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
		LBM_swapGrids( &CUDA_srcGrid, &CUDA_dstGrid );

		if( (t & 63) == 0 ) {
			printf( "timestep: %i\n", t );
#if 0
			CUDA_LBM_getDeviceGrid((float**)&CUDA_srcGrid, (float**)&TEMP_srcGrid);
			LBM_showGridStatistics( *TEMP_srcGrid );
#endif
		}
	}
*/
/*        int nt=1;
	double time;
	for( t = 1; t <= param.nTimeSteps; t++ ) {
	  nt = nt *2;
	  struct timeval stop, start;
	  for (i =0; i< 3; i++) {
            gettimeofday(&start, NULL);  
            CUDA_LBM_kernel_loop(nt, CUDA_srcGrid, CUDA_srcGridb, CUDA_dstGrid, CUDA_dstGridb );
	    gettimeofday(&stop, NULL);
            printf("nt: %d took %lu us\n", nt, (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec); 
	  }
	  //printf("%f %f %f %f\n", CUDA_srcGrid[0],  CUDA_srcGrid[0], CUDA_dstGrid[0], CUDA_dstGrid[0]);
	  printf("%p %p %p %p\n", CUDA_srcGrid,  CUDA_srcGrid, CUDA_dstGrid, CUDA_dstGrid);
        }
*/

	 struct timeval stop, start;
	gettimeofday(&start, NULL);  
       pb_SwitchToTimer(&timers, pb_TimerID_KERNEL);
            CUDA_LBM_kernel_loop(param.nTimeSteps, CUDA_srcGrid, CUDA_srcGridb, CUDA_dstGrid, CUDA_dstGridb );
	    cudaDeviceSynchronize();
	 pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
            gettimeofday(&stop, NULL);
            printf("nt: %d took %lu us\n", param.nTimeSteps, (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);
    	    MAIN_finalize( &param );

	LBM_freeGrid( (float**) &TEMP_srcGrid );

        pb_SwitchToTimer(&timers, pb_TimerID_NONE);
        pb_PrintTimerSet(&timers);
        pb_FreeParameters(params);
	return 0;
}

/*############################################################################*/

void MAIN_parseCommandLine( int nArgs, char* arg[], MAIN_Param* param, struct pb_Parameters * params ) {
	struct stat fileStat;

	if( nArgs < 2 ) {
		printf( "syntax: lbm <time steps>\n" );
		exit( 1 );
	}

	param->nTimeSteps     = atoi( arg[1] );

	if( params->inpFiles[0] != NULL ) {
		param->obstacleFilename = params->inpFiles[0];

		if( stat( param->obstacleFilename, &fileStat ) != 0 ) {
			printf( "MAIN_parseCommandLine: cannot stat obstacle file '%s'\n",
					param->obstacleFilename );
			exit( 1 );
		}
		if( fileStat.st_size != SIZE_X*SIZE_Y*SIZE_Z+(SIZE_Y+1)*SIZE_Z ) {
			printf( "MAIN_parseCommandLine:\n"
					"\tsize of file '%s' is %i bytes\n"
					"\texpected size is %i bytes\n",
					param->obstacleFilename, (int) fileStat.st_size,
					SIZE_X*SIZE_Y*SIZE_Z+(SIZE_Y+1)*SIZE_Z );
			exit( 1 );
		}
	}
	else param->obstacleFilename = NULL;

        param->resultFilename = params->outFile;
}

/*############################################################################*/

void MAIN_printInfo( const MAIN_Param* param ) {
	printf( "MAIN_printInfo:\n"
			"\tgrid size      : %i x %i x %i = %.2f * 10^6 Cells\n"
			"\tnTimeSteps     : %i\n"
			"\tresult file    : %s\n"
			"\taction         : %s\n"
			"\tsimulation type: %s\n"
			"\tobstacle file  : %s\n\n",
			SIZE_X, SIZE_Y, SIZE_Z, 1e-6*SIZE_X*SIZE_Y*SIZE_Z,
			param->nTimeSteps, param->resultFilename, 
			"store", "lid-driven cavity",
			(param->obstacleFilename == NULL) ? "<none>" :
			param->obstacleFilename );
}

/*############################################################################*/

void MAIN_initialize( const MAIN_Param* param ) {
	static LBM_Grid TEMP_srcGrid, TEMP_dstGrid;

        pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	//Setup TEMP datastructures
	LBM_allocateGrid( (float**) &TEMP_srcGrid );
	LBM_allocateGrid( (float**) &TEMP_dstGrid );
	LBM_initializeGrid( TEMP_srcGrid );
	LBM_initializeGrid( TEMP_dstGrid );

        pb_SwitchToTimer(&timers, pb_TimerID_IO);
	if( param->obstacleFilename != NULL ) {
		LBM_loadObstacleFile( TEMP_srcGrid, param->obstacleFilename );
		LBM_loadObstacleFile( TEMP_dstGrid, param->obstacleFilename );
	}

        pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	LBM_initializeSpecialCellsForLDC( TEMP_srcGrid );
	LBM_initializeSpecialCellsForLDC( TEMP_dstGrid );

        pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	//Setup DEVICE datastructures
	CUDA_LBM_allocateGrid( (float**) &CUDA_srcGrid );
	CUDA_LBM_allocateGrid( (float**) &CUDA_dstGrid );
#ifdef ALLOW_AD
	printf( "Allocating Derivative pointers\n");
	CUDA_LBM_allocateGrid( (float**) &CUDA_srcGridb );
	CUDA_LBM_allocateGrid( (float**) &CUDA_dstGridb );
#endif

	//Initialize DEVICE datastructures
	CUDA_LBM_initializeGrid( (float**)&CUDA_srcGrid, (float**)&TEMP_srcGrid );
	CUDA_LBM_initializeGrid( (float**)&CUDA_dstGrid, (float**)&TEMP_dstGrid );
#ifdef ALLOW_AD
	CUDA_LBM_initializeGrid( (float**)&CUDA_srcGridb, (float**)&TEMP_srcGrid );
	CUDA_LBM_initializeGrid( (float**)&CUDA_dstGridb, (float**)&TEMP_dstGrid );
#endif

        pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	LBM_showGridStatistics( TEMP_srcGrid );

	LBM_freeGrid( (float**) &TEMP_srcGrid );
	LBM_freeGrid( (float**) &TEMP_dstGrid );
}

/*############################################################################*/

void MAIN_finalize( const MAIN_Param* param ) {
	LBM_Grid TEMP_srcGrid;

	//Setup TEMP datastructures
	LBM_allocateGrid( (float**) &TEMP_srcGrid );

        pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	CUDA_LBM_getDeviceGrid((float**)&CUDA_srcGrid, (float**)&TEMP_srcGrid);

        pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	LBM_showGridStatistics( TEMP_srcGrid );

	LBM_storeVelocityField( TEMP_srcGrid, param->resultFilename, TRUE );

	LBM_freeGrid( (float**) &TEMP_srcGrid );
	CUDA_LBM_freeGrid( (float**) &CUDA_srcGrid );
	CUDA_LBM_freeGrid( (float**) &CUDA_dstGrid );
}

