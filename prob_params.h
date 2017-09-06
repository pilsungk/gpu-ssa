#ifndef PROB_PARAMS_H
#define PROB_PARAMS_H

#define XBLOCKSIZE 128       // cuda thread block x
#define YBLOCKSIZE 1       // cuda thread block y
#define XGRIDSIZE  1       // cuda grid size x
#define YGRIDSIZE  1        // cuda grid size y
#define NTHREADS   ((XBLOCKSIZE)*(YBLOCKSIZE)*(XGRIDSIZE)*(YGRIDSIZE))

// Input Problem Constants
#define NX 3                // number of spicies
#define FINALTIME 1000.0    // time when the evolution finishes 

#define NCHANNEL 3
#define DIMX_NU NCHANNEL
#define DIMY_NU NX

#endif 
