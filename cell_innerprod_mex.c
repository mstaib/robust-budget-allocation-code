#include "mex.h"
#include "matrix.h"
#include <string.h>
#include <blas.h>


// this file can be compiled with "mex -O fw_dual_objective_custom.c"
// or, if using GCC and need to specify C99m "mex -O CFLAGS="\$CFLAGS -std=c99" fw_dual_objective_custom.c"

inline int max(int a, int b) {
    return a > b ? a : b;
}

inline int min(int a, int b) {
    return a > b ? b : a;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
	const mxArray *x; // cell array #1
    const mxArray *y; // cell array #2
    
    /* check for proper number of arguments */
    if(nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:cell_innerprod_mex:nlhs","One output required.");
    }

    if(nrhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:cell_innerprod_mex:nrhs","Two inputs required.");
    }
    
    x = prhs[0];
    y = prhs[1];
    
    if (mxGetM(x) != mxGetM(y)) {
        mexErrMsgIdAndTxt("MATLAB:cell_innerprod_mex:celldim","Cell dimensions must match.");
    }
    
    double val = 0;
    const ptrdiff_t inc = 1;
    
    mwSize n = mxGetM(x);
    
    for (int i = 0; i < n; ++i) {
        mxArray *x_i = mxGetCell(x, i);
        mxArray *y_i = mxGetCell(y, i);
        const ptrdiff_t n_i = (ptrdiff_t)min(mxGetM(x_i), mxGetM(y_i));
        
        double *x_i_double = mxGetPr(x_i);
        double *y_i_double = mxGetPr(y_i);
        
        val += ddot(&n_i, x_i_double, &inc, y_i_double, &inc);
    }
    
    plhs[0] = mxCreateDoubleScalar(val);
}
