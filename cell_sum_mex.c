#include "mex.h"
#include "matrix.h"
#include <string.h>
#include <blas.h>


// this file can be compiled with "mex -O -largeArrayDims -lblas filename.c"
// or, if using GCC and need to specify C99m "mex -O -largeArrayDims -lblas CFLAGS="\$CFLAGS -std=c99" filename.c"

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
    const mxArray *alpha; // multiplier; what fraction of y to add to x
    
    /* check for proper number of arguments */
    if(nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:cell_sum_mex:nlhs","One output required.");
    }

    if(nrhs != 3) {
        mexErrMsgIdAndTxt("MATLAB:cell_sum_mex:nrhs","Three inputs required.");
    }
    
    x = prhs[0];
    y = prhs[1];
    alpha = prhs[2];
    double *alpha_double = mxGetPr(alpha);
    const double alpha_scalar = alpha_double[0];
    
    if (mxGetM(x) != mxGetM(y)) {
        mexErrMsgIdAndTxt("MATLAB:cell_innerprod_mex:celldim","Cell dimensions must match.");
    }
    
    mxArray *out = mxCreateCellMatrix(mxGetM(x), (mwSize)1);

    double val = 0;
    const ptrdiff_t inc = 1;
    
    mwSize n = mxGetM(x);
    
    for (int i = 0; i < n; ++i) {
        mxArray *x_i = mxGetCell(x, i);
        mxArray *y_i = mxGetCell(y, i);
        const ptrdiff_t n_i = (ptrdiff_t)min(mxGetM(x_i), mxGetM(y_i));

        mxArray *out_i = mxCreateDoubleMatrix((mwSize)n_i, (mwSize)1, mxREAL);
        
        double *x_i_double = mxGetPr(x_i);
        double *y_i_double = mxGetPr(y_i);
        double *out_i_double = mxGetPr(out_i);

        memcpy(out_i_double, x_i_double, n_i * sizeof(double));
        daxpy(&n_i, &alpha_scalar, y_i_double, &inc, out_i_double, &inc);

        mxSetCell(out, i, out_i);
    }
    
    plhs[0] = out;
}
