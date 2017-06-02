#include "mex.h"
#include "matrix.h"
#include <string.h>


// this file can be compiled with "mex -O fw_dual_objective_custom.c"
// or, if using GCC and need to specify C99m "mex -O CFLAGS="\$CFLAGS -std=c99" fw_dual_objective_custom.c"

inline int max(int a, int b) {
    return a > b ? a : b;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
	const mxArray *F_vals_all; // pre-computed function values
    const mxArray *is; // indexes
    const mxArray *js; // values post-increment
    const mxArray *rho_lengths; // length of each array in the rho cell array
    
    /* check for proper number of arguments */
    if(nlhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:fw_dual_objective:nlhs","Two outputs required.");
    }

    if(nrhs != 4) {
        mexErrMsgIdAndTxt("MATLAB:fw_dual_objective:nrhs","Four inputs required.");
    }
    
    F_vals_all = prhs[0];
    is = prhs[1];
    js = prhs[2];
    rho_lengths = prhs[3];
    
//     if( !mxIsDouble(step) || mxIsComplex(step) ||
//         !(mxGetM(step)==1 && mxGetN(step)==1) ) {
//         mexErrMsgIdAndTxt( "MATLAB:fw_dual_objective:inputNotRealScalarDouble",
//         "Input variable \"step\" must be a noncomplex scalar double.");
//     }
// 
//     // validate that the input cells are the same dimensions
//     size_t num_cell_dims = mxGetM(w);
//     if (num_cell_dims != mxGetM(direction) ||
//         num_cell_dims != mxGetM(weights)) {
//         mexErrMsgIdAndTxt("MATLAB:fw_dual_objective:inputNotAligned",
//         "Input cells w, direction, and weights must have the same dimensions.");
//     }

    double *rho_lengths_double = mxGetPr(rho_lengths);
    mwSize n = max(mxGetM(rho_lengths), mxGetN(rho_lengths));
    
    // initialize w to all zeros
    mxArray *w = mxCreateCellMatrix((mwSize)n, (mwSize)1);
    for (int i = 0; i < n; ++i) {
        mxArray *zeros = mxCreateDoubleMatrix((mwSize)rho_lengths_double[i], (mwSize)1, mxREAL);
        mxSetCell(w, i, zeros);
    }
    
    double *F_vals_all_double = mxGetPr(F_vals_all);
    double Fold = F_vals_all_double[0];
    double Fmin = Fold;
    double Fnew;
    
    int is_length = max(mxGetM(is), mxGetN(is));
    double *is_double = mxGetPr(is);
    double *js_double = mxGetPr(js);
    for (int i = 0; i < is_length; ++i) {
        Fnew = F_vals_all_double[i + 1];
        if (Fnew < Fmin)
            Fmin = Fnew;
        
        mxArray *w_i = mxGetCell(w, (int)is_double[i] - 1);
        double *w_i_double = mxGetPr(w_i);
        w_i_double[(int)js_double[i] - 1] = Fnew - Fold;
        
        Fold = Fnew;
    }
    
    plhs[0] = w;
    plhs[1] = mxCreateDoubleScalar(Fmin);
}
