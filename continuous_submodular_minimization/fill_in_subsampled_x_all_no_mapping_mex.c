#include "mex.h"
#include "matrix.h"
#include <string.h>
#include <math.h>


// this file can be compiled with "mex -O filename.c"
// or, if using GCC and need to specify C99m "mex -O CFLAGS="\$CFLAGS -std=c99" filename.c"

inline int max(int a, int b) {
    return a > b ? a : b;
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
	mxArray *val;	// the array of function values
    const mxArray *is;
    const mxArray *js;
    const mxArray *n;
    const mxArray *num_subsampled_vecs;
    const mxArray *start_inx;
    const mxArray *end_inx;
    const mxArray *skip;

    
    double *outMatrix; // output matrix

    /* check for proper number of arguments */
    if(nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:fill_in_subsampled_x_all_no_mapping:nlhs","One output required.");
    }

    if(nrhs != 7) {
        mexErrMsgIdAndTxt("MATLAB:fill_in_subsampled_x_all_no_mapping:nrhs","Seven inputs required.");
    }
    
    is = prhs[0];
    js = prhs[1];
    n = prhs[2];
    num_subsampled_vecs = prhs[3];
    start_inx = prhs[4];
    end_inx = prhs[5];
    skip = prhs[6];

    double *is_double = mxGetPr(is);
    double *js_double = mxGetPr(js);
    
    double *n_double = mxGetPr(n);
    int n_int = (int)n_double[0];

    double *num_subsampled_vecs_double = mxGetPr(num_subsampled_vecs);
    int num_subsampled_vecs_int = (int)num_subsampled_vecs_double[0];

    double *start_inx_double = mxGetPr(start_inx);
    int start_inx_int = (int)start_inx_double[0];

    double *end_inx_double = mxGetPr(end_inx);
    int end_inx_int = (int)end_inx_double[0];

    double *skip_double = mxGetPr(skip);
    int skip_int = (int)skip_double[0];

    // now define variables for this function
    int num_total_rhos = max((int)mxGetM(is), (int)mxGetN(is));

    double *xcurr = mxCalloc(n_int, sizeof(double));

    mxArray *x_all = mxCreateDoubleMatrix(n_int, num_subsampled_vecs_int, mxREAL);
    double *x_all_double = mxGetPr(x_all);

    int k = 0;
    for (int i = 0; i < num_total_rhos; ++i) {
        xcurr[(int)is_double[i] - 1] = js_double[i];

        if ((i + 1) == end_inx_int) {
            int col_to_update = (num_subsampled_vecs_int - 1);
            for (int j = 0; j < n_int; ++j) {
                x_all_double[col_to_update * n_int + j] = xcurr[j];
            }
            break;
        }

        if ((i + 1) == start_inx_int + k * skip_int) {
            int col_to_update = (k - 1) + 1;
            for (int j = 0; j < n_int; ++j) {
                x_all_double[col_to_update * n_int + j] = xcurr[j];
            }
            k += 1;
        }
    }

    mxFree(xcurr);

    plhs[0] = x_all;
}
