#include "mex.h"
#include "matrix.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>


// this file can be compiled with "mex -O filename.c"
// or, if using GCC and need to specify C99m "mex -O CFLAGS="\$CFLAGS -std=c99" filename.c"

// needs to be global so we can get its sorted _indices_
double *all_rhos;


inline int max(int a, int b) {
    return a > b ? a : b;
}

int cmp(const void *a, const void *b) {
    size_t ia = *(size_t *)a;
    size_t ib = *(size_t *)b;
    
    // swapped ia and ib from a stackoverflow example
    // because we want to sort in descending order
    return all_rhos[ib] < all_rhos[ia] ? -1 : all_rhos[ib] > all_rhos[ia];
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
	const mxArray *rho; 
    
    /* check for proper number of arguments */
    if(nlhs != 2) {
        mexErrMsgIdAndTxt("MATLAB:build_increment_ordering_mex:nlhs","Two outputs required.");
    }

    if(nrhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:build_increment_ordering_mex:nrhs","One input required.");
    }
    
    rho = prhs[0];
    size_t n = max(mxGetM(rho), mxGetN(rho));

    size_t *k_vec = mxMalloc(n * sizeof(size_t));
    size_t num_total_rhos = 0;
    for (size_t i = 0; i < n; ++i) {
        mxArray *rho_i = mxGetCell(rho, i);
        k_vec[i] = mxGetM(rho_i);
        num_total_rhos += k_vec[i];
    }
    
    // vertcat
    all_rhos = mxMalloc(num_total_rhos * sizeof(double));
    size_t curr_start_inx = 0;
    for (size_t i = 0; i < n; ++i) {
        mxArray *rho_i = mxGetCell(rho, i);
        double *rho_i_double = mxGetPr(rho_i);
        memcpy(&all_rhos[curr_start_inx], rho_i_double, k_vec[i] * sizeof(double));
        
        curr_start_inx += k_vec[i];
    }

    // create array of indices so we can sort them 
    size_t *index = mxMalloc(num_total_rhos * sizeof(size_t));
    for (size_t i = 0; i < num_total_rhos; ++i) {
        index[i] = i;
    }

    qsort(index, num_total_rhos, sizeof(*index), cmp);
    mxFree(all_rhos);

    // is_orig, js_orig section
    size_t *is_orig = mxMalloc(num_total_rhos * sizeof(size_t));
    size_t *js_orig = mxMalloc(num_total_rhos * sizeof(size_t));
    size_t k_cum = 0;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k_vec[i]; ++j) {
            size_t inx = j + k_cum;
            is_orig[inx] = i + 1; //+1 b/c matlab
            js_orig[inx] = j + 1; //+1 b/c matlab
        }
        
        k_cum += k_vec[i];
    }
    mxFree(k_vec);

    // finally, intialize is and js
    mxArray *is = mxCreateDoubleMatrix((mwSize)1, (mwSize)num_total_rhos, mxREAL);
    mxArray *js = mxCreateDoubleMatrix((mwSize)1, (mwSize)num_total_rhos, mxREAL);
    
    double *is_double = mxGetPr(is);
    double *js_double = mxGetPr(js);

    for (size_t i = 0; i < num_total_rhos; ++i) {
        is_double[i] = (double)is_orig[index[i]];
        js_double[i] = (double)js_orig[index[i]];
    }
    mxFree(index);
    mxFree(is_orig);
    mxFree(js_orig);
    
    plhs[0] = is;
    plhs[1] = js;
}
