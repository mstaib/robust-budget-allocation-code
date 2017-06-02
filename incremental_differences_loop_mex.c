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
	const mxArray *curr_x; 
    const mxArray *is;
    const mxArray *js;
    const mxArray *val_init;
    const mxArray *log_prods;
    const mxArray *mult;
    const mxArray *x_lower_vec;
    const mxArray *y_mat;
    const mxArray *S_vec;
    const mxArray *T_vec;
    
    double *outMatrix; // output matrix

    /* check for proper number of arguments */
    if(nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:fw_dual_objective:nlhs","One output required.");
    }

    if(nrhs != 10) {
        mexErrMsgIdAndTxt("MATLAB:fw_dual_objective:nrhs","Ten inputs required.");
    }
    
    curr_x = prhs[0];
    is = prhs[1];
    js = prhs[2];
    val_init = prhs[3];
    log_prods = prhs[4];
    mult = prhs[5];
    x_lower_vec = prhs[6];
    y_mat = prhs[7];
    S_vec = prhs[8];
    T_vec = prhs[9];
    
    double *curr_x_double = mxGetPr(curr_x);
    
    mxArray *curr_x_copy = mxCreateDoubleMatrix(mxGetM(curr_x), mxGetN(curr_x), mxREAL);
    double *curr_x_copy_double = mxGetPr(curr_x_copy);
    memcpy(curr_x_copy_double, curr_x_double, sizeof(double) * mxGetM(curr_x) * mxGetN(curr_x));
    
    double *is_double = mxGetPr(is);
    double *js_double = mxGetPr(js);
    double *val_init_double = mxGetPr(val_init);
    double *log_prods_double = mxGetPr(log_prods);
    double *mult_double = mxGetPr(mult);
    double *x_lower_vec_double = mxGetPr(x_lower_vec);
    double *y_mat_double = mxGetPr(y_mat);
    double *S_vec_double = mxGetPr(S_vec);
    double *T_vec_double = mxGetPr(T_vec);
    
    //////
    ptrdiff_t n = (ptrdiff_t)max(mxGetM(is), mxGetN(is));
    
    val = mxCreateDoubleMatrix((mwSize)(n + 1), (mwSize)1, mxREAL);
    double *val_double = mxGetPr(val);
    val_double[0] = val_init_double[0];
    
    for (int k = 0; k < n; ++k) {
        int i = (int)is_double[k] - 1;
        int j = (int)js_double[k];
        
        double old_val = curr_x_copy_double[i];
        double new_val = mult_double[i] * j + x_lower_vec_double[i];
        curr_x_copy_double[i] = new_val;
        
        int S_affected = (int)S_vec_double[i] - 1;
        int T_affected = (int)T_vec_double[i] - 1;
        
        double log_diff = y_mat_double[S_affected] * (log(new_val) - log(old_val));
        double log_prods_T_new = log_prods_double[T_affected] + log_diff;
        double exp_diff = exp(log_prods_T_new) - exp(log_prods_double[T_affected]);
        
        log_prods_double[T_affected] = log_prods_T_new;
        val_double[k + 1] = val_double[k] - exp_diff;
    }
    
    
    /*
    if( !mxIsDouble(step) || mxIsComplex(step) ||
        !(mxGetM(step)==1 && mxGetN(step)==1) ) {
        mexErrMsgIdAndTxt( "MATLAB:fw_dual_objective:inputNotRealScalarDouble",
        "Input variable \"step\" must be a noncomplex scalar double.");
    }
    */

    /*
    // validate that the input cells are the same dimensions
    size_t num_cell_dims = mxGetM(w);
    if (num_cell_dims != mxGetM(direction) ||
        num_cell_dims != mxGetM(weights)) {
        mexErrMsgIdAndTxt("MATLAB:fw_dual_objective:inputNotAligned",
        "Input cells w, direction, and weights must have the same dimensions.");
    }
     */    
    
    plhs[0] = val;
}
