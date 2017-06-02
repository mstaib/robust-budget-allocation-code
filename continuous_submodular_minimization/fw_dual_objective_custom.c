#include "mex.h"
#include "matrix.h"
#include <string.h>
#include <blas.h>


// this file can be compiled with "mex -O -largeArrayDims -lblas fw_dual_objective_custom.c"
// or, if using GCC and need to specify C99m "mex -O -largeArrayDims -lblas CFLAGS="\$CFLAGS -std=c99" fw_dual_objective_custom.c"

inline int max(int a, int b) {
    return a > b ? a : b;
}

void pav(double *ghat, double *y, double *w, size_t n)
{
    int *index = (int *)mxMalloc(sizeof(int) * n);
    double *len = (double *)mxMalloc(sizeof(double) * n);

    int ci = 0;
    index[ci] = 0;
    len[ci] = w[0];
    ghat[ci] = y[0];

    for (int j = 1; j < n; ++j) {
        ci += 1;
        index[ci] = j;
        len[ci] = w[j];
        ghat[ci] = y[j];

        while (ci >= 1 && ghat[max(ci - 1, 0)] >= ghat[ci]) {
            double nw = len[ci - 1] + len[ci];
            ghat[ci - 1] = ghat[ci - 1] + (len[ci] / nw) * (ghat[ci] - ghat[ci - 1]);
            len[ci - 1] = nw;
            ci -= 1;
        }
    }

    while (n >= 1) {
        for (int j = index[ci]; j < n; ++j) {
            ghat[j] = ghat[ci];
        }
        n = index[ci];
        ci -= 1;
    }

    mxFree(index);
    mxFree(len);
}

void grad_w_block(double *rho_i, double *w_i, double *weights_i, size_t n)
{
    double *y_i = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; ++i) {
        y_i[i] = w_i[i] / weights_i[i];
    }

    // rho_i is still the wrong sign so we will need to flip it
    pav(rho_i, y_i, weights_i, n);

    // we only needed y_i for the pav call
    free(y_i);

    // flip the sign of rho_i
    const double alpha = -1;
    const ptrdiff_t inc = 1;
    dscal(&n, &alpha, rho_i, &inc);
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
	double *y;	// the vector we want to project
	const mxArray *w; // the dual variable w
    const mxArray *step; // the step size
    const mxArray *direction; // the Frank-Wolfe direction to move in
    const mxArray *weights; // the weights corresponding to the regularizer
    
    double *outMatrix; // output matrix -- actually just a scalar

    /* check for proper number of arguments */
    if(nlhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:fw_dual_objective:nlhs","One output required.");
    }

    if(nrhs != 4) {
        mexErrMsgIdAndTxt("MATLAB:fw_dual_objective:nrhs","Four inputs required.");
    }
    
    w = prhs[0];
    step = prhs[1];
    direction = prhs[2];
    weights = prhs[3];
    
    if( !mxIsDouble(step) || mxIsComplex(step) ||
        !(mxGetM(step)==1 && mxGetN(step)==1) ) {
        mexErrMsgIdAndTxt( "MATLAB:fw_dual_objective:inputNotRealScalarDouble",
        "Input variable \"step\" must be a noncomplex scalar double.");
    }

    // validate that the input cells are the same dimensions
    size_t num_cell_dims = mxGetM(w);
    if (num_cell_dims != mxGetM(direction) ||
        num_cell_dims != mxGetM(weights)) {
        mexErrMsgIdAndTxt("MATLAB:fw_dual_objective:inputNotAligned",
        "Input cells w, direction, and weights must have the same dimensions.");
    }

    double *step_double = mxGetPr(step);

    double val = 0; // this is the objective val we will return

    mxArray *w_new = mxCreateCellMatrix((mwSize)num_cell_dims, (mwSize)1);
    for (int i = 0; i < num_cell_dims; ++i) {
        mxArray *w_i = mxGetCell(w, i);
        mxArray *direction_i = mxGetCell(direction, i);
        mxArray *weights_i = mxGetCell(weights, i);

        ptrdiff_t ith_dim = (ptrdiff_t)mxGetM(w_i);

        if (ith_dim != mxGetM(direction_i) ||
            ith_dim != mxGetM(weights_i)) {
            mexErrMsgIdAndTxt("MATLAB:fw_dual_objective:inputNotAligned",
            "Input cells w, direction, and weights must have the same dimensions, in each dimension.");
        }

        mxArray *w_new_i = mxCreateDoubleMatrix((mwSize)ith_dim, (mwSize)1, mxREAL);

        double *w_new_i_double = mxGetPr(w_new_i);
        double *w_i_double = mxGetPr(w_i);
        double *direction_i_double = mxGetPr(direction_i);
        double *weights_i_double = mxGetPr(weights_i);

        // dummy variable that is always one
        const ptrdiff_t inc = 1;
        
        // copy vector using BLAS
        dcopy(&ith_dim, w_i_double, &inc, w_new_i_double, &inc);
        // scale by (1-step)
        const double one = 1; // - step_double[0];
        dscal(&ith_dim, &one, w_new_i_double, &inc);
        // add step*direction_i
        daxpy(&ith_dim, step_double, direction_i_double, &inc, w_new_i_double, &inc);

        // now we can compute the corresponding part of the gradient
        // and then add the inner product + regularized sum
        double *rho_i = (double *)malloc(sizeof(double) * ith_dim);
        grad_w_block(rho_i, w_new_i_double, weights_i_double, ith_dim);

        // inner product component
        val += ddot(&ith_dim, w_new_i_double, &inc, rho_i, &inc);
        
        // square all elements of rho_i in-place
        for (int j = 0; j < ith_dim; ++j) {
            rho_i[j] = rho_i[j] * rho_i[j];
        }

        val += 0.5 * ddot(&ith_dim, rho_i, &inc, weights_i_double, &inc);
        free(rho_i);
    }

    /* create the output scalar and set it to the output value, val */
    plhs[0] = mxCreateDoubleScalar(val);
}
