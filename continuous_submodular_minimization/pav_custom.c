#include "mex.h"
#include "matrix.h"
#include <string.h>

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

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
	double *y;	// the vector we want to project
	double *w; // the weights for the regression
	size_t n; // the dimension of y and w (should be the same)
    
    double *outMatrix;              /* output matrix */

    /* check for proper number of arguments */
    if(nlhs != 1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required.");
    }
    if(nrhs < 1 || nrhs > 2) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","One or two inputs required.");
    }
    
    y = mxGetPr(prhs[0]);
    n = mxGetM(prhs[0]);

    if(nrhs == 1) {
        mxArray *w_obj = mxCreateDoubleMatrix((mwSize)n, (mwSize)1, mxREAL);
        w = mxGetPr(w_obj);
        for (int i = 0; i < n; ++i) {
            w[i] = 1;
        }
    } else {
        w = mxGetPr(prhs[1]);
        size_t w_size = mxGetM(prhs[1]);

        if (w_size != n) {
            mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Both inputs must be the same dimension.");
        }
    }    

    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((mwSize)n, (mwSize)1, mxREAL);

    /* get a pointer to the real data in the output matrix */
    outMatrix = mxGetPr(plhs[0]);

    pav(outMatrix, y, w, n);
}
