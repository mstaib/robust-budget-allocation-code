#include "mex.h"
#include "matrix.h"
#include <string.h>

inline int max(int a, int b) {
    return a > b ? a : b;
}

void build_all_x_vectors(double *out_matrix, double *x_init, size_t x_ncols, double *indices, size_t indices_ncols)
{
	memcpy(out_matrix, x_init, x_ncols*sizeof(double));

    double *prev_vector = x_init;
	for (int i = 0; i < indices_ncols; ++i)
	{
		size_t offset = i * x_ncols;
		
		// copy the previous vector before we add one in some index
		memcpy(out_matrix + offset, prev_vector, x_ncols*sizeof(double));
		
		// increase the corresponding index by one
		size_t inx_to_update = offset + ( (int)indices[i] - 1 );
		out_matrix[inx_to_update] += 1;

        // update the pointer to the previous vector
        prev_vector = out_matrix + offset;
	}
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
	double *x_init;	// the starting x value which we incrementally add to
	size_t x_ncols; // number of elements of x, which is a column vector
	double *indices; // the ordered list of indices which determines in what order we increment x
	size_t indices_ncols;
    
    double *outMatrix;              /* output matrix */

    /* check for proper number of arguments */
    if(nrhs!=2) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","Two inputs required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required.");
    }
    
    x_init = mxGetPr(prhs[0]);
    x_ncols = max(mxGetN(prhs[0]), mxGetM(prhs[0]));

    indices = mxGetPr(prhs[1]);
    indices_ncols = max(mxGetN(prhs[1]), mxGetM(prhs[1]));

    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((mwSize)x_ncols, (mwSize)indices_ncols, mxREAL);

    /* get a pointer to the real data in the output matrix */
    outMatrix = mxGetPr(plhs[0]);

    build_all_x_vectors(outMatrix, x_init, x_ncols, indices, indices_ncols);
}