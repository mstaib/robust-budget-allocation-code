mex -O CFLAGS="\$CFLAGS -std=c99 -march=native -O3" build_all_x_vectors_custom.c
mex -O CFLAGS="\$CFLAGS -std=c99 -march=native -O3" build_increment_ordering_mex.c
mex -O CFLAGS="\$CFLAGS -std=c99 -march=native -O3" build_w_from_F_vals_mex.c
mex -O -largeArrayDims -lblas CFLAGS="\$CFLAGS -std=c99 -march=native -O3" cell_innerprod_mex.c
mex -O -largeArrayDims -lblas CFLAGS="\$CFLAGS -std=c99 -march=native -O3" cell_sum_mex.c
mex -O CFLAGS="\$CFLAGS -std=c99 -march=native -O3" fill_in_subsampled_x_all_no_mapping_mex.c
mex -O -largeArrayDims -lblas CFLAGS="\$CFLAGS -std=c99 -march=native -O3" fw_dual_objective_custom.c
mex -O -largeArrayDims -lblas CFLAGS="\$CFLAGS -std=c99 -march=native -O3" incremental_differences_loop_mex.c
mex -O -largeArrayDims -lblas CFLAGS="\$CFLAGS -std=c99 -march=native -O3" pav_custom.c