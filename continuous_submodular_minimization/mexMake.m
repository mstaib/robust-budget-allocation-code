mex -O CFLAGS="\$CFLAGS -std=c99 -march=native -O3" continuous_submodular_minimization/build_all_x_vectors_custom.c -outdir continuous_submodular_minimization/
mex -O CFLAGS="\$CFLAGS -std=c99 -march=native -O3" continuous_submodular_minimization/build_increment_ordering_mex.c -outdir continuous_submodular_minimization/
mex -O CFLAGS="\$CFLAGS -std=c99 -march=native -O3" continuous_submodular_minimization/build_w_from_F_vals_mex.c -outdir continuous_submodular_minimization/
mex -O -largeArrayDims -lblas CFLAGS="\$CFLAGS -std=c99 -march=native -O3" continuous_submodular_minimization/cell_innerprod_mex.c -outdir continuous_submodular_minimization/
mex -O -largeArrayDims -lblas CFLAGS="\$CFLAGS -std=c99 -march=native -O3" continuous_submodular_minimization/cell_sum_mex.c -outdir continuous_submodular_minimization/
mex -O CFLAGS="\$CFLAGS -std=c99 -march=native -O3" continuous_submodular_minimization/fill_in_subsampled_x_all_no_mapping_mex.c -outdir continuous_submodular_minimization/
mex -O -largeArrayDims -lblas CFLAGS="\$CFLAGS -std=c99 -march=native -O3" continuous_submodular_minimization/fw_dual_objective_custom.c -outdir continuous_submodular_minimization/
mex -O -largeArrayDims -lblas CFLAGS="\$CFLAGS -std=c99 -march=native -O3" continuous_submodular_minimization/incremental_differences_loop_mex.c -outdir continuous_submodular_minimization/
mex -O -largeArrayDims -lblas CFLAGS="\$CFLAGS -std=c99 -march=native -O3" continuous_submodular_minimization/pav_custom.c -outdir continuous_submodular_minimization/