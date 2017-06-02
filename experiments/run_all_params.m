function [] = run_all_params(test_params)
% each test_param will have the following fields:
% x_constraint
% y_constraint
% eps
% verbose
% maxiters
% problem
%   (among other) max_num_trials
%   S
%   T
%   confidence
% uncertainty_set_type
% adversary_algorithm

for ii=1:length(test_params)
    test_param = test_params(ii);

    try
        [ summary, out_struct_max ] = test_robust_influence_maximization_instance(test_param);
    catch ME
        fprintf('Caught exception for case confidence=%0.2f, skipping\n', test_param.problem.confidence);
        fprintf('Exception message: %s\n', ME.message);
        continue
    end

    uuid = char(java.util.UUID.randomUUID);
    filepath = strcat('all_local_data/', uuid, '.mat');
    fprintf('Finished run, saving to %s\n', filepath);
    save(filepath, 'out_struct_max', 'test_param', 'summary');
end

end