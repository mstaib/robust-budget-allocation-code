clear;
files = dir('yahoo_test_data/*.mat');

%%%% THIS MAY TAKE LOTS OF MEMORY

%% load in all out_structs and test_params
k=1;
for file = files'
    regexp_struct = regexp(file.name,'yconstraint_(?<confidence>.+)_confidence_(?<eps>.+)_eps_(?<data_frac>.+)_data_(?<num_edges>\d+)_edges','names');
    if isempty(regexp_struct)
        continue;
    end
    
    eps = str2num(regexp_struct.eps);
    confidence = str2num(regexp_struct.confidence);
    data_frac = str2num(regexp_struct.data_frac);
    num_edges = str2num(regexp_struct.num_edges);
    
    %if T == 20 && maxnumtrials == 20
    
        load(strcat('yahoo_test_data/',file.name));
        test_param.confidence = confidence;
    
        fprintf(file.name); fprintf('\n');
        
        out_structs(k) = out_struct_max;
        test_params(k) = test_param;
        summaries(k) = summary;
        k=k+1;
       
   % end
    
end

%% plots
figure;
plot([test_params.confidence], [summaries.adversarial_y_robust], 'LineWidth', 2); hold on
%plot([test_params.confidence], [summaries.adversarial_y_robust_upper_bound], 'LineWidth', 2); hold on
plot([test_params.confidence], [summaries.adversarial_y_nom], 'LineWidth', 2); hold on
%plot([test_params.confidence], [summaries.adversarial_y_expect], 'LineWidth', 2); hold on
ylim([0 1.1*max([summaries.adversarial_y_robust_upper_bound])]);
legend('Maximize robustly (lower bound)', 'Maximize per nominal values', 'Location', 'southwest');
xlabel('confidence interval width');
ylabel('influence');
title('Influence in the adversarial case');

% figure;
% plot([test_params.confidence], [summaries.expected_y_robust], 'LineWidth', 2); hold on
% plot([test_params.confidence], [summaries.expected_y_nom], 'LineWidth', 2); hold on
% plot([test_params.confidence], [summaries.expected_y_expect], 'LineWidth', 2); hold on
% ylim([0 1.1*max([summaries.expected_y_expect])]);
% legend('Maximize robustly (lower bound)', 'Maximize per nominal values', 'Maximize expectation', 'Location', 'southwest');
% xlabel('confidence interval width');
% ylabel('influence');
% title('Influence in the Beta distribution case');

figure;
plot([test_params.confidence], [summaries.nominal_y_robust], 'LineWidth', 2); hold on
plot([test_params.confidence], [summaries.nominal_y_nom], 'LineWidth', 2); hold on
%plot([test_params.confidence], [summaries.nominal_y_expect], 'LineWidth', 2); hold on
ylim([0 1.1*max([summaries.nominal_y_nom])]);
legend('Maximize robustly (lower bound)', 'Maximize per nominal values', 'Location', 'southwest');
xlabel('confidence interval width');
ylabel('influence');
title('Influence in the nominal case (probabilities were exact)');

%% copied from test_robust_influence_maximization_instance

for kk=1:length(test_params)
    test_param = test_params(kk);
    summary = summaries(kk);
    
    problem = test_param.problem;
    eps = test_param.eps;
    
    % setup the parameters needed for function evaluation
    param_F.S = problem.S;
    param_F.T = problem.T;

    %param_F.x_centers = problem.x_median; % used only to fill in the missing values
    adversarial_frac = 1;
    param_F.x_centers = adversarial_frac * problem.x_upper + (1-adversarial_frac) * problem.x_median;
    param_F.alpha = problem.alpha;

    param_F.k_mat = discretize_bipartite_problem(eps, problem);
    param_F.x_lower = problem.x_median; %problem.x_lower
    param_F.x_upper = problem.x_upper;
    param_F.edges = problem.real_edges;
    param_F.var_edges = problem.var_edges;

    param_F.k_vec = param_F.k_mat(problem.var_edges);
    param_F.x_lower_vec = param_F.x_lower(problem.var_edges);
    param_F.x_upper_vec = param_F.x_upper(problem.var_edges);
    
    param_F.use_gpu = false;
    
    if gpuDeviceCount > 0
        param_F.use_gpu = true;
        
        % go ahead and create copies of some parameters on the GPU;
        param_F.x_centers_gpu = gpuArray(single(param_F.x_centers));
        param_F.var_edges_gpu = gpuArray(single(param_F.var_edges));
        param_F.var_edges_inx_gpu = gpuArray(single(find(param_F.var_edges)));

        param_F.k_vec_gpu = gpuArray(single(param_F.k_vec));
        param_F.x_lower_vec_gpu = gpuArray(single(param_F.x_lower_vec));
        param_F.x_upper_vec_gpu = gpuArray(single(param_F.x_upper_vec));
        
        param_F.not_edges_inx_gpu = gpuArray(single(find(~param_F.edges)));
        
        param_F.alpha_gpu = gpuArray(single(param_F.alpha));
        param_F.S_gpu = gpuArray(single(param_F.S));
        param_F.T_gpu = gpuArray(single(param_F.T));
    end
    
    confidences(kk) = test_param.confidence;
    worst_case_all_edges_robust(kk) = submodular_fct_influence_adversary_vec_wrapper(param_F.x_upper_vec_gpu, summary.y_robust, param_F);
    worst_case_all_edges_nominal(kk) = submodular_fct_influence_adversary_vec_wrapper(param_F.x_upper_vec_gpu, summary.y_nom, param_F);
    
    % see how each budget compares to the same type of adversary (robust budget)
    var_edges_robust = choose_var_edges(problem, eps, test_param.y_constraint, num_edges, summary.y_robust);
    param_F.var_edges = var_edges_robust;
    param_F.k_vec = param_F.k_mat(param_F.var_edges);
    param_F.x_lower_vec = param_F.x_lower(param_F.var_edges);
    param_F.x_upper_vec = param_F.x_upper(param_F.var_edges);
     if gpuDeviceCount > 0
        param_F.use_gpu = true;
        
        % go ahead and create copies of some parameters on the GPU;
        param_F.var_edges_gpu = gpuArray(single(param_F.var_edges));
        param_F.var_edges_inx_gpu = gpuArray(single(find(param_F.var_edges)));

        param_F.k_vec_gpu = gpuArray(single(param_F.k_vec));
        param_F.x_lower_vec_gpu = gpuArray(single(param_F.x_lower_vec));
        param_F.x_upper_vec_gpu = gpuArray(single(param_F.x_upper_vec));
    end
    
end

% more plots
figure;
plot(confidences, worst_case_all_edges_robust, 'LineWidth', 2); hold on
plot(confidences, worst_case_all_edges_nominal, 'LineWidth', 2); hold on
ylim([0 1.1*max(max(worst_case_all_edges_robust), max(worst_case_all_edges_nominal))]);
legend('Maximize robustly (lower bound)', 'Maximize per nominal values', 'Location', 'southwest');
xlabel('confidence interval width');
ylabel('influence');
title('Influence in the case with maximal adversary on all edges');