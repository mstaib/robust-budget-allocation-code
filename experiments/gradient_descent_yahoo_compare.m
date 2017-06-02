%% solve optimization problem

file_to_load = strcat('data_from_cluster/all_local_yahoo_data/', 'eb949097-415b-4393-9a96-330d66d323b6.mat');
load(file_to_load);

[param_max, param_min, R_cell, oracle_R] = prepare_inputs(test_param, false);
param_min.y_mat = summary.y_robust;
% 
% %% continuous submodular
% param_F = param_min.param_F;
% param_F.y_mat = param_min.y_mat;
% param_min.param_F = param_F;
% F = @(x,param_F) submodular_fct_influence_adversary_vec_wrapper(x,param_min.y_mat,param_F);
% param_min.maxiter = 20;
% [xmin,Fmin,out_struct_min,csg] = minimize_submodular_constrained(F,R_cell,oracle_R,param_min);
% xmin_unscaled = interpolate(xmin, param_F.x_lower_vec, param_F.x_upper_vec, param_F.k_vec);
% 

%% nonconvex frank-wolfe
% set up linear constraints
a = (param_min.param_F.x_upper_vec - param_min.param_F.x_lower_vec).^(-1);
a = a';
b = param_min.constraint + dot(a, param_min.param_F.x_lower_vec);

n = length(R_cell);
x = param_min.param_F.x_lower_vec;
iters = 1000;
x_hist = x;
y_mat_diag = diag(param_min.y_mat);
tol = 1e-5;
fval = submodular_fct_influence_adversary_vec_wrapper_no_scaling(x, param_min.y_mat, param_min.param_F);

[r,res] = mosekopt('param');
res.param.MSK_IPAR_LOG=0;

g = Inf;
g_hist = [];
gamma_hist = [];
for iter=1:iters
    x_mat = param_min.param_F.x_lower_sparse;
    x_mat(param_min.param_F.var_edges) = interpolate(x, param_min.param_F.x_lower_vec, param_min.param_F.x_upper_vec, param_min.param_F.k_vec);
    grad_x_mat = submodular_fct_influence_adversary_x_grad_sparse(spfun(@log, x_mat), y_mat_diag);
    grad_x = grad_x_mat(param_min.param_F.var_edges);

%     % FW Linear oracle
%     %[s, ~] = linprog(grad_x, a, b, [], [], param_min.param_F.x_lower_vec, param_min.param_F.x_upper_vec);
%     [res_new] = msklpopt(grad_x,a,[],b,param_min.param_F.x_lower_vec, param_min.param_F.x_upper_vec,res.param);
%     s = res_new.sol.itr.xx;
% 
%     d = s - x;
%     g = dot(d, -grad_x);
%     if g < tol
%         break
%     end
% 
%     gamma = fmincon(@(t) submodular_fct_influence_adversary_vec_wrapper_no_scaling(x + t * d, param_min.y_mat, param_min.param_F), 0.5, [], [], [], [], 0, 1);
%     %gamma = 0.1;
%     x = x + gamma * d;
    
    % Projection oracle
    gamma = 0.5;
    y = x - gamma * grad_x;
    [res_new] = mskqpopt(speye(n),-y,a,[],b,param_min.param_F.x_lower_vec, param_min.param_F.x_upper_vec,res.param);
    x = res_new.sol.itr.xx;
    



    g_hist = [g_hist g];
    x_hist = [x_hist x];
    gamma_hist = [gamma_hist gamma];
    fval = [fval submodular_fct_influence_adversary_vec_wrapper_no_scaling(x, param_min.y_mat, param_min.param_F)];
end

%gamma_hists{ii} = gamma_hist;
g_hists{ii} = g_hist;
fvals{ii} = fval;
x_hists{ii} = x_hist;


%% comparison plot
figure('Units','inches', ...
    'Position',[0 0 1.9 1.3], ...
    'PaperPositionMode', 'auto');

fval = fvals{1};
niters = length(fval);

ax1 = axes;
plot(ax1, 0:niters-1, fval, 'LineWidth', 2); hold on;
plot(ax1, 0:niters-1, summary.adversarial_y_robust*ones(niters,1), 'LineWidth', 2);

axis(ax1, [0 80 7500 10500]);
set(ax1,...
    'Units','normalized',...
    'YTick',7500:1000:10500,...
    'XTick',0:40:niters,...
    'Position', [0.2059 0.1940 0.62 0.7310],...
    'FontUnits','points',...
    'FontWeight','normal',...
    'FontSize',6,...
    'FontName','Times');

xlabel(ax1, 'Iteration',...
    'FontUnits','points',...
    'interpreter','latex',...
    'FontSize',6,...
    'FontName','Times');
ylabel(ax1, 'Influence',...
    'FontUnits','points',...
    'interpreter','latex',...
    'FontSize',6,...
    'FontName','Times');
legend(ax1, {'FW','SFM'},...
    'Location','northeast',...
    'FontUnits','points',...
    'interpreter','latex',...
    'FontSize',6,...
    'FontName','Times');

% set(subplot(1,2,1,ax1),'Position', [0.15 0.2 0.3347 0.77]);
% 
% % plot 2
% ax2 = axes;
% plot(ax2, 0:niters-1, fval, 'LineWidth', 2); hold on;
% plot(ax2, 0:niters-1, summary.adversarial_y_robust*ones(niters,1), 'LineWidth', 2);
% 
% axis(ax2, [40 80 7500 8000]);
% set(ax2,...
%     'Units','normalized',...
%     'YTick',7500:250:8000,...
%     'XTick',40:20:niters,...
%     'Position',[.3 .2 .65 .7],...
%     'FontUnits','points',...
%     'FontWeight','normal',...
%     'FontSize',6,...
%     'FontName','Times');
% 
% xlabel(ax2, 'Iteration',...
%     'FontUnits','points',...
%     'interpreter','latex',...
%     'FontSize',6,...
%     'FontName','Times');
% % ylabel(ax2, 'Influence',...
% %     'FontUnits','points',...
% %     'interpreter','latex',...
% %     'FontSize',6,...
% %     'FontName','Times');
% legend(ax2, {'FW','SFM'},...
%     'Location','northeast',...
%     'FontUnits','points',...
%     'interpreter','latex',...
%     'FontSize',6,...
%     'FontName','Times');
% 
% set(subplot(1,2,2,ax2),'Position', [0.5903 0.2 0.3347 0.77]);

%print -depsc2 ../plots/fw-compare.eps



function val = submodular_fct_influence_adversary_vec_wrapper_no_scaling(x_var_edges,y_mat,param)
% wrapper for submodular_fct_influence_adversary which converts x_vec to
% a matrix as needed

[~, num_vecs] = size(x_var_edges);

% how many channels
S = length(y_mat);
y_mat_diag = spdiags(y_mat,0,S,S);

log_x_centers_sparse = spfun(@log, param.x_centers_sparse);
log_x_var_edges = log(x_var_edges);

val = zeros(num_vecs,1);
for ii=1:num_vecs
    log_x = log_x_centers_sparse;
    log_x(param.var_edges_inx) = log_x_var_edges(:,ii);
    val(ii) = submodular_fct_influence_adversary_sparse(log_x,y_mat_diag);
end
    
end


function [param_max, param_min, R_cell, oracle_R] = prepare_inputs(test_param, from_beta_dist)
% input: test_param must contain the fields:
%   eps
%   maxiters
%   x_constraint
%   y_constraint
%   problem

    problem = test_param.problem;
    
    if test_param.uncertainty_set_type == UncertaintySetType.Ellipsoidal
        problem.x_upper = ones(size(problem.x_upper));
    end
    
    param_F = submodular_fct_influence_adversary_build_param_from_problem(problem, test_param.eps);
    
    
    %% input setup for minimization problem
    param_min.param_F = param_F;
    param_min.maxiter = 200;
    param_min.k_vec = param_F.k_vec; %param_F.k_mat(param_F.var_indices);
    param_min.x_lower = param_F.x_lower; %param_F.x_lower(param_F.var_indices);
    param_min.x_upper = param_F.x_upper; %param_F.x_upper(param_F.var_indices);
    param_min.constraint = test_param.x_constraint;

    % we should get eps/2 error from each of discretization and the actual
    % minimization, param_F
    param_min.early_terminate_gap = test_param.eps/5;%1e-3;
    param_min.verbose = false;

    %% input setup for maximization problem
    param_max.maxiters = test_param.maxiters;
    param_max.simplex_constraint = test_param.y_constraint;
    param_max.early_terminate_gap = test_param.eps;%1e-4;

    if isfield(test_param, 'verbose')
        param_max.verbose = (test_param.verbose >= 1);
        param_min.verbose = (test_param.verbose >= 2);
    else
        param_max.verbose = false;
    end
    
    R_cell = cell(size(param_min.k_vec));
    if test_param.uncertainty_set_type == UncertaintySetType.Ellipsoidal
        assert(from_beta_dist, 'Ellipsoidal uncertainty set only supported if the problem comes from a beta distribution\n');
        
        alphas = test_param.problem.alphas(param_F.var_edges_inx);
        betas = test_param.problem.betas(param_F.var_edges_inx);
        variance = alphas .* betas ./ ( (alphas+betas).^2 .* (alphas+betas+1) );
        
        for ii=1:length(param_min.k_vec)
            R_cell{ii} = @(t) (t - param_F.x_lower_vec(ii)).^2 / variance(ii).^2;
        end
        oracle_R = @(x) sum((x - param_F.x_lower_vec).^2 ./ variance.^2);
    else % UncertaintySetType.Dnorm
        for ii=1:length(param_min.k_vec)
            R_cell{ii} = @(t) (t - param_F.x_lower_vec(ii)) / (param_F.x_upper_vec(ii) - param_F.x_lower_vec(ii));
            %R_cell{ii} = @(t) abs(t - param_F.x_lower_vec(ii));
        end
        oracle_R = @(x) sum((x - param_F.x_lower_vec) ./ (param_F.x_upper_vec - param_F.x_lower_vec));
    end
end