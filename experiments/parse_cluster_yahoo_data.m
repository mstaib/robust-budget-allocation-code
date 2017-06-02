clear;
files = dir('data_from_cluster/all_local_yahoo_data/*.mat');

%% load in all out_structs and test_params
k=1;
for file = files'
    load(strcat('data_from_cluster/all_local_yahoo_data/',file.name));
    fprintf(file.name); fprintf('\n');
    
    % prune some dense matrices
    prob = test_param.problem;
    prob = rmfield(prob, 'alphas');
    prob = rmfield(prob, 'betas');
    prob = rmfield(prob, 'x_lower');
    prob = rmfield(prob, 'x_upper');
    prob = rmfield(prob, 'x_median');
    prob = rmfield(prob, 'x_centers_empirical');
    test_param.problem = prob;
    
    %if isfield(out_struct_max, 'constrained_certified_suboptimality_gap')
        out_structs(k) = out_struct_max;
        test_params(k) = test_param;
        summaries(k) = summary;
        filenames{k} = file.name;
        k=k+1;    
    %end
end
k=k-1;

fprintf('Done loading files\n');

% sort the test cases by oblivious_gap so we can diagnose any weird cases
[~,I] = sort([summaries.adversarial_y_nom]);
out_structs = out_structs(I);
test_params = test_params(I);
summaries = summaries(I);

%% filter into dnorm and ellipsoidal
e_k = 1; d_k = 1;

for ii=1:k
    if test_params(ii).uncertainty_set_type == UncertaintySetType.Ellipsoidal
        out_structs_ellipsoidal(e_k) = out_structs(ii);
        %test_params_ellipsoidal(e_k) = test_params(ii);
        summaries_ellipsoidal(e_k) = summaries(ii);
        
        test_param = test_params(ii);
        
        test_param.x_constraint_frac = ...
            test_param.x_constraint / ...
            nnz(test_param.problem.var_edges)^2;
        test_param.y_constraint_frac = ...
            test_param.y_constraint / ...
            test_param.problem.S;
        
        test_params_ellipsoidal(e_k) = test_param;
        filenames_ellipsoidal{e_k} = filenames{ii};
        
        e_k = e_k + 1;
    elseif test_params(ii).uncertainty_set_type == UncertaintySetType.Dnorm
        out_structs_dnorm(d_k) = out_structs(ii);
        %test_params_dnorm(d_k) = test_params(ii);
        summaries_dnorm(d_k) = summaries(ii);
        
        test_param = test_params(ii);
        
        test_param.x_constraint_frac = ...
            test_param.x_constraint / ...
            nnz(test_param.problem.var_edges);
        test_param.y_constraint_frac = ...
            test_param.y_constraint / ...
            test_param.problem.S;
        
        test_params_dnorm(d_k) = test_param;
        filenames_dnorm{d_k} = filenames{ii};
        
        d_k = d_k + 1;
    end
end
e_k = e_k - 1; d_k = d_k - 1;


%% plot oblivious gap vs the entire parameter space
figure;
scatter3([test_params_dnorm.x_constraint_frac], [test_params_dnorm.y_constraint_frac], [summaries_dnorm.oblivious_gap]);
xlabel('x constraint frac');
ylabel('y constraint frac');
zlabel('oblivious gap');
title('Dnorm');

figure;
scatter3([test_params_ellipsoidal.x_constraint_frac], [test_params_ellipsoidal.y_constraint_frac], [summaries_ellipsoidal.oblivious_gap]);
xlabel('x constraint frac');
ylabel('y constraint frac');
zlabel('oblivious gap');
title('Ellipsoidal');

%% plot optimality gap vs the entire parameter space
figure;
scatter3([test_params_dnorm.x_constraint_frac], [test_params_dnorm.y_constraint_frac], arrayfun(@(x) x.dual_gap(end), out_structs_dnorm));
xlabel('x constraint frac');
ylabel('y constraint frac');
zlabel('optimality gap');
title('Dnorm');

figure;
scatter3([test_params_ellipsoidal.x_constraint_frac], [test_params_ellipsoidal.y_constraint_frac], arrayfun(@(x) x.dual_gap(end), out_structs_ellipsoidal));
xlabel('x constraint frac');
ylabel('y constraint frac');
zlabel('optimality gap');
title('Ellipsoidal');



%%
% %%
% figure('Units','inches', ...
%     'Position',[0 0 3.3 3], ...
%     'PaperPositionMode', 'auto');
% 
% adversarial_y_nom = [summaries.adversarial_y_nom];
% adversarial_y_robust = [summaries.adversarial_y_robust];
% 
% max_val = max( max(adversarial_y_nom), max(adversarial_y_robust) );
% 
% plot(adversarial_y_nom, adversarial_y_robust, 'o'); hold on;
% plot(0:0.001:1.1*max_val, 0:0.001:1.1*max_val);
% axis([0 max_val 0 max_val]);
% set(gca,...
%     'Units','normalized',...
%     'YTick',0:0.05:max_val,...
%     'XTick',0:0.05:max_val,...
%     'Position',[.15 .2 .75 .7],...
%     'FontUnits','points',...
%     'FontWeight','normal',...
%     'FontSize',10,...
%     'FontName','Times');
% 
% xlabel('Worst-case influence of $y_{\mathrm{nom}}$',...
%     'FontUnits','points',...
%     'interpreter','latex',...
%     'FontSize',10,...
%     'FontName','Times');
% ylabel('Worst-case influence of $y_{\mathrm{robust}}$',...
%     'FontUnits','points',...
%     'interpreter','latex',...
%     'FontSize',10,...
%     'FontName','Times');
% 
% title({'Robust vs non-robust budgets', 'for ellipsoidal uncertainty'},...
%     'FontUnits','points',...
%     'FontWeight','normal',...
%     'FontSize',10,...
%     'FontName','Times');
% 
% %print -depsc2 ../plots/ellipsoidal-robust-vs-nom.eps

%% diagnostic plot
figure('Units','inches', ...
    'Position',[0 0 3.3 3], ...
    'PaperPositionMode', 'auto');

gammas = [test_params.x_constraint];
% sort the test cases by gammas
[~,I] = sort(gammas);
gammas = gammas(I);
out_structs = out_structs(I);
test_params = test_params(I);
summaries = summaries(I);

adversarial_y_nom = [summaries.adversarial_y_nom];
adversarial_y_robust = [summaries.adversarial_y_robust];
adversarial_y_expect = [summaries.adversarial_y_expect];

max_gamma = max(gammas);
min_gamma = min(gammas);
max_influence = max( [max(adversarial_y_nom), max(adversarial_y_robust), max(adversarial_y_expect)] );



plot(gammas, adversarial_y_robust, '-'); hold on;
plot(gammas, adversarial_y_nom, '-'); hold on;
plot(gammas, adversarial_y_expect, '-');

axis([min_gamma max_gamma 0 max_influence]);
set(gca,...
    'Units','normalized',...
    'YTick',0:0.1:0.65,...%max_influence,...
    'XTick',min_gamma:300:max_gamma,...
    'Position',[.15 .2 .75 .7],...
    'FontUnits','points',...
    'FontWeight','normal',...
    'FontSize',10,...
    'FontName','Times');

xlabel('Adversary constraint $\gamma$',...
    'FontUnits','points',...
    'interpreter','latex',...
    'FontSize',10,...
    'FontName','Times');
ylabel('Worst-case influence',...
    'FontUnits','points',...
    'interpreter','latex',...
    'FontSize',10,...
    'FontName','Times');
legend({'$y_{\mathrm{robust}}$','$y_{\mathrm{nom}}$','$y_{\mathrm{expect}}$'},...
    'FontUnits','points',...
    'interpreter','latex',...
    'FontSize',10,...
    'FontName','Times');


title({'Influence as adversary power changes','for ellipsoidal uncertainty'},...
    'FontUnits','points',...
    'FontWeight','normal',...
    'FontSize',10,...
    'FontName','Times');

%print -depsc2 ../plots/ellipsoidal-robust-vs-nom.eps


%% how well do we do on the 'true' x_centers value?
true_y_nom = zeros(k,1);
true_y_robust = zeros(k,1);
true_y_expect = zeros(k,1);
for ii=1:k
    param_F = submodular_fct_influence_adversary_build_param_from_problem(test_params(ii).problem, test_params(ii).eps);
    x_true_vec = test_params(ii).problem.x_centers_true(test_params(ii).problem.var_edges);
    
    true_y_nom(ii) = submodular_fct_influence_adversary_vec_wrapper(x_true_vec, summaries(ii).y_nom, param_F);
    true_y_robust(ii) = submodular_fct_influence_adversary_vec_wrapper(x_true_vec, summaries(ii).y_robust, param_F);
    true_y_expect(ii) = submodular_fct_influence_adversary_vec_wrapper(x_true_vec, summaries(ii).y_expect, param_F);
end

figure('Units','inches', ...
    'Position',[0 0 3.3 3], ...
    'PaperPositionMode', 'auto');

gammas = [test_params.x_constraint];
max_gamma = max(gammas);
min_gamma = min(gammas);
max_influence = max( [max(true_y_nom), max(true_y_robust), max(true_y_expect)] );

plot(gammas, true_y_robust, '-'); hold on;
plot(gammas, true_y_nom, '-'); hold on;
plot(gammas, true_y_expect, '-');

axis([min_gamma max_gamma 0 max_influence]);
set(gca,...
    'Units','normalized',...
    'YTick',0:0.1:max_influence,...
    'XTick',min_gamma:300:max_gamma,...
    'Position',[.15 .2 .75 .7],...
    'FontUnits','points',...
    'FontWeight','normal',...
    'FontSize',10,...
    'FontName','Times');

xlabel('Adversary constraint $\gamma$',...
    'FontUnits','points',...
    'interpreter','latex',...
    'FontSize',10,...
    'FontName','Times');
ylabel('Worst-case influence',...
    'FontUnits','points',...
    'interpreter','latex',...
    'FontSize',10,...
    'FontName','Times');
legend({'$y_{\mathrm{robust}}$','$y_{\mathrm{nom}}$','$y_{\mathrm{expect}}$'},...
    'location','southwest',...
    'FontUnits','points',...
    'interpreter','latex',...
    'FontSize',10,...
    'FontName','Times');


title({'Robust vs non-robust budgets', 'for ellipsoidal uncertainty', 'as adversary power changes', 'for true influence function'},...
    'FontUnits','points',...
    'FontWeight','normal',...
    'FontSize',10,...
    'FontName','Times');