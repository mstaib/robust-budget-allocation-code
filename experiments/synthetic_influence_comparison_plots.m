clear;
files = dir('data_from_cluster/all_local_data_2_no_rhos/*.mat');

% the specific y_constraint vals we want
vals = logspace(log10(0.2), log10(10), 8);
y_constraints = 2*vals([1 5]);

%% load in all out_structs and test_params
dsk=1;
esk=1;
dlk=1;
elk=1;
for file = files'
    load(strcat('data_from_cluster/all_local_data_2_no_rhos/',file.name));
    %fprintf(file.name); fprintf('\n');

    if test_param.adversary_algorithm == AdversaryAlgorithm.SubmodularMinimization & ...
            find(y_constraints == test_param.y_constraint)
        if test_param.uncertainty_set_type == UncertaintySetType.Dnorm
            if test_param.y_constraint == y_constraints(1)
                dsdata(dsk).out_struct = out_struct_max;
                dsdata(dsk).test_param = test_param;
                dsdata(dsk).summary = summary;
                dsk=dsk+1; 
            elseif test_param.y_constraint == y_constraints(2)
                dldata(dlk).out_struct = out_struct_max;
                dldata(dlk).test_param = test_param;
                dldata(dlk).summary = summary;
                dlk=dlk+1; 
            end
        else
            if test_param.y_constraint == y_constraints(1)
                esdata(esk).out_struct = out_struct_max;
                esdata(esk).test_param = test_param;
                esdata(esk).summary = summary;
                esk=esk+1; 
            elseif test_param.y_constraint == y_constraints(2)
                eldata(elk).out_struct = out_struct_max;
                eldata(elk).test_param = test_param;
                eldata(elk).summary = summary;
                elk=elk+1; 
            end
        end
    end
end
dsk=dsk-1; esk=esk-1; dlk=dlk-1; elk=elk-1; 


%% surface of where the best (relative) gains are
% figure;
% x_constraints = arrayfun(@(d) d.test_param.x_constraint, data);
% y_constraints = arrayfun(@(d) d.test_param.y_constraint, data);
% rel_gaps = arrayfun(@(d) d.summary.rel_oblivious_gap, data);
% scatter3(x_constraints, y_constraints, rel_gaps)
% xlabel('x constraint');
% ylabel('y constraint');
% zlabel('rel oblivious gap');
% title('Dnorm');
% 
% x_constraints = sort(unique(arrayfun(@(d) d.test_param.x_constraint, data)));
% y_constraints = sort(unique(arrayfun(@(d) d.test_param.y_constraint, data)));
% Z = zeros(length(x_constraints), length(y_constraints));
% for d=data
%     ii = find(x_constraints == d.test_param.x_constraint);
%     jj = find(y_constraints == d.test_param.y_constraint);
%     Z(ii,jj) = d.summary.rel_oblivious_gap;
% end
% surf(x_constraints, log(y_constraints), Z');
% xlabel('x constraint'); ylabel('y constraint');

% %% comparison to ground truth
% x_constraints = sort(unique(arrayfun(@(d) d.test_param.x_constraint, data)));
% y_constraints = sort(unique(arrayfun(@(d) d.test_param.y_constraint, data)));
% Z = zeros(length(x_constraints), length(y_constraints));
% for d=data
%     ii = find(x_constraints == d.test_param.x_constraint);
%     jj = find(y_constraints == d.test_param.y_constraint);
% 
%     param_F = submodular_fct_influence_adversary_build_param_from_problem(d.test_param.problem, d.test_param.eps);
%     x_true_vec = d.test_param.problem.x_centers_true(d.test_param.problem.var_edges);
%     
%     true_y_nom = submodular_fct_influence_adversary_vec_wrapper(x_true_vec, d.summary.y_nom, param_F);
%     true_y_robust = submodular_fct_influence_adversary_vec_wrapper(x_true_vec, d.summary.y_robust, param_F);
%     Z(ii,jj) = true_y_robust - true_y_nom;
% end
% surf(x_constraints, log(y_constraints), Z');
% xlabel('x constraint'); ylabel('y constraint');

%% plots
% y_constraints = sort(unique(arrayfun(@(d) d.test_param.y_constraint, data)));
% yc1 = y_constraints(1); %0.400
% yc2 = y_constraints(5); %3.7402
% 
% for ii=[1 5]%1:length(y_constraints)
%     inx = arrayfun(@(d) d.test_param.y_constraint == y_constraints(ii), data);
%     data_this = data(inx);
%     out_structs = [data_this.out_struct];
%     test_params = [data_this.test_param];
%     summaries = [data_this.summary];
% 
%     figure('Units','inches', ...
%         'Position',[0 0 1.6 1.5], ...
%         'PaperPositionMode', 'auto');
% 
%     gammas = [test_params.x_constraint];
%     % sort the test cases by gammas
%     [~,I] = sort(gammas);
%     gammas = gammas(I);
%     out_structs = out_structs(I);
%     test_params = test_params(I);
%     summaries = summaries(I);
% 
%     adversarial_y_nom = [summaries.adversarial_y_nom];
%     adversarial_y_robust = [summaries.adversarial_y_robust];
%     adversarial_y_expect = [summaries.adversarial_y_expect];
% 
%     max_gamma = max(gammas);
%     min_gamma = min(gammas);
%     max_influence = max( [max(adversarial_y_nom), max(adversarial_y_robust), max(adversarial_y_expect)] );
% 
% 
% 
%     plot(gammas, adversarial_y_robust, '-'); hold on;
%     plot(gammas, adversarial_y_nom, '-'); hold on;
%     plot(gammas, adversarial_y_expect, '-');
% 
%     axis([min_gamma max_gamma 0 max_influence]);
%     set(gca,...
%         'Units','normalized',... %YTick...
%         'XTick',0:1:max_gamma,...%min_gamma:3:max_gamma,...
%         'Position',[.15 .2 .75 .7],...
%         'FontUnits','points',...
%         'FontWeight','normal',...
%         'FontSize',6,...
%         'FontName','Times');
% 
%     xlabel('Adversary constraint $\Gamma$',...
%         'FontUnits','points',...
%         'interpreter','latex',...
%         'FontSize',6,...
%         'FontName','Times');
%     ylabel('Worst-case influence',...
%         'FontUnits','points',...
%         'interpreter','latex',...
%         'FontSize',6,...
%         'FontName','Times');
%     legend({'$y_{\mathrm{robust}}$','$y_{\mathrm{nom}}$','$y_{\mathrm{expect}}$'},...
%         'FontUnits','points',...
%         'interpreter','latex',...
%         'Location','northeast',...
%         'FontSize',6,...
%         'FontName','Times');
%     lgd = legend('show');
%     lgd.Position = [0.4 0.64 0.4882 0.2463];
% 
% 
%     title({'Influence as adversary power changes',sprintf('for D-norm uncertainty, C = %0.3f', y_constraints(ii))},...
%         'FontUnits','points',...
%         'FontWeight','normal',...
%         'FontSize',6,...
%         'FontName','Times');
% 
%     %print -depsc2 ../plots/dnorm-robust-vs-nom.eps
% end

%% all plots

figure('Units','inches', ...
    'Position',[0 0 3.25 2.3], ...
    'PaperPositionMode', 'auto');
set(gcf, 'Renderer', 'painters'); % so that it is vector graphics


ax1 = prepare_subplot(dsdata, 1);
ylabel(ax1, 'Worst-case influence',...
        'FontUnits','points',...
        'interpreter','latex',...
        'FontSize',6,...
        'FontName','Times');
axis(ax1, [0 15 0 0.3]);
set(ax1, 'XTick', 0:5:15);
ax2 = prepare_subplot(esdata, 2);
axis(ax2, [0 800 0 0.3]);
set(ax2, 'XTick', 0:400:800);
legend(ax2, {'$y_{\mathrm{robust}}$','$y_{\mathrm{nom}}$','$y_{\mathrm{expect}}$'},...
        'FontUnits','points',...
        'interpreter','latex',...
        'Location','northeast',...
        'FontSize',6,...
        'FontName','Times');
ax3 = prepare_subplot(dldata, 3);
axis(ax3, [0 15 0 1.6]);
set(ax3, 'XTick', 0:5:15);
ylabel(ax3, 'Worst-case influence',...
        'FontUnits','points',...
        'interpreter','latex',...
        'FontSize',6,...
        'FontName','Times');
ax4 = prepare_subplot(eldata, 4);
axis(ax4, [0 800 0 1.6]);
set(ax4, 'XTick', 0:400:800);


subplot_pos{1} = [0.13   0.5838    0.3347    0.3612];
subplot_pos{2} = [0.5703    0.5838    0.3347    0.3612];
subplot_pos{3} = [0.13    0.11    0.3347    0.3612];
subplot_pos{4} = [0.5703    0.11    0.3347    0.3612];

set(subplot(2,2,1,ax1),'Position', subplot_pos{1});
set(subplot(2,2,2,ax2),'Position', subplot_pos{2});
set(subplot(2,2,3,ax3),'Position', subplot_pos{3});
set(subplot(2,2,4,ax4),'Position', subplot_pos{4});


%print -depsc2 ../plots/synthetic-adversary-strengths.eps

function [ax] = prepare_subplot(data, pos)
    out_structs = [data.out_struct];
    test_params = [data.test_param];
    summaries = [data.summary];

%     figure('Units','inches', ...
%         'Position',[0 0 1.6 1.5], ...
%         'PaperPositionMode', 'auto');

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


    ax = axes;

    plot(ax, gammas, adversarial_y_robust, '-', 'LineWidth', 2); hold on;
    plot(ax, gammas, adversarial_y_nom, '-', 'LineWidth', 2); hold on;
    plot(ax, gammas, adversarial_y_expect, '-', 'LineWidth', 2);
    
    axis(ax, [min_gamma max_gamma 0 max_influence]);
    set(ax,...
        'Units','normalized',... %YTick...
        'XTick',0:1:max_gamma,...%min_gamma:3:max_gamma,...
        'Position',[.15 .2 .75 .7],...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',6,...
        'FontName','Times');
    
    if find([3 4] == pos)
        xlabel(ax, 'Adversary constraint $\gamma$',...
            'FontUnits','points',...
            'interpreter','latex',...
            'FontSize',6,...
            'FontName','Times');
    end
%     ylabel(ax, 'Worst-case influence',...
%         'FontUnits','points',...
%         'interpreter','latex',...
%         'FontSize',6,...
%         'FontName','Times');
%     legend(ax, {'$y_{\mathrm{robust}}$','$y_{\mathrm{nom}}$','$y_{\mathrm{expect}}$'},...
%         'FontUnits','points',...
%         'interpreter','latex',...
%         'Location','northeast',...
%         'FontSize',6,...
%         'FontName','Times');
    %lgd = legend('show');
    %lgd.Position = [0.4 0.64 0.4882 0.2463];

    if test_params(1).uncertainty_set_type == UncertaintySetType.Dnorm
        set_name = 'D-norm';
    else
        set_name = 'Ellipsoidal';
    end
    title(ax, sprintf('%s, C = %0.3f', set_name, test_params(1).y_constraint),...
        'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',6,...
        'FontName','Times');
end