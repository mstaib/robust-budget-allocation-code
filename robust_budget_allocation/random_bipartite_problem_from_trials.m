function [ out_struct, problem_with_confidence_func ] = random_bipartite_problem_from_trials( num_channels, num_cust, prob_edge, max_trials )
%RANDOM_BIPARTITE_PROBLEM Summary of this function goes here
%   Detailed explanation goes here

% x_centers(s,t) = 0 iff that edge is not picked
%max_trials = 15;
%confidence = 0.95;

S = num_channels;
T = num_cust;
x_centers_true = zeros(S,T); % the coin flip probabilities

for s=1:S
    for t=1:T
        if rand <= prob_edge
            x_centers_true(s,t) = 0.8 + 0.2 * rand; %bias towards high values of x
        end
    end
end

real_edges = (x_centers_true > 0);

% get particular alphas and betas for one random draw from ~ x_centers
alphas = ones(S,T);
betas = ones(S,T);
num_trials = randi(max_trials, S, T);

num_heads = binornd(num_trials, x_centers_true);
num_tails = num_trials - num_heads;

alphas = alphas + num_heads;
betas = betas + num_tails;

empirical_means = alphas ./ (alphas + betas);
x_median = betainv(0.5, alphas, betas);

% get out_struct for any arbitrary confidence level
function [ problem ] = problem_with_confidence(confidence)
    problem.alphas = alphas;
    problem.betas = betas;
    
    
    problem.x_lower = betainv(1 - (1+confidence)/2, alphas, betas);
    problem.x_upper = betainv((1+confidence)/2, alphas, betas);
    problem.x_median = x_median;

    problem.x_centers_true = x_centers_true;
    problem.x_centers_empirical = empirical_means;
    problem.real_edges = real_edges;
    problem.var_edges = real_edges;
    
    problem.S = S;
    problem.T = T;
    
    problem.confidence = confidence;
end

problem_with_confidence_func = @problem_with_confidence;

% empirical_means = alphas ./ (alphas + betas);
% x_lower = betainv(1 - confidence, alphas, betas);
% x_upper = betainv(confidence, alphas, betas);



out_struct.alphas = alphas;
out_struct.betas = betas;
%out_struct.x_lower = x_lower;
out_struct.x_centers_true = x_centers_true;
out_struct.x_centers_empirical = empirical_means;
out_struct.x_median = x_median;
%out_struct.x_upper = x_upper;
out_struct.real_edges = real_edges;

out_struct.S = S;
out_struct.T = T;

end
