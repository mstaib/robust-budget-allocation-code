function [ x_centers, alpha ] = random_bipartite_problem( num_channels, num_cust, num_advertisers, prob_edge )
%RANDOM_BIPARTITE_PROBLEM Summary of this function goes here
%   Detailed explanation goes here

% x_centers(s,t) = 0 iff that edge is not picked

alpha = rand(num_advertisers,1);

S = num_channels;
T = num_cust;
x_centers = zeros(S,T);

%prob_edge = 0.3;

for s=1:S
    for t=1:T
        if rand <= prob_edge
            x_centers(s,t) = 0.8 + 0.2 * rand; %bias towards high values of x
        end
    end
end

end

