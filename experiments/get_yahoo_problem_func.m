function problem_with_confidence_func = get_yahoo_problem_func(frac)
% reads in the bipartite yahoo bidding graph and returns the corresponding
% robust influence maximization problem

savefile_path = 'experiments/yahoo.mat';
if exist(savefile_path, 'file')
    load(savefile_path); % will contain prices_mat and counts_mat
else
    [prices_mat, counts_mat] = parse_yahoo_data();
    save(savefile_path, 'prices_mat', 'counts_mat');
end

if nargin < 1
    frac = 1;
end

[S,T] = size(prices_mat);
prices_mat = prices_mat(1:round(frac*S),1:round(frac*T));
[S,T] = size(prices_mat);

x_centers_true = zeros(S,T);
% for now just randomly generate
x_centers_rand = 0.8 + 0.2 * rand(S,T);
x_centers_true(prices_mat > 0) = x_centers_rand(prices_mat > 0);

real_edges = (prices_mat > 0);

num_trials = counts_mat;
num_trials_vec = num_trials(real_edges);
x_centers_true_vec = x_centers_true(real_edges);

num_heads_vec = do_binom_draw(num_trials_vec, x_centers_true_vec);
num_tails_vec = num_trials_vec - num_heads_vec;

alphas_vec = ones(size(num_heads_vec));
betas_vec = ones(size(num_heads_vec));

alphas_vec = alphas_vec + num_heads_vec;
betas_vec = betas_vec + num_tails_vec;

empirical_means_vec = alphas_vec ./ (alphas_vec + betas_vec);
x_median_vec = betainv(0.5, alphas_vec, betas_vec);

alphas = zeros(S,T); alphas(real_edges) = alphas_vec;
betas = zeros(S,T); betas(real_edges) = betas_vec;
empirical_means = zeros(S,T); empirical_means(real_edges) = empirical_means_vec;
x_median = zeros(S,T); x_median(real_edges) = x_median_vec;

function [ problem ] = problem_with_confidence(confidence)
    problem.alphas = alphas;
    problem.betas = betas;
    
    x_lower_vec = betainv(1 - (1+confidence)/2, alphas_vec, betas_vec);
    x_upper_vec = betainv((1+confidence)/2, alphas_vec, betas_vec);
    
    problem.x_lower = zeros(S,T); problem.x_lower(real_edges) = x_lower_vec;
    problem.x_upper = zeros(S,T); problem.x_upper(real_edges) = x_upper_vec;
    
    problem.x_median = x_median;

    problem.x_centers_true = x_centers_true;
    problem.x_centers_empirical = empirical_means;
    problem.real_edges = real_edges;
    
    problem.S = S;
    problem.T = T;
    
    problem.confidence = confidence;
end
problem_with_confidence_func = @problem_with_confidence;

end

function [prices_mat, counts_mat] = parse_yahoo_data()
    filepath = '../Webscope_A1/ydata-ysm-advertiser-bids-v1_0.txt';

    % we want to store an adjacency matrix of dimension (S,T)

    % the file is only 600 megs so we can do this
    %text = fileread(filepath);
    fid = fopen(filepath);
    C = textscan(fid, '%s %d %d %f %d', 'Delimiter', '\t');
    fclose(fid);

    phrases = C{2};
    accounts = C{3};
    prices = C{4};

    num_phrases = max(phrases); %phrases are 1-indexed
    num_accounts = max(accounts) + 1; %accounts are 0-indexed
    num_entries = length(phrases);

    % interpretation: find set of phrases maximally associated to advertisers
    S = num_phrases;
    T = num_accounts;

    prices_mat = zeros(S,T);
    counts_mat = zeros(S,T);
    for kk=1:num_entries
        s = phrases(kk);
        t = accounts(kk) + 1;
        prices_mat(s,t) = prices_mat(s,t) + prices(kk);
        counts_mat(s,t) = counts_mat(s,t) + 1;
    end
end

function [num_heads] = do_binom_draw(num_trials, probs)
    num_elems = length(num_trials);
    num_heads = zeros(num_elems, 1);
    for ii=1:num_elems
        num_heads(ii) = sum(rand(1,num_trials(ii))<probs(ii));
    end
end