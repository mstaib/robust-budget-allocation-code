function [h] = plot_bipartite( A )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

% adapted from https://www.mathworks.com/matlabcentral/answers/277484-constructing-a-bipartite-graph-from-0-1-matrix
[m,n] = size(A); %adjacency matrix
big_a = [zeros(m,m), A; A', zeros(n,n)];

g = graph(big_a);
h = plot(g);
h.XData(1:m) = 1;
h.XData((m+1):end) = 2;
h.YData(1:m) = linspace(0,1,m);
h.YData((m+1):end) = linspace(0,1,n);
end

