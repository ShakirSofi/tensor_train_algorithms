function [Ym, Idx] = give_miss_n(Ym, p)
if nargin < 2 || isempty(p)
    p = 0.1;
end
N = ndims(Ym); 
sze = size(Ym);
Ym_n = tens2mat(Ym,N);
rand_vec = rand(1, size(Ym_n, 2));
Idx = rand_vec < p;
Ym_n(:, Idx) = nan;
Ym = mat2tens(Ym_n,sze,N); 
end