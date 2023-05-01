function [Cores] = tt_tensor2mytt(tt_G)
assert(isa(tt_G, 'tt_tensor'), 'Given tensor is not in tt_tensor format')

r = tt_G.r;
d = tt_G.d;
ps = tt_G.ps;
ttcr = tt_G.core;
n = tt_G.n;

Cores = cell(1,d);

for i=1:d
    Cores{i} = reshape(ttcr(ps(i):ps(i+1)-1), r(i), n(i), r(i+1));
end