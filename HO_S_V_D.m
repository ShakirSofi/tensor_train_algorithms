function [Uk, Gk] = HO_S_V_D(X, ranks)
if nargin<2
    ranks = [1, 1, 1];
end
sz = size(X);
Gk = X;
Uk={};
n=ndims(X);

if ranks(end)==sz(end)
    nd = n-1;
else
    nd =n;
end

for k=1:nd
    Gk = tens2mat(Gk, [k]);
    [Uk{k}, Sk,Vk]=svds(Gk, ranks(k));
    sz(k) = ranks(k);
    Gk = mat2tens(Sk*Vk', sz, [k]);
end
end