function [VRight]  = Right_Orth(G, mu)
% right unfolding of cores;
[GR] = Right_Unfold(G);
d=length(GR);
% Size of tensor
sze = [];
for i=1:d
     sze=[sze size(G{i},2)];
end
% TT_ranks % ranks =[1 r1 r2 r3.. 1]
r = [];
r(1) = 1;
for i=1:d-1
     r=[r size(G{i},3)];
end
r(end+1)=1;
% orthogonalization using qr-factorization
for k = d:-1:mu+1
    cr = GR{k}.';
    [Q,R] = qr(cr, 0);
    GR{k} = Q.';
    r(k) = size(R,2);
    GR{k-1} = reshape(tens2mat(G{k-1},[], 3)*R, r(k-1), sze(k-1)*r(k)) ; % tensor-matrix multiplication in mode-3;
end
% right folding of cores;
VRight=Right_Fold(GR, r);
end