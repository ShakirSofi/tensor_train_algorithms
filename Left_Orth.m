function [ULeft]  = Left_Orth(G, mu)
% left unfolding of cores;
[GL] = Left_Unfold(G);
d=length(GL);
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
for k=1:mu-1
    cr = GL{k};
    [Q,R] = qr(cr, 0);
    GL{k} = Q;
    r(k+1) = size(R,2);
    GL{k+1} = reshape(tens2mat(G{k+1},[], 1)*R, r(k+1)*sze(k+1), r(k+2)) ; % tensor-matrix multiplication in mode-1;
end
ULeft=GL;
end