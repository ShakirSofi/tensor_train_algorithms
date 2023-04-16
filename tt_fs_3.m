function [Cores] = tt_fs_3(Tm, W, ranks)
assert(ndims(Tm)==3, 'Only for 3D tensors');
szeT=size(Tm);
dT =ndims(Tm);
% Init
Cores=cell(1,dT);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
%Last Core
[core_N,Tr] = CoreN(Tm, W, ranks(3));
Cores{dT} = core_N;
% First core
n=1; 
Wn=W;
Tmr = Tr;
sze=size(Tmr);
%------------------------
notn = [1:n-1 n+1:dT-1];
Tn = permute(Tmr, [notn n dT]);
Tn = reshape(Tn, [prod(sze(notn)) sze(n) sze(dT)]);
Wn = permute(Wn, [notn n]);
Wn = reshape(Wn, [prod(sze(notn)) sze(n)]);
[cr] = grid_complete(Tn, Wn, ranks(n+1));
Cores{1} = reshape(cr, [1, size(cr)]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% solve for B: Ysub_(3) = C*B_(3)*kron(I, A.T)
A = squeeze(Cores{1});
C=squeeze(Cores{3});
S3 = row_sel(tens2mat(Tm,3).');
H = kron(eye(size(Tm,2)), A.')*S3.';
Ysub3 = tens2mat(Tm,3);
Ysub3 = Ysub3(:, W(:));
Yc = C*Ysub3;
B=Yc*pinv(H);
Cores{2} = reshape(B.',ranks(2), szeT(2), ranks(3));
end