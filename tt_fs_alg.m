function[Cores] = tt_fs_alg(T, ranks)
% ttranks  =[1,r1,r2,...1]
% T: Tensor with missing last mode fibres 
%% Init
szeT = size(T);
dT=ndims(T); 
Cores=cell(1,dT);
idx = repmat({':'}, 1, dT - 1);
W = ~isnan(squeeze(T(idx{:},1)));
ranksT = ranks;
%% Last Core
[Cores{dT}, Tr_n] = CoreN(T, W, ranksT(dT));

%% Other cores
Wn=W;
sze_n=size(Tr_n);
d_n = dT;

% Loop over d-2 cores--excluding last and next to last core

for n=1:dT-2
    notn = 2:d_n-1;
    Tn = permute(Tr_n, [notn 1 d_n]);
    Tn = reshape(Tn, [prod(sze_n(notn)) sze_n(1) sze_n(d_n)]);
    Wn = permute(Wn, [notn 1]);
    Wn = reshape(Wn, [prod(sze_n(notn)) sze_n(1)]);
    [cr] = grid_complete(Tn, Wn, ranksT(n+1));
    Cores{n} = reshape(cr,ranksT(n), szeT(n), ranksT(n+1));
    % compress in n-th mode
    Tr_n = tmprod(rep0(Tr_n),cr.', 1);
    Tr_n(Tr_n==0)=NaN;
    sze_n=size(Tr_n);
    s12 = prod(sze_n(1:2));
    sze_n(1:2)=[];
    sze_n = [s12 sze_n];
    Tr_n = reshape(Tr_n, sze_n);
    % update
    d_n = ndims(Tr_n);
    idx = repmat({':'}, 1, d_n - 1);
    Wn = ~isnan(squeeze(Tr_n(idx{:},1)));
end
Tr_n(isnan(Tr_n))=0;
Cores{dT-1} = reshape(Tr_n,ranksT(dT-1), szeT(dT-1), ranksT(dT));
 % refinement of dT-1 core
[G_nl,G_nr]=G_nmode_devide(Cores,dT-1);
GL_m = reshape(G_nl, [], ranks(dT-1));
GR_m = reshape(G_nr, ranks(dT), []);
FME = kron(GR_m', eye(szeT(dT-1))); %I_n: size(sze(n)); Frame matrix 
FME = kron(FME, GL_m);
x0 = reshape(Cores{dT-1}, [],1);
b_known = reshape(T,[], 1);
[core_opt,~, ~,~] = solve_LS_miss(FME, b_known, x0, [], []);
Cores{dT-1} = reshape(core_opt, size(Cores{dT-1}));
end