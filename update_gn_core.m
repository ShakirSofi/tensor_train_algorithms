function [Gn] = update_gn_core(G, T, n)
d = ndims(T);
sze = size(T);
% TT_ranks % ranks =[1 r1 r2 r3.. 1]
ranks = [];
ranks(1) = 1;
for i=1:d-1
     ranks=[ranks size(G{i},3)];
end
ranks(end+1)=1;
v_T = reshape(T, [], 1); % Vectorization of input tensor
% Mode splitting
[G_nl,G_nr]=G_nmode_devide(G,n);

if n==1  % For G{1}
    GL_m = G_nl; %reshape(G_nl, [], 1);
    GR_m = reshape(G_nr, ranks(1), []);
else
    GL_m = reshape(G_nl, [], ranks(n));
    GR_m = reshape(G_nr, ranks(n+1), []);
end    
    FME = kron(GR_m', eye(sze(n))); %I_n: size(sze(n)); Frame matrix 
    FME = kron(FME, GL_m);
    x0 = reshape(G{n}, [],1);
    vh = solve_LS_miss(FME, v_T, x0, [], []);
    Gn = reshape(vh, size(G{n}));
end