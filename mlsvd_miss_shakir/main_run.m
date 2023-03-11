function[T, Tmiss, Srow3,  W,G, A, B, C, R1,R2,R3] =  main_run(alg)
if nargin < 1
alg = 1;
end
R1=2;R2=3;R3=3;
% generate
T = reshape(1:60, 3,4,5);
% remove some fibres, say 20% of tube fibres
[Tmiss,~,~] = give_miss_3(T, 0.3);
% Get C and  Run Grid Structured M.C
 W=~isnan(Tmiss(:,:,1));
[C, Srow3, Tred, Ysub] = give_c(Tmiss, R3);
Which_Algo = alg; % here we can change algorithms. 1-> Left Null Space Based; 2-> Principal Subspace;

if Which_Algo==1
% Algo-1
   [B]= Grid_MC(Tmiss,Tred,W,R2);
else
% Algo-2
   [B]= PSSM(Tmiss, Tred, W, R2);
end

Tmissp = permute(Tmiss, [2 1 3]);
Tredp = permute(Tred, [2 1 3]);

if Which_Algo==1
% Algo-1
   [A]= Grid_MC(Tmissp, Tredp,W',R1);
else
% Algo-2
 [A]= PSSM(Tmissp, Tredp, W',R1);
end
% get core
[G] = get_core(A, B, C, Srow3, Ysub, R1,R2,R3);
end