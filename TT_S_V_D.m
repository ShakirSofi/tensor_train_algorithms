function [Ucore] = TT_S_V_D(X, ranks)
 d = ndims(X); %order of tensor
 % ranks =[1 r1 r2 r3.. 1]
 sze = size(X);
 Ucore = cell(1, d); 
 Mk = tens2mat(X, [1]);
 for k = 2:d
     [Uk, Sk, Vk] =  svds(Mk, ranks(k));
     Ucore{k-1} = reshape(Uk, [ranks(k-1), sze(k-1), ranks(k)]);
     Mk =  reshape(Sk*Vk', [ranks(k)*sze(k) prod(sze(k+1:d))]);
 end
 Ucore{d} = reshape(Mk, [ranks(end-1) sze(end) ranks(end)]);
end