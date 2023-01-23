function [Ucore] = TT_S_V_D(X, ranks)
 d = ndims(X); %order of tensor
 sze = size(X);
 r = zeros(d+1, 1);
 r(1) = 1;
 r(end) = 1;
 r(2:end-1)=ranks;
 Ucore = cell(1, d); 
 Mk = tens2mat(X, [1]);
 for k = 2:d
     [Uk, Sk, Vk] =  svds(Mk, r(k));
     Ucore{k-1} = reshape(Uk, [r(k-1), sze(k-1), r(k)]);
     Mk =  reshape(Sk*Vk', [r(k)*sze(k) prod(sze(k+1:d))]);
 end
 Ucore{d} = reshape(Mk, [r(end-1) sze(end) r(end)]);
end