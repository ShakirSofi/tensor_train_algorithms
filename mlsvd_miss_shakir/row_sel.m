function [Srow] = row_sel(Y, type)
if nargin < 2
    type='NaN';
end
[I,J]= size(Y);
if strcmp(type, 'NaN')
   [r,c] = find(~isnan(Y));
elseif strcmp(type, 'zeros')
   [r,c] = find(Y~=0);
end
aL = unique(r);
bL = unique(c);

S = e_k(I, aL(1));

for i=2:length(aL)
    S = [S  e_k(I, aL(i))];
end

Srow = S'; 
end