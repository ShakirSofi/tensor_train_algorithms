function [M] = iReshape(N, axis, k)
[i, j] = size(N);
if (mod(i, k)==0 && axis==0)
   M = {};
   for ki=1:k:i
       M{end+1} = N(ki:k+ki-1,:);
   end
   M  = cat(2,M{:});
elseif (mod(j, k)==0 && axis==1)
   M = {};
   for ki=1:k:j
       M{end+1} = N(:, ki:k+ki-1);
   end
   M  = cat(1,M{:});
else
    error('Cannot reshape to a given size, change axis or k')
end