function [Ym, r,c] = give_miss_3(Ym, p)
if nargin < 2 || isempty(p)
    p = 0.1;
end
Index    = randperm(numel(Ym(:,:,1)), ceil(numel(Ym(:,:,1))*p)); 
[r, c] = ind2sub(size(Ym(:,:,1)), Index);

  for i=1:length(r)
      Ym(r(i),c(i),:)=nan;
  end  
  
end