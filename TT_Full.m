function [x_full] = TT_Full(cores, ranks, sze)
d=length(cores);
if nargin < 3 || isempty(sze)
    % Size of tensor
    sze = [];
    for i=1:d
         sze=[sze size(cores{i},2)];
    end
end
if nargin < 2 || isempty(ranks)
   % TT_ranks % ranks =[1 r1 r2 r3.. 1]
   ranks = [];
   ranks(1) = 1;
   for i=1:d-1
       ranks=[ranks size(cores{i},3)];
   end
   ranks(end+1)=1;
end

%%
x_full = cores{1};
for k=2:d
   x_full = reshape(x_full, numel(x_full)/ranks(k), ranks(k));
   %size(x_full)
   cr = cores{k};
   %size(cr)
   cr = reshape(cr, ranks(k), sze(k)*ranks(k+1));
   x_full = x_full*cr;
end
x_full=reshape(x_full, sze);

end