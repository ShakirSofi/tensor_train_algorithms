function [x_full] = TT_Full(cores, ranks, sze)
d=length(cores);
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