function [t] = mytt2tt_tensor(G)
d = length(G);

% TT_ranks % ranks =[1 r1 r2 r3.. 1]
r = [];
r(1) = 1;
for i=1:d-1
   r=[r size(G{i},3)];
end
r(end+1)=1;

%%%%%
% Size of tensor
sze = [];
for i=1:d
     sze=[sze size(G{i},2)];
end
%%% Constructor;
t = tt_tensor;
t.d=d;
n= sze;
t.n=n;
t.r=r;
ps=cumsum([1;t.n.*t.r(1:d).*t.r(2:d+1)]); 
t.ps=ps;
crs = cellfun(@(x) x(:), G, 'UniformOutput', false);
core = cat(1,crs{:});
t.core=core;
end