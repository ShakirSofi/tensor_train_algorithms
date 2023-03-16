function [G] = Right_Fold(GR, ranks)
% Fold each  matricized core from r(n-1) X I(n)*r(n) ---> r(n-1) X I(n) X r(n);
% ranks =[1 r1 r2 r3.. 1]
%{
sz_start = size(GR{1});
G = cellfun(@(x) tens2mat(squeeze(x),1), GR, 'UniformOutput',false);
G{1} = reshape(G{1}, [], sz_start(end));
%}
d=length(GR);
G=cell(1,d);
for i=1:d
    G{i}=reshape(GR{i}, size(GR{i},1),[], ranks(i+1));
end
G{1} = permute(G{1}, [2 1 3]);

end