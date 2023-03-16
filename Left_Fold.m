function [G] = Left_Fold(GL, ranks)
% Fold each  matricized core from r(n-1)*I(n) X r(n) ---> r(n-1) X I(n) X r(n); 
d=length(GL);
G=cell(1,d);
for i=1:d
    G{i}=reshape(GL{i}, ranks(i),[], size(GL{i},2));
end
G{end} = permute(G{end}, [1 3 2]);
%}
end