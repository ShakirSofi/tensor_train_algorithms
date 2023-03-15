function G=initcoreten(sze,ranks)
% receive the original tensor size and ranks 
% return cell G contains tt random core tensors
d=numel(sze);
G=cell(1,d);
for i=2:d-1
    G{i}=randn(ranks(i-1),sze(i),ranks(i));
end
G{1}=randn(1,sze(1),ranks(1));
G{d}=randn(ranks(d-1),sze(d),1);
for i=1:d
    G{i}=G{i}./max(abs(G{i}(:)));
end
end