function [GR] = Right_Unfold(G)
% Unfold each core as: r(n-1) X I(n)*r(n);
sz_start = size(G{1});
GR = cellfun(@(x) tens2mat(squeeze(x),1), G, 'UniformOutput',false);
GR{1} = reshape(GR{1}, [], sz_start(end));
end