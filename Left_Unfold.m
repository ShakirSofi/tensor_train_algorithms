function [GL] = Left_Unfold(G)
% Unfold each core as: r(n-1)*I(n) X r(n);
sz_end = size(G{end});
GL = cellfun(@(x) tens2mat(x,[], 3), G, 'UniformOutput',false);
GL{end} = reshape(GL{end}, sz_end);
end