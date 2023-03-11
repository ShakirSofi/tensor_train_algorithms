    function [Di] = get_Di(W, i)
            Di = diag(W'*e_k(size(W,1), i));
    end
