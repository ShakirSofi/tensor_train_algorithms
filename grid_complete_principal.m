function [A, output] = grid_complete_principal(T, W, R)
% slices T(i,:,:) are used.
    sliceind = find(sum(W,2) >= R);
    nbrows = sum(W(sliceind,:),2);
    bases_dim = R + size(T, 2) - nbrows;
    col_ind = cumsum([1; bases_dim]);
    nb_entries = (R * nbrows) + (size(T, 2) - nbrows);
    entries_ind = cumsum([1; nb_entries]);
    indi = zeros(entries_ind(end) - 1, 1);
    indj = zeros(entries_ind(end) - 1, 1);
    vals = zeros(entries_ind(end) - 1, 1);

    if nargout > 1
        output.slice_cond = zeros(length(sliceind), 1);
    end

    for i = 1:length(sliceind)
        % Compute a basis for the principal subspace of the sampled rows of
        % each mode-1 slice.
        ind = sliceind(i);
        slice = squeeze(T(ind, W(ind,:), :));
        [Ui,Si,~] = svd(slice, 'econ');
        if nargout > 1
            output.slice_cond(i) = Si(1,1) / Si(R,R);
        end
        % Also add standard basis vectors corresponding to the missing rows
        % in each slice
        [tmpj, tmpi] = meshgrid(col_ind(i):col_ind(i)+R-1, find(W(ind,:)));
        vals(entries_ind(i):entries_ind(i+1)-1) = [tens2vec(Ui(:,1:R)); ...
                                         ones(size(T,2) - nbrows(i), 1)];
        indi(entries_ind(i):entries_ind(i+1)-1) = [tmpi(:); find(~W(ind,:)).'];
        indj(entries_ind(i):entries_ind(i+1)-1) = [tmpj(:); ...
                                         (col_ind(i) + R:col_ind(i+1) -1).'];
    end

    U = sparse(indi, indj, vals, size(T, 2), col_ind(end)-1);
    % Accuracy in the exact case is on the order of 10^-9 when using svds.
    % Using svd on a dense matrix makes it about as accurate as the
    % nullspace approach.
    [A, ~, ~] = svds(U, R);
    output = struct;
end
