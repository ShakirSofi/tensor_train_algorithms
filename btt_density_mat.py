def btt_parallel(x, ranks, K):
    if x.ndim % 2 != 0:
        raise ValueError('Number of dimensions must be a multiple of 2.')

    order = x.ndim // 2 
    if order < 3:
        raise ValueError('This algorithm requires order >= 3 (tensor of dimension >= 6).')

    shape = x.shape
    row_dims = shape[:order]
    col_dims = shape[order:]

    A = []
    ranks = list(tt_rank)              # [1, r1, r2, ..., r_{N-1}, 1]

    for n in range(order):
        if n == order - 1:
            perm = list(range(order-1)) + [order-1, order] + list(range(order+1, 2*order))
            x3n = x.permute(*perm)
            # Reshape: (prod(row[:order-1]), I_{order-1} * I_order, prod(col[1:]))
            x3n = x3n.reshape(
                tt._prod(row_dims[:order-1]),
                row_dims[-1] * col_dims[0],
                tt._prod(col_dims[1:])
            )
            # Compute subspace basis
            # Un = subspace_intersection(x3n, K) 
            U, S, _ = tt.truncated_svd(x3n.reshape(tt._prod(row_dims[:order]), -1), max_rank=K)   
        else:
            perm = list(range(n+1)) + [order-1, order] + \
                    list(range(n+1, order-1)) + list(range(order+1, 2*order))
            x3n = x.permute(*perm)
            # Reshape: (prod(row[:n+1]), I_{order-1} * I_order, prod(row[n+1:-1]) * prod(col[1:]))
            x3n = x3n.reshape(
                tt._prod(row_dims[:n+1]),
                row_dims[-1] * col_dims[0],
                tt._prod(row_dims[n+1:-1]) * tt._prod(col_dims[1:])
            )
            # Compute subspace basis
            # Un = subspace_intersection(x3n, ranks[n+1])
            U, _, _ = tt.truncated_svd(x3n.reshape(tt._prod(row_dims[:n+1]), -1), max_rank=ranks[n+1])

        A.append(U)

    # First core (mode 0)
    btt_cores = [None]*order
    btt_cores[0] = A[0].reshape(1, A[0].shape[0], 1, A[0].shape[1])    # (1, I1, 1, r1)

    # Intermediate cores (modes 1 .. order-2)
    for ci in range(order - 2):
        left = A[ci].conj().T                               # (r_{ci+1}, prod_{0..ci})
        prod_prev = tt._prod(row_dims[:ci+1])
        right = A[ci+1].reshape(prod_prev, row_dims[ci+1] * ranks[ci+2])
        Wci = left @ right                                   # (r_{ci+1}, I{ci+1}*r_{ci+2})
        btt_cores[ci+1]=  Wci.reshape(ranks[ci+1], row_dims[ci+1], 1, ranks[ci+2])

    # Final core
    ci = order - 2
    left = A[ci].conj().T
    prod_prev = tt._prod(row_dims[:ci+1])
    right = A[ci+1].reshape(prod_prev, row_dims[ci+1] * K)
    W_last = left @ right
    core_last = W_last.reshape(ranks[order-1], row_dims[order-1], 1, K)
    btt_cores[order-1] = torch.einsum('ijlk,kk->ijkl', core_last, torch.diag(torch.sqrt(S)))

    return btt_cores

# cores_rec = btt_parallel(T, tt_rank, K)
# cores_rec_ahrev = [g.permute(3, 2, 1, 0).conj() for g in cores_rec[::-1]]
# middle_rec = tt.c_product(cores_rec[-1], cores_rec_ahrev[0])
# rL, d, d, rR = middle_rec.shape
# middle_rec = middle_rec.reshape(rL, d, d, rR)
# cores_rho_rec = cores_rec[:-1] +[middle_rec] + cores_rec_ahrev[1:]
# print(tt.check_tt_orthogonality(cores_rho_rec, N-1))
