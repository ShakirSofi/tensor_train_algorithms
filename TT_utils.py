
## TT-QST
# Author: Shakir Sofi;
# Some code is used from https://github.com/PGelss/scikit_tt
 
def add_extra_dims(matrix_list): # input M: m x n -> Mh: 1 x m x n x 1
    output = [np.expand_dims(np.expand_dims(matrix, axis=0), axis=-1) for matrix in  matrix_list]
    return output


def tt_pauli_operators(sampled_strings, matrix_map): # Efficiently represent Pauil operators in TTM-format
    combinations = [[matrix_map[char] for char in combo] for combo in sampled_strings]
    tt_cores_ops = [add_extra_dims(combo) for combo in combinations]
    return tt_cores_ops

def tt_perform_measurement(ttcores_A, tt_cores_ops): # Forward Operator (sensing map)
    "Compute Born probabilities"
    measurements = []
    for op in tt_cores_ops:
        OpA = ttmulcores(op, ttcores_A)
        redmat = ttmulcores(transpose_cores(ttcores_A, conjugate=True), OpA) # ~ pi = Tr(AH @ obs_i @A)
        meas = np.trace(cores2ten(redmat, matricize=True))
        measurements.append(meas)
    return np.real(measurements)

# Extra functions related to TT
def get_dimsrank(ttcores):
    order = len(ttcores)
    row_dims = [core.shape[1] for core in ttcores]
    col_dims = [core.shape[2] for core in ttcores]
    tt_rank = [core.shape[0] for core in ttcores] + [ttcores[-1].shape[3]]
    return order, row_dims, col_dims, tt_rank

def cores2vec(ttcores): # serialize
    vec = []
    for k in range(len(ttcores)):
        core_k = ttcores[k]
        vec_k = core_k.flatten()
        vec.append(vec_k)
    return np.concatenate(vec)

def vec2cores(vec_cores, row_dims, col_dims, tt_rank): # deserialize
    ttcores = []
    pos_i = 0
    for k in range(len(row_dims)):
        pos_f = pos_i+tt_rank[k]*row_dims[k]*col_dims[k]*tt_rank[k+1]
        core_k = vec_cores[pos_i:pos_f].reshape(tt_rank[k], row_dims[k], col_dims[k], tt_rank[k+1])
        pos_i = pos_f
        ttcores.append(core_k)
    return ttcores

def copy_cores(ttcores):
    copycores = []
    for core in ttcores:
        copycores.append(core.copy())
    return copycores

def conj_cores(ttcores, overwrite = True):
    if overwrite is False:
        copycores = copy_cores(ttcores)
    else:
        copycores = ttcores 

    conjcores = []
    for core in copycores:
        conjcores.append(np.conjugate(core))
    return conjcores

def transpose_cores(ttcores, overwrite = True, conjugate = False):
    if overwrite is False:
        copycores = copy_cores(ttcores)
    else:
        copycores = ttcores

    transpose_cores = []
    for core in copycores:
        if conjugate is True:
            core = np.conjugate(core)
        transpose_cores.append(np.transpose(core, [0, 2, 1, 3]))

    return transpose_cores

def scalar_mul(ttcores, scalar):
    assert isinstance(scalar, (int, float, complex)), 'Unsupported parameter'
    new_cores = copy_cores(ttcores)
    # multiply first core by scalar
    new_cores[0] = scalar * new_cores[0]
    return new_cores


def ttsumcores(ttcores_a, ttcores_b):

    order_a, row_dims_a, col_dims_a, tt_rank_a = get_dimsrank(ttcores_a)
    order_b, row_dims_b, col_dims_b, tt_rank_b = get_dimsrank(ttcores_b)

    assert (row_dims_a==row_dims_b) and (col_dims_a==col_dims_b), "Dimensions do not match"
    
    sumcores = []
    tt_rank_sum = [1] + [tt_rank_a[i] + tt_rank_a[i] for i in range(1, order_a)] + [1]

    for i in range(order_a):
        sumcores.append(np.zeros([tt_rank_sum[i], row_dims_a[i], col_dims_a[i], tt_rank_sum[i + 1]], dtype=complex))

        sumcores[i][0:tt_rank_a[i], :, :, 0:tt_rank_a[i + 1]] = ttcores_a[i]

        # insert core of ttcores_b
        r_1 = tt_rank_sum[i] - tt_rank_b[i]
        r_2 = tt_rank_sum[i]
        r_3 = tt_rank_sum[i + 1] - tt_rank_b[i + 1]
        r_4 = tt_rank_sum[i + 1]
        sumcores[i][r_1:r_2, :, :, r_3:r_4] = ttcores_b[i]  

    return sumcores

def ttsubcores(ttcores_a, ttcores_b):
    return ttsumcores(ttcores_a, scalar_mul(ttcores_b, -1))


def core_mul(core_1: np.ndarray, core_2: np.ndarray):
    """
    Multiplies two 4-dimensional cores of the following shapes:

        (r1 x m x n x r2) (s1 x n x p x s2)
    Returns:

        Product of shape (r1 * s1 x m x p x r2 * s2)
    """
    # Prepare cores for matrix multiplication
    c1_row = np.arange(core_1.shape[ 0], dtype = np.intp)[:, None]
    c1_col = np.arange(core_1.shape[-1], dtype = np.intp)[None, :]

    c2_row = np.arange(core_2.shape[ 0], dtype = np.intp)[:, None]
    c2_col = np.arange(core_2.shape[-1], dtype = np.intp)[None, :]

    # Index and broadcast accordingly
    core1_broad = core_1[c1_row, :, :, c1_col][:, None, :, None, :, :]
    core2_broad = core_2[c2_row, :, :, c2_col][None, :, None, :, :, :]

    contraction = core1_broad @ core2_broad

    reshape_contraction = contraction.reshape(

            core_1.shape[ 0] * core_2.shape[ 0],
            core_1.shape[-1] * core_2.shape[-1],
            core_1.shape[ 1], 
            core_2.shape[ 2]
    )
    result = reshape_contraction.transpose(0, 2, 3, 1)
    return result

def ttmulcores(ttcores_a, ttcores_b):
    order_a, row_dims_a, col_dims_a, tt_rank_a = get_dimsrank(ttcores_a)
    order_b, row_dims_b, col_dims_b, tt_rank_b = get_dimsrank(ttcores_b)
    assert (col_dims_a==row_dims_b), "Dimensions do not match"
    # multiply TT cores
    mulcores = [core_mul(core_a, core_b) for core_a, core_b in zip(ttcores_a, ttcores_b)]
    return mulcores

def cores2ten(ttcores, matricize = False):
      """
      Conversion to full format.
      """
      order, row_dims, col_dims, tt_rank = get_dimsrank(ttcores)
      if tt_rank[0] != 1 or tt_rank[-1] != 1:
          raise ValueError("The first and last rank have to be 1!")

      # reshape first core
      full_tensor = ttcores[0].reshape(row_dims[0] * col_dims[0], tt_rank[1])

      for i in range(1, order):
          # contract full_tensor with next TT core and reshape
          full_tensor = full_tensor.dot(ttcores[i].reshape(tt_rank[i], row_dims[i] * col_dims[i] * tt_rank[i + 1]))
          full_tensor = full_tensor.reshape(np.prod(row_dims[:i + 1]) * np.prod(col_dims[:i + 1]),tt_rank[i + 1])

      # reshape and transpose full_tensor
      p = [None] * 2 * order
      p[::2] = row_dims
      p[1::2] = col_dims
      q = [2 * i for i in range(order)] + [1 + 2 * i for i in range(order)]
      full_tensor = full_tensor.reshape(p).transpose(q)

      if matricize is True:
          full_tensor = full_tensor.reshape(np.prod(row_dims), np.prod(col_dims))

      return full_tensor    


def recompress_l2r_ortho(ttcores, new_tt_rank = np.inf, threshold = 1e-10, start_index = 0):
    order, row_dims, col_dims, tt_rank = get_dimsrank(ttcores)	
    if isinstance(new_tt_rank, (int, np.int32, np.int64)) or new_tt_rank == np.inf:
          new_tt_rank = [1] + [new_tt_rank for _ in range(order-1)] + [1]

    assert all(isinstance(x, int) and x > 0 for x in new_tt_rank), "tt_rank must be positive integers"
    assert len(ttcores) == len(new_tt_rank)-1, "Improper size"
    assert threshold >= 0, "Threshold must be greater or equal 0"

    recompressedcores = copy_cores(ttcores)
    end_index = order - 2
    for i in range(start_index, end_index + 1):

        # apply SVD to ith TT core
        try:
            [u, s, v] = sp.linalg.svd(
                recompressedcores[i].reshape(tt_rank[i] * row_dims[i] * col_dims[i],
                                      tt_rank[i + 1]), full_matrices=False, overwrite_a=True,
                check_finite=False)
        except:
            [u, s, v] = sp.linalg.svd(
                recompressedcores[i].reshape(tt_rank[i] * row_dims[i] * col_dims[i],
                                      tt_rank[i + 1]), full_matrices=False, overwrite_a=True,
                check_finite=False, lapack_driver='gesvd')

        # rank reduction
        if threshold != 0:
            indices = np.where(s / s[0] > threshold)[0]
            u = u[:, indices]
            s = s[indices]
            v = v[indices, :]
        if new_tt_rank[i+1] != np.inf:
            u = u[:, :np.minimum(u.shape[1], new_tt_rank[i+1])]
            s = s[:np.minimum(s.shape[0], new_tt_rank[i+1])]
            v = v[:np.minimum(v.shape[0], new_tt_rank[i+1]), :]
        # updated the core
        tt_rank[i + 1] = u.shape[1]
        recompressedcores[i] = u.reshape(tt_rank[i], row_dims[i], col_dims[i], tt_rank[i + 1])
        # shift non-orthonormal part to next core
        recompressedcores[i + 1] = np.tensordot(np.diag(s).dot(v), recompressedcores[i + 1], axes=(1, 0))

    return recompressedcores

def ttrandcores(row_dims, col_dims, tt_rank, rng = np.random.standard_normal):
    randcores = []
    for k in range(len(row_dims)):
        core = rng((tt_rank[k], row_dims[k], col_dims[k], tt_rank[k+1]))
        randcores.append(core)
    return randcores