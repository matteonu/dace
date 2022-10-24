import csv
from enum import auto
import dace
import numpy as np
import timeit

from dace.transformation.auto.auto_optimize import auto_optimize
from dace.sdfg import utils
from datetime import datetime
from os.path import exists
from scipy import sparse



"""
formula:  σ (S × H × W)
S: T // (T × 1L x 1R)
T: A ⊙ exp(σ(C))
C: (H x W x aL x 1R).t + (H x W x aT x 1R) (size is Arows, Acols)
1L: ones(Wcols, 1)
1R: ones(1, Wcols)
σ: ReLU activation function
A: Sparse (CSR) matrix (Arows, Acols). It is an adjacency matrix, i.e. its values are 0 and 1.
aL,aR: vectors (Wcols, 1)
H: Dense matrix (Hrows, Hcols)
W: Dense matrix (Wrows, Wcols)
⊙: Hadamard product
//: Hadamard division
t: transopose
x: Dot product (matrix multiplication)
ù

Notes:
- H, W are dense matrices.
- Hrows = Acols = Arows (the two terms of C must have the same shape)
- H x W (Arows, Wcols)
- H x W x aR (Arows, 1)
- H x W x aR x 1R (Arows, Acols);
- 1L x aL.t x W.t x H.t = (HWT x 1L.t).t = (Acols, Arows)
- Each term of C is the product of a vector (e.g. HWaL) with a vector of 1s. Thus, each term of C
    is just that vector replicated horizontally or vertically. 
    The element C(i,j) is then HWaL[i] + HWaR[j].   
- T = A ⊙ exp(σ(C)) is a sparse matrix (Arows, Acols) with the non-zeros at exactly the same places as A.
- S = T // (T × 1L x 1R) is T with every row divided by its sum
- The result is a dense matrix (Arows, Wcols).

Decomposition:
- We assume that Arows = Acols = Hrows >> Hcols = Wrows ≈ Wcols.
- We use a 2D process grid (Px, Py).
- A is split into (Px, Py) blocks. Each block is assigned to a single process.
- H is distributed twice:
  - H1 is split into (Px,) blocks. Each block is replicated in Py processes. Used as H in (A ⊙ (H x HT)).
  - H2 is split into (Py,) blocks. Each block is replicated in Px processes. Used as HT in (A ⊙ (H x HT)) and as H in (H x W).
- W, aR, aL are replicated in all processes.
Computation:
- Computing both terms of C, i.e. H x (W x aL) and H x (W x aR), costs 4*Hcols x Wcols + 4*Arows x Wcols.
- Computing T = A ⊙ (exp(σ(C)) having HWaL and HWaR costs 2*nnz 
- Computing S = T // (T × 1L x 1R) is dividing every row by its sum, so it costs 2*nnz
- S × H × W costs 2 * nnz * Hcols * Wcols.
- ReLU costs Arows * Wcols.
- Total cost is (ignoring constants) Wcols x (Arows + Wcols + nnz x HCols) + nnz. 
- Total cost is (taking into account Arows >> Hcols ≈ Wcols)  O(Arows + nnz)

"""

dctype = dace.float64
nptype = np.float64


grid = {
    #     [Px, Py]
    1:    [ 1,  1],
    2:    [ 1,  2],
    4:    [ 2,  2],
    8:    [ 2,  4],
    16:   [ 4,  4],
    32:   [ 4,  8],
    64:   [ 8,  8],
    128:  [ 8, 16],
    256:  [16, 16],
    512:  [16, 32],
}


# Each node does 28.2 GFLOPs.
# Scaling formula is for A rows is ceiling(base * sqrt(nodes) / nodes) * nodes
weak_scaling = {
    #:   ( Arows, Hcols, Wcols)
    1:   ( 20480,   128,   128),
    2:   ( 28964,   128,   128),
    4:   ( 40960,   128,   128),
    8:   ( 57928,   128,   128),
    16:  ( 81920,   128,   128),
    32:  (115872,   128,   128),
    64:  (163840,   128,   128),
    128: (231808,   128,   128),
    256: (327680,   128,   128),
    512: (463872,   128,   128),
}


# Global symbols
GArows, GAnnz, GHcols, GWcols = (dace.symbol(s) for s in ('GArows', 'GAnnz', 'GHcols', 'GWcols'))
# Local symbols
LArows, LAcols, LAnnz, LHcols, LWcols = (dace.symbol(s) for s in ('LArows', 'LAcols', 'LAnnz', 'LHcols', 'LWcols'))
# Process grid
Px, Py = (dace.symbol(s) for s in ('Px', 'Py'))
num_layers = dace.symbol('num_layers')



@dace.program
def GAT_dace(A_rowptr: dace.int32[LArows+1],
                 A_colidx: dace.int32[LAnnz],
                 A_rowidx: dace.int32[LAnnz],
                 A_data: dctype[LAnnz],
                 aR: dctype[LWcols],
                 aL: dctype[LWcols],
                 H1: dctype[LArows, LHcols],
                 #H2: dctype[LAcols, LHcols],
                 W: dctype[LHcols, LWcols]) -> dctype[LArows, LWcols]:
    
    # Process grid
    # Process grid
    parent_grid = dace.comm.Cart_create([Px, Py, 1])
    reduce_grid = dace.comm.Cart_sub(parent_grid, [False, True, False])
    h1_grid = dace.comm.Cart_sub(parent_grid, [True, False, True], exact_grid=0)
    h2_grid = dace.comm.Cart_sub(parent_grid, [False, True, True], exact_grid=0)
    bcast_grid = dace.comm.Cart_sub(parent_grid, [True, False, False])

    H2 = np.empty((LAcols, LHcols), dtype=H1.dtype)
    arr_h1 = dace.comm.Subarray((GArows, GHcols), H1, process_grid=h1_grid)
    arr_h2 = dace.comm.Subarray((GArows, GHcols), H2, process_grid=h2_grid)
    dace.comm.Redistribute(H1, arr_h1, H2, arr_h2)
    dace.comm.Bcast(H2, grid=bcast_grid)

    # HW = H x W  
    HW = H2 @ W
    HWL = H1 @ (W @ aL)
    HWR = HW @ aR

    def relu(x):
        return np.maximum(x,0)

    def exp_relu(x):
        return np.exp(relu(x))


    #C: (1L x HWL.t) + (HWR x 1R)
    values = np.zeros_like(A_data)
    for i in dace.map[0:LAnnz]:
        values[j] = np.exp(np.maximum(HWL[A_rowidx[i]] + HWR[A_colidx[i]]), 0)
    # for i in dace.map[0:LArows]:
    #     start = A_rowptr[i]
    #     finish = A_rowptr[i+1]
    #     for j in dace.map[start:finish]:
    #         values[j] = exp_relu(HWL[i] + HWR[A_colidx[j]])
    
    #S = T // (T × 1L x 1R)
        # find degree
    row_degree = np.zeros(LArows)
    for i in dace.map[0:LArows]:
        start = A_rowptr[i]
        finish = A_rowptr[i+1]
        row_degree[i] += np.sum(values[start:finish])

    dace.comm.Allreduce(row_degree, 'MPI_SUM', grid=reduce_grid)

        # divide each element by degree
    for i in dace.map[0:LArows]:
        start = A_rowptr[i]
        finish = A_rowptr[i+1]
        for j in dace.map[start:finish]:
            values[j] /= row_degree[i]


    # S x W
    out = np.empty((LArows, LWcols), dtype=nptype)

    dace.csrmm(A_rowptr, A_colidx, values, HW, out, 1, 0)

    # Reduce
    dace.comm.Allreduce(out, 'MPI_SUM', grid=reduce_grid)
    
    return np.maximum(out, 0)



#TODO normalize H after each loop
@dace.program
def GAT_dace_loop(A_rowptr: dace.int32[LArows+1],
                 A_colidx: dace.int32[LAnnz],
                 A_data: dctype[LAnnz],
                 aR: dctype[LWcols],
                 aL: dctype[LWcols],
                 H1: dctype[LArows, LHcols],
                 #H2: dctype[LAcols, LHcols],
                 W: dctype[num_layers,LHcols, LWcols]) -> dctype[num_layers,LArows, LWcols]:
    
    # Process grid
    # Process grid
    parent_grid = dace.comm.Cart_create([Px, Py, 1])
    reduce_grid = dace.comm.Cart_sub(parent_grid, [False, True, False])
    h1_grid = dace.comm.Cart_sub(parent_grid, [True, False, True], exact_grid=0)
    h2_grid = dace.comm.Cart_sub(parent_grid, [False, True, True], exact_grid=0)
    bcast_grid = dace.comm.Cart_sub(parent_grid, [True, False, False])

    H2 = np.empty((LAcols, LHcols), dtype=H1.dtype)
    out = np.empty((LArows, LWcols), dtype=nptype)


    def relu(x):
        return np.maximum(x,0)

    def exp_relu(x):
        return np.exp(relu(x))

    for i in range(num_layers):

        arr_h1 = dace.comm.Subarray((GArows, GHcols), H1, process_grid=h1_grid)
        arr_h2 = dace.comm.Subarray((GArows, GHcols), H2, process_grid=h2_grid)
        dace.comm.Redistribute(H1, arr_h1, H2, arr_h2)
        dace.comm.Bcast(H2, grid=bcast_grid)

        # HW = H x W  
        HW = H2 @ W
        HWL = H1 @ (W @ aL)
        HWR = HW @ aR

        #C: (1L x HWL.t) + (HWR x 1R)
        values = np.zeros_like(A_data)
        for i in dace.map[0:LArows]:
            start = A_rowptr[i]
            finish = A_rowptr[i+1]
            for j in dace.map[start:finish]:
                values[j] = exp_relu(HWL[i] + HWR[A_colidx[j]])
        
        #S = T // (T × 1L x 1R)
        # find degree
        row_degree = np.zeros(LArows)
        for i in dace.map[0:LArows]:
            start = A_rowptr[i]
            finish = A_rowptr[i+1]
            row_degree[i] += np.sum(values[start:finish])

        dace.comm.Allreduce(row_degree, 'MPI_SUM', grid=reduce_grid)

            # divide each element by degree
        for i in dace.map[0:LArows]:
            start = A_rowptr[i]
            finish = A_rowptr[i+1]
            for j in dace.map[start:finish]:
                values[j] /= row_degree[i]


        # S x W

        dace.csrmm(A_rowptr, A_colidx, values, HW, out, 1, 0)

        # Reduce
        dace.comm.Allreduce(out, 'MPI_SUM', grid=reduce_grid)
        H1 = relu(out)

    #END LAYERS LOOP
    return H




def GAT_npsn(A: sparse.csr_matrix,
                 H: np.ndarray,
                 W: np.ndarray,
                 aL: np.ndarray,
                 aR: np.ndarray) -> np.ndarray:
    
    HW = H @ W
    HWL = HW @ aL
    HWR = HW @ aR
    for i in range(A.indptr.size - 1):
        start = A.indptr[i]
        finish = A.indptr[i+1]
        for j in range(start, finish):
            A.data[j] = np.exp(np.maximum(HWL[i] + HWR[A.indices[j]],0))

    for i in range(A.indptr.size - 1):
        start = A.indptr[i]
        finish = A.indptr[i+1]
        degree = np.sum(A.data[start:finish])
        A.data[start:finish] /= degree

    out = A @ HW
    return np.maximum(out, 0)


def GAT_dense(A: sparse.csr_matrix,
                 H: np.ndarray,
                 W: np.ndarray,
                 aL: np.ndarray,
                 aR: np.ndarray) -> np.ndarray:
    
    HW = H @ W
    HWL = HW @ aL
    HWR = HW @ aR
    # C = HWL.T + HWR
    C = np.add.outer(HWL, HWR)
    for i in range(A.indptr.size - 1):
        start = A.indptr[i]
        finish = A.indptr[i+1]
        for j in range(start, finish):
            A.data[j] = np.exp(np.maximum(C[i, A.indices[j]],0))

    for i in range(A.indptr.size - 1):
        start = A.indptr[i]
        finish = A.indptr[i+1]
        degree = np.sum(A.data[start:finish])
        A.data[start:finish] /= degree

    out = A @ HW
    return np.maximum(out, 0)



def write_csv(file_name, field_names, values, append=True):
    write_mode = 'w'
    if append:
        write_mode = 'a'
    new_file = not exists(file_name)
    with open(file_name, mode=write_mode) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        if new_file:
            writer.writeheader()
        for entry in values:
            writer.writerow(entry)


def write_time(dtime, bench, frmwrk, nodes, sizes, time_list, file_name, field_names, append=True):
    entries = []
    sockets = MPI.COMM_WORLD.Get_size()
    for t in time_list:
        entries.append(
            dict(datetime=dtime, benchmark=bench, framework=frmwrk, nodes=nodes, sizes=sizes, time=t))
    write_csv(file_name, field_names, entries, append=append)


def csr_to_coo(rowptr: np.ndarray) -> np.ndarray:
    """ Converts CSR row-pointer representation to COO row-indices. """
    nnz = rowptr[-1]  # Is this always correct?
    row_indices = np.empty((nnz,), dtype=rowptr.dtype)

    row = 0
    for i in range(rowptr.size - 1):
        row_indices[rowptr[i]:rowptr[i+1]] = row
        row += 1
    return 


def normalize_mat(X):
    for row in X:
        row = row /np.sum(row)
    return X

def normalize_vec(x):
    return x/np.sum(x)

if __name__ == '__main__':

    from mpi4py import MPI
    commworld = MPI.COMM_WORLD
    rank = commworld.Get_rank()
    size = commworld.Get_size()

    if size not in grid:
        raise ValueError("Selected number of MPI processes is not supported.")

    file_name = "dace_cpu_{n}_nodes.csv".format(n=size)
    field_names = ["datetime", "benchmark", "framework", "nodes", "sizes", "time"]
    
    sdfg, sdfgc = (None, ) * 2
    if rank == 0:
        sdfg = GAT_dace.to_sdfg(simplify=True)
        sdfg = auto_optimize(sdfg, dace.DeviceType.CPU)
    func = utils.distributed_compile(sdfg, commworld)

    rng = np.random.default_rng(42)

    # Global sizes
    Nx, Ny = grid[size]
    NArows, NHcols, NWcols = weak_scaling[size]
    density = 0.01
    num_layers = 3


    # Global data
    A = sparse.random(NArows, NArows, density=density, format='csr', dtype=nptype, random_state=rng)
    H = normalize_mat(rng.random((NArows, NHcols), dtype=nptype))
    W = normalize_mat(rng.random((num_layers, NHcols, NWcols), dtype=nptype))


    
    aR = normalize_vec(rng.random((NWcols, 1), dtype=nptype))
    aL = normalize_vec(rng.random((NWcols, 1), dtype=nptype))

    # Local data
    cart_comm = commworld.Create_cart((Nx, Ny))
    x, y = cart_comm.Get_coords(rank)
    tx, ty = NArows // Nx, NArows // Ny
    lA = A[x*tx:(x+1)*tx, y*ty:(y+1)*ty]
    A_rowptr = lA.indptr.copy()
    A_colidx = lA.indices.copy()
    A_data = lA.data.copy()
    H1 = H[x*tx:(x+1)*tx, :].copy()
    H2 = H[y*ty:(y+1)*ty, :].copy()

    out = np.ndarray((tx, NWcols), dtype=nptype)

    if rank == 0:
        print(f"##### GAT #####\nGlobal Sizes: {weak_scaling[size]}\nGrid: {grid[size]}""", flush=True)
    
    # runtimes = timeit.repeat(
    #     """out[:] = func(A_rowptr=A_rowptr, A_colidx=A_colidx, A_data=A_data, aL=aL, aR=aR, H1=H1, H2=H2, W=W, Px=Nx, Py=Ny,
    #                      LArows=tx, LAcols=ty, LAnnz=A_data.size, LHcols=NHcols, LWcols=NWcols); commworld.Barrier()
    #     """,
    #     setup="commworld.Barrier()",
    #     repeat=10,
    #     number=1,
    #     globals=locals()
    # )


    runtimes = timeit.repeat(
        """out[:] = func(A_rowptr=A_rowptr, A_rowidx=A_rowidx, A_colidx=A_colidx, A_data=A_data, H1=H1, W1=W, W2=W2, aL=aL, aR=aR
                         num_layers=num_layers, GArows=NArows, GAcols=NArows, GHcols=NHcols,
                         LArows=tx, LAcols=ty, LAnnz=A_data.size, LHcols=NHcols, LWcols=NWcols,
                         Px=Nx, Py=Ny); commworld.Barrier()
        """,
        setup="commworld.Barrier()",
        repeat=10,
        number=1,
        globals=locals()
    )

    if rank == 0:
        print(f"Median total runtime: {np.median(runtimes)} seconds", flush=True)
        write_time(str(datetime.now()), "GAT", "dace_cpu", size, weak_scaling[size], runtimes, file_name, field_names, append=True)
