# Implements layer norm in triton

import torch
import torch.nn as nn

import triton
from triton import language as tl
from triton import autotune, Config

@autotune(
    configs=[
        Config({}, num_warps=1),
        Config({}, num_warps=2),
        Config({}, num_warps=4),
        Config({}, num_warps=8),
        Config({}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def _layer_norm_fwd_fused(
    X, #input pointer
    Y, #output pointer
    W,
    B,
    Mean, #pointer to the mean
    RStd, #pointer to the 1/std
    stride, #how much to skip to get to the next row
    N, #Number of columns in X
    eps, #avoid division by zero
    BLOCK_SIZE: tl.constexpr
):
    """
        Layer Norm applied to embeddings (Not only embedding)
        Mean: A 1d tensor with #rows-dimension
    """
    row = tl.program_id(0)
    X += row * stride # this makes sure we loads the correct row X
    Y += row * stride
    # WE CANNOT DO:
    # selected_x = tl.load(X + stride)
    # BECAUSE tl.any_op is run through threads in parallel and we do not have that many threads in a block
    # X_LOAD have size = block_size but we can make it to N (but no beneifit, only extra ___ memory usage)
    # TO DO SO: X_LOAD = tl.zeros([N], dtype=tl.float32)
    # IN THE FOR LOOP X_LOAD += off -> X_LOAD += tl.load(X + off + tl.arange(0, BLOCK_SIZE)), extra operations also
    # ---
    # _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32) #reduce the dtype to bfloat
    # # RUNNING_MEAN = 0
    # mean = 0
    # for off in range(0, N, BLOCK_SIZE): # off=0, BLOCK_SIZE, 2*BLOCK_SIZE, ...
    #     # WE ARE LOOPING THROUGH ALL COLUMNS TO LOAD A ROW BUT WITH THREADS = BLOCK_SIZE
    #     # off = 0 + [0,1,..,BLOCK_SIZE-1] => [0,1,..,BLOCK_SIZE-1], BLOCK_SIZE + [0,1,..,BLOCK_SIZE-1] => [BLOCK_SIZE, BLOCK_SIZE+1, ..., 2*BLOCK_SIZE-1] ...
    #     cols = off + tl.arange(0, BLOCK_SIZE)
    #     mask = cols < N
    #     # A thread with idx loading X with mask=False would not through error - rather load from other[idx]
    #     # X + cols => ptr_starting, ptr_starting+memory_address, ..., 
    #     # I think the increment memory_address on the basis of data X contains -> int4, int8
    #     x_chunk = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    #     _mean += x_chunk
    #     # CAN DO BUT THE LAST ONE IS BETTER AS WE ONLY HAVE TO CALL THE SAME NUMBER OF THREADS ONLY ONE
    #     # RUNNING_MEAN = tl.sum(_mean, axis=0) / N
    #     # [0,1,..,BLOCK_SIZE-1]
    #     # THIS STARTEGY IS WRONG BECAUSE we will not be subtracting the correct mean till we reaches the last iteration
    #     # iter: [x1, x2, ..., x_block_size-1] - sum([x1, x2, ..., x_block_size-1]) / N
    #     # iter_next: [x_block_size, x_block_size+1, ..., 2*x_block_size-1] - sum([x1+x_block_size, x2+x_block_size+2, ..., x_block_size-1+2*x_block_size-1]) / N
    #     # _VAR = tl.where(X_LOAD - tl.sum(_MEAN, axis=0) / N)
    # mean = tl.sum(_mean, axis=0) / N

    # # WE CANNOT FIND VARIANCE WHILE BEING IN ABOVE LOOP
    # _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # for off in range(0, N, BLOCK_SIZE):
    #     cols = off + tl.arange(0, BLOCK_SIZE)
    #     mask = cols<N
    #     x_chunk = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    #     diff = tl.where(mask, x_chunk - mean, 0.0)
    #     _var += diff * diff # element wise multiplication
    # var = tl.sum(_var, axis=0) / N
    # rstd = 1 / tl.sqrt(var + eps)
    # ---

    # # Store mean/rstd of a row to the correct idx
    # var = tl.sum(_var, axis=0) / N
    # rstd = 1 / tl.sqrt(var + eps)
    # # Write mean / rstd
    # tl.store(Mean + row, mean)
    # tl.store(Rstd + row, rstd)

    # # This implementation wastes registers
    # mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # m2 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # count = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # for off in range(0, N, BLOCK_SIZE):
    #     cols = off + tl.arange(0, BLOCK_SIZE)
    #     mask = cols < N
    #     # X[row, off + tl.arange(0, BLOCK_SIZE)]
    #     # e.g. for off = BLOCK_SIZE (2nd iteration) 
    #     # X[row, BLOCK_SIZE + [0, ..., BLOCK_SIZE]] => X[row, [BLOCK_SIZE, ..., 2*BLOCK_SIZE]]
    #     x_chunk = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    #     # chunk_count = BLOCK_SIZE (but mask might be off, and we may be considering elements less than BLOCK_SIZE_N)
    #     # chunk_count = sum(mask)
    #     count_chunk = tl.sum(mask.to(tl.float32)) # cannot directly sum the pointer
    #     delta = x_chunk - mean
    #     # new_mean = old_mean * [1, ..., 1] # the last is one if cols < N
    #     # mean = [mean1, mean2, mean2, ..., mean{BLOCK_SIZE}]
    #     # mean * ([1, ..., 1] (of length BLOCK_SIZE_N) / ( 1 + [1, ..., 1] (of length BLOCK_SIZE_N) ) )
    #     mean += delta * (mask.to(tl.float32) / (count + mask.to(tl.float32)))
    #     delta2 = x_chunk - mean
    #     m2 += mask * delta * delta2
    #     count += count_chunk
        
    # variance = m2 / N
    # rstd = 1 / tl.sqrt(variance + eps)
    # mean_scaler = tl.sum(mean, axis=0) / N
    # variance_scaler = tl.sum(m2, axis=0) / N
    # rstd_scalar = 1 / tl.sqrt(variance_scaler + eps)
    # tl.store(Mean + row, mean_scaler)
    # tl.store(RStd + row, rstd_scalar)

    # Scalar accumulators (not vectors)
    mean = 0.0
    m2 = 0.0
    count = 0.0
    
    # This implementatation uses:
    # Vector reduction - tl.sum (rather than loading this data into vector)
    # So leads to avoiding intermediate vector storage
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        # , cache_modifier=".ca"
        x_chunk = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        count_chunk = tl.sum(mask.to(tl.float32), axis=0)
        # [x4, x5, x6, x7] - mean = [x4, x5, x6, x7-mean]
        # x7 > cols
        # [x4, x5, x6, 0-mean] = [x4, x5, x6, -mean]
        delta = x_chunk - mean
        # Why we need mask?
        # [x4, x5, x6, -mean] * [1, 1, 1, 0] = [x4, x5, x6, 0] 
        # count = 4 (from previous loop = [x0, x1, x2, x3])
        # count * count_check = 4 * 3 = prev_elements_done * this iteration
        mean += tl.sum(delta * mask.to(tl.float32) / (count * count_chunk))
        delta2 = x_chunk - mean
        m2 += tl.sum(delta * delta2 * mask)
        count += count_chunk

    variance = m2 / count
    rstd = 1 / tl.sqrt(variance + eps)
    

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x_chunk = tl.load(X+cols, mask=mask, other=0.0).to(tl.float32) # This was just a pointer before tl.float32 (values)
        w = tl.load(W+cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B+cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x_chunk - mean) * rstd # Cannot use Mean and Rstd as they are pointers
        # Storing in y
        y = x_hat * w + b
        tl.store(Y+cols, y, mask=mask)


@triton.jit
def _layer_norm_bwd_dx_fused(
    DX, # pointer to the input gradient => dL/dx
    DY, # pointer to the output gradient => dL/dy, [B, N] — Incoming gradient from next layer
    DW, # pointer to the partial sum of weights gradient, [G, N] where 0 < G < GROUP_SIZE_M-1
    DB, # pointer to the partial sum of biases gradient
    N, # number of columns in X
    X, # pointer to the input => x
    W, # pointer to the weights
    Mean, # mean of each embedding : [B,]
    RStd, # Std ...
    stride,
    Lock, #pointer to the lock and counter = [LOCK[0], LOCK[1], ..., LOCK[GROUP_SIZE_M-1], COUNT[0], COUNT[1], ..., COUNT[GROUP_SIZE_M-1]]
    GROUP_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    # GROUP_SIZE_M = 4 <- decides the jump between the elements which aquires the same lock
    # row = 1 => lock_id = 1
    # row = 5 => lock_id = 1
    # row % GROUP_SIZE_M => groups rows in lock_id for each jump of size = GROUP_SIZE_M
    # the #unique lock_id = GROUP_SIZE_M
    # lock_id_set = {0, 1, 2, ..., GROUP_SIZE_M-1}
    # Need lock other wise multiple threads will be writing the same location, dw.
    lock_id = row % GROUP_SIZE_M # This could also be called group_id
    # LOCK is a pointer to the vector of size = GROUP_SIZE_M * 2
    # LOCK += lock_id => [0,.., lock_id, ..., GROUP_SIZE_M-1] <- starting of LOCK has LOCKS only
    Lock += lock_id
    # LOCK[0] and COUNT[0]
    # As, the LOCK = [LOCK[0], LOCK[1], ..., LOCK[GROUP_SIZE_M-1], COUNT[0], COUNT[1], ..., COUNT[GROUP_SIZE_M-1]]
    Count = Lock + GROUP_SIZE_M # LOCK[0] -> COUNT[0] or LOCK[1] -> COUNT[1]
    # DW = [[dwg11, dwg12, ..., dwg1n], [dwg21, dwg22, ..., dwg2n], ... [dwg{group_size_m-1}1, dwg{group_size_m-1}2, ..., dwg{group_size_m-1}n]]
    # DW += lock_id + N => DW[lock_id+N] => DW[dwg21] or DW[dwg11] -> DW[dwg11] + tl.arange(0, BLOCK_SIZE_M)
    DW = DW + lock_id * N + cols
    DB = DB + lock_id * N + cols

    # GETTING DATA TO THE SRAM: For now it is shared memory
    # SRAM: 
    # Static: SRAM retains data as long as power is supplied to the memory chip, meaning it doesn't require periodic refreshing. 
    # Random Access: It can access any memory location in a fixed amount of time, regardless of the address.
    # Just a fact: As a programmer we should care about not loading data redundantly - some specific thread should do that
    # We can control which thread will load the data but not when (may be the thread 3 executes the program first, but thread 0 is supposed to load the data)
    # We use __syncthreads(), such as
    """
    __shared__ float tile[32][32];

    if (threadIdx.x == 0) {
        tile[0][0] = global_data[...];  // Only one thread loads
    }

    __syncthreads();  // <=== BARRIER

    float val = tile[0][0];  // All threads read it after it's ready
    """
    x = tl.load(X+cols, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY+cols, mask=mask, other=0.0).tol(tl.float32)
    w = tl.load(W + cols, mask=mask).to(tl.float32) # Use for X
    mean = Mean + row
    rstd = RStd + row

    # Computing dx
    x_hat = (x - mean) * rstd
    dyw = dy * w # that is element wise multiplication
    ## correction terms
    c1 = tl.sum(x_hat * dyw, axis=0) / N
    c2 = tl.sum(dyw, axis=0) / N
    dx = (dyw - (x_hat * c1 + c2)) * rstd
    # Write dx
    tl.store(DX + cols, dx, mask=mask)
    # Accumulate partial sums for dw/db
    partial_dw = (dy * x_hat).to(w.dtype)
    partial_db = (dy).to(w.dtype)
    # Spinlock: this is inefficient as every thread without lock is just burning while loops, I might think about remvoving it
    # Keep checking on Lock (pointing to a memory address) if equal = 0, then change to 1
    # until then other thread can do nothig
    # If Lock == 0, CAS write 1 and return 0 (CAS returns data stored at the pointer before opration, like updating), 
    # the current thread acquires the lock and moves out of the loop as 0 == 1 -> False
    # If Lock == 1, CAS fails to update return 1, 1 == 1 -> True. stay in loop
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    # Load the Count
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        # Say Row 1 (any row could come first)
        # tl.store(DW, partial_dw, mask=mask), this makes sure we writes the first grad
        tl.atomic_xchg(Count, 1)
    else:
        # Why are we not directly accumulating in DW?
        # Dw += partial_dw
        # Because each thread will be writing DW -> race condition
        partial_dw += tl.load(DW, mask=mask)
        partial_db += tl.load(DB, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    tl.store(DB, partial_db, mask=mask)

    # need a barrier to ensure all threads finished before
    # releasing the lock
    tl.debug_barrier()

    # Release the lock
    tl.atomic_xchg(Lock, 0)
    ...

@triton.jit
def _layer_norm_bwd_dwdb(
    DW,
    DB,
    FINAL_DW,
    FINAL_DB,
    GROUP_SIZE_M, # GROUP_SIZE_M
    N, # Number of cols in dW or W
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr # We doing tiling here
):
    # We are only launching the number of programs = Number of cols / BLOCK_SIZE_N
    # Rows are handles using the for loop
    pid = tl.program_id(0) #lock_id
    # BLOCK0[thread-1, ..., thread-{BLOCK_SIZE_N}], BLOCK1[thread-1, ..., thread-{BLOCK_SIZE_N}], ...
    # pid = 0 -> cols = 0 * [thread-1, ..., thread-{BLOCK_SIZE_N}] + [thread-1, ..., thread-{BLOCK_SIZE_N}] = [thread-1, ..., thread-{BLOCK_SIZE_N}] <- maps the block0
    # pid = 1 -> cols = 1 * [thread-1, ..., thread-{BLOCK_SIZE_N}] + [thread-1, ..., thread-{BLOCK_SIZE_N}] = [2*thread-1, ..., 2*thread-{BLOCK_SIZE_N}] <- maps the block1
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # LOOPING over set of groups of size = BLOCK_SIZE_M
    for i in range(0, GROUP_SIZE_M, BLOCK_SIZE_M):
        # 0 + [0, 1, ..., BLOCK_SIZE_M] = [0, 1, ..., BLOCK_SIZE_M]
        # 2 + [0, 1, ..., BLOCK_SIZE_M] = [2, 3, ..., BLOCK_SIZE_M + 2]
        block_rows = i + tl.arange(0, BLOCK_SIZE_M)
        # rows[:, None] -> (shape,) -> (shape, 1) # An extra axis at the end
        # cols[None, :] -> (shape,) -> (1, shape)
        mask = (block_rows[:, None] < GROUP_SIZE_M) & (cols[None, :] < N)
        # Starting element of each row: the row we have above are the rows of the block in the grid
        # [2, 3, ..., BLOCK_SIZE_M + 2] * N := [2, 3, ..., BLOCK_SIZE_M + 2] * 4 = [8, 12, ]
        offs = block_rows[:, None]
        dw += tl.load(DW + offs, mask=mask, other=0.0).to(tl.float32)
        db += tl.load(DB + offs, mask=mask, other=0.0).to(tl.float32)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)
    tl.store(FINAL_DB + cols, sum_db, mask=cols < N)


class LayerNorm(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x,
        normalized_shape,
        weight, # [x.shape[-1],]
        bias, # [x.shape[-1],]
        eps
    ):
        """
            weight: A torch.Tensor with requires_grad = True
            bias: A torch.Tensor with requires_grad = True
        """
        # Reshape x: [1, N] if it is not a 2d tensor
        # Can this handle the shapes like - B, N, D
        # If not, just launch kernels for each batch
        x_arg = x.reshape(-1, x.shape[-1])
        y = torch.empty_like(x)
        M, N = x_arg.shape
        stride = N
        # Mean
        Mean = torch.empty((M,), dtype=x.dtype, device=x.device) # it is fast to initialize the torch.empty
        RStd = torch.empty((M,), dtype=x.dtype, device=x.device)

        # Less than 64KB per feature: enqueue fused kernel
        # Gives the number of elements with element_size = x.element_size can fit in 64KB
        # 64KB is a hardware limit and should be improved (according to hardware)
        MAX_FUSED_SIZE = 65536 // x.element_size()
        # triton.next_power_of_2: 8 < 9 -> 16, 16 < 19 -> 32
        BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        # Number of elements > BLOCK_SIZE or threads per block : !!problem
        if N > BLOCK_SIZE:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        # Number of blocks we need: BLOCK_SIZE//256
        # max(Number of blocks we need, 1) -> if BLOCK_SIZE < 256, BLOCK_SIZE // 256 = 0, since we cannot have 0 blocks
        # min(max(BLOCK_SIZE // 256, 1), 8): we will not like to have more than 8 blocks
        # As we have the threads in spin lock
        # Increasing the number of blocks by reducing the number of rows 
        # For a fact it is possible because then we just need more than one block handling a row
        # But have to calculate the gradients in that way
        num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

        # Each block handles a row
        _layer_norm_fwd_fused[(M,)](
            x_arg,
            y,
            weight,
            bias,
            Mean,
            RStd, 
            x_arg.stride(0), #Just a fancy way
            N,
            eps,
            BLOCK_SIZE,
            # num_warps=num_warps, 
            # num_ctas=1
        )
        ctx.save_for_backward(x, weight, bias, Mean, RStd)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, b, m, r = ctx.saved_tensors
        N = w.shape[0]
        # This could be handled using triton.config, I guess.
        # Less the GROUP_SIZE_M more the number of groups
        if N <= 8192: GROUP_SIZE_M = 96
        if N <= 4096: GROUP_SIZE_M = 128
        if N <= 1024: GROUP_SIZE_M = 256

        Lock = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
        DX = torch.empty_like(dy) # like takes care of device and dtype
        _dw = torch.empty((GROUP_SIZE_M, N), dtype=w.dtype, device=w.device)
        _db = torch.empty((GROUP_SIZE_M, N), dtype=w.dtype, device=w.device)

        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape
        _layer_norm_bwd_dx_fused[(M,)](
            DX,
            dy,
            _dw,
            _db,
            N,
            x_arg,
            w,
            m,
            r,
            x.stride(0),
            Lock,
            GROUP_SIZE_M,
            ctx.BLOCK_SIZE
        )
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), )
        final_dw = torch.empty((N, ), dtype=w.dtype, device=x.device)
        final_db = torch.empty((N, ), dtype=w.dtype, device=x.device)
        _layer_norm_bwd_dwdb[grid](
            _dw,
            _db,
            final_dw,
            final_db,
            # if M < GROUP_SIZE_M -> then the number of groups = M
            # M = 4; 0, 1, 2, 3
            # GROUP_SIZE_M = 8; 0, 1, 2, 3, 4, 5, 6, 7
            # M: group = 0: 0, 1: 1, 2: 2, 3: 3 # No more groups as compared than M
            min(GROUP_SIZE_M, M),
            N,
            BLOCK_SIZE_M=32, # This is again what needs to be redefined
            BLOCK_SIZE_N=128, num_ctas=1
        )


class LayerNormModule(nn.Module):
    def __init__(self, normalized_shape, weight, bias, eps):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def forward(self, x):
        return LayerNorm.apply(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps
        )
