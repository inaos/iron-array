/*
 * Copyright INAOS GmbH, Thalwil, 2018.
 * Copyright Francesc Alted, 2018.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of INAOS GmbH
 * and Francesc Alted ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include <libiarray/iarray.h>

#include <iarray_private.h>

#if __linux__
#include <sys/sysinfo.h>
#include <sched.h>
#endif

static int _ina_inited = 0;
static int _blosc_inited = 0;

INA_API(ina_rc_t) iarray_init()
{
    if (!_ina_inited) {
       ina_init();
        _ina_inited = 1;
    }
    if (!_blosc_inited) {
        blosc_init();
        _blosc_inited = 1;
    }

#if __linux__
    int nprocs = get_nprocs();
    cpu_set_t  mask;
    CPU_ZERO(&mask);
    for(int i = 0; i < nprocs; i++) {
        CPU_SET(i, &mask);
    }
    sched_setaffinity(0, sizeof(mask), &mask);
#endif

    return INA_SUCCESS;
}

INA_API(void) iarray_destroy()
{
    blosc_destroy();
    _blosc_inited = 0;
}

int64_t get_nearest_power2(int64_t value)
{
    int64_t power2 = 2;
    while (power2 <= value && power2 < INT32_MAX) {
        power2 *= 2;
    }
    power2 /= 2;
    return power2;
}

// Given a shape, offer advice on the partition size
INA_API(ina_rc_t) iarray_partition_advice(iarray_context_t *ctx, iarray_dtshape_t *dtshape,
                                          int64_t low, int64_t high)
{
    INA_UNUSED(ctx);  // we could use context in the future
    INA_ASSERT_NOT_NULL(dtshape);

    if (high == 0) {
        // TODO: Use INAC to determine L3 cache size
        const int L3 = 4 * 1024 * 1024;
        // High value should allow to hold (2x operand, 1x temporary, 1x reserve) in L3
        high = L3 / 4;
    }
    if (low == 0) {
        // TODO: Use INAC to determine L2 cache size
        const int L2 = 256 * 1024;
        low = L2 / 2;
    }

    if (low > high) {
        printf("low has to be smaller than high\n");
        return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
    }

    iarray_data_type_t dtype = dtshape->dtype;
    int ndim = dtshape->ndim;
    int64_t *shape = dtshape->shape;
    int64_t *pshape = dtshape->pshape;
    int itemsize = 0;
    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            itemsize = 8;
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            itemsize = 4;
            break;
        default:
            return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
    }

    for (int i = 0; i < ndim; i++) {
        pshape[i] = get_nearest_power2(shape[i]);
    }

    // Shrink partition until we get its size into the [low, high] boundaries
    int64_t psize = 0;
    do {
        for (int i = 0; i < ndim; i++) {
            // The size of the partition so far
            psize = itemsize;
            for (int j = 0; j < ndim; j++) {
                psize *= pshape[j];
            }
            if (psize <= high) {
                break;
            }
            else if (psize < low) {
                pshape[i] = shape[i];
                break;
            }
            pshape[i] /= 2;
        }
    } while (psize > high);

    // Lastly, if some pshape axis is too close to the original shape, split it again
    if (psize > low) {
        for (int i = 0; i < ndim; i++) {
            if (((float) (shape[i] - pshape[i]) / (float) pshape[i]) < 0.1) {
                pshape[i] = pshape[i] / 2;
            }
            psize = itemsize;
            for (int j = 0; j < ndim; j++) {
                psize *= pshape[j];
            }
            if (psize < low) {
                break;
            }
        }
    }

    if (psize > INT32_MAX) {
        // The partition size can never be larger than 2 GB
        return INA_ERROR(INA_ERR_EXCEEDED);
    }

    return INA_SUCCESS;
}

// Given a matmul operation (C = A * B), provide advice on the blocks for iteration A and B
// A and B are supposed to have (M, K) and (K, N) dimensions respectively
// C is supposed to have a partition size of (m, n)
// The hint for the blockshapes are going to be (m, k) and (k, n) respectively
INA_API(ina_rc_t) iarray_matmul_advice(iarray_context_t *ctx,
                                       iarray_container_t *a,
                                       iarray_container_t *b,
                                       iarray_container_t *c,
                                       int64_t *bshape_a,
                                       int64_t *bshape_b,
                                       int64_t low,
                                       int64_t high)
{
    INA_UNUSED(ctx);  // we could use context in the future
    INA_ASSERT_NOT_NULL(a);
    INA_ASSERT_NOT_NULL(b);
    INA_ASSERT_NOT_NULL(c);
    INA_ASSERT_NOT_NULL(bshape_a);
    INA_ASSERT_NOT_NULL(bshape_b);

    if (high == 0) {
        // TODO: Use INAC to determine L3 cache size
        const int L3 = 4 * 1024 * 1024;
        // High value should allow to hold (2x operand, 1x temporary, 1x reserve) in L3
        high = L3 / 4;
    }
    if (low == 0) {
        // TODO: Use INAC to determine L2 cache size
        const int L2 = 256 * 1024;
        low = L2 / 2;
    }

    if (low > high) {
        printf("low has to be smaller than high\n");
        return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
    }

    // Take the dtype of the first array (we don't support mixing data types yet)
    iarray_data_type_t dtype = a->dtshape->dtype;
    int itemsize = (dtype == IARRAY_DATA_TYPE_DOUBLE) ? 8 : 4;

    // First, the m and n values *have* to be the same for the partition of the output
    int64_t m_dim = c->dtshape->pshape[0];
    int64_t n_dim = c->dtshape->pshape[1];

    // Now that we have a hint for M and K, get a guess of the N
    int64_t k_dim_guess1 = high / (m_dim * itemsize);
    k_dim_guess1 = get_nearest_power2(k_dim_guess1);
    int64_t k_dim_guess2 = high / (n_dim * itemsize);
    k_dim_guess2 = get_nearest_power2(k_dim_guess2);

    // Get the mean value and nearest power of 2
    int64_t k_dim = (k_dim_guess1 + k_dim_guess2) / 2;
    k_dim = get_nearest_power2(k_dim);

    if (k_dim > a->dtshape->shape[1]) {
        k_dim = get_nearest_power2(a->dtshape->shape[1]);
    }
    if (k_dim > b->dtshape->shape[0]) {
        k_dim = get_nearest_power2(b->dtshape->shape[0]);
    }

    // Correct the blocksize in case it is too small for one of the matrices
    while (((m_dim * k_dim * itemsize) < low) || (((k_dim * n_dim * itemsize) < low))) {
        k_dim *= 2;
    }

    // Correct the blocksize in case it is too large for one of the matrices
    while (((m_dim * k_dim * itemsize) > high) || (((k_dim * n_dim * itemsize) > high))) {
        k_dim /= 2;
    }

    // The block shape cannot be larger than the shape
    if (m_dim > a->dtshape->shape[0]) {
        m_dim = a->dtshape->shape[0];
    }
    if (k_dim > a->dtshape->shape[1]) {
        k_dim = a->dtshape->shape[1];
    }
    if (n_dim > b->dtshape->shape[1]) {
        n_dim = b->dtshape->shape[1];
    }

    // We are done.  Fill the block shapes and return.
    bshape_a[0] = m_dim;
    bshape_a[1] = k_dim;
    bshape_b[0] = k_dim;
    bshape_b[1] = n_dim;

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_context_new(iarray_config_t *cfg, iarray_context_t **ctx)
{
    INA_VERIFY_NOT_NULL(ctx);
    *ctx = ina_mem_alloc(sizeof(iarray_context_t));
    INA_RETURN_IF_NULL(ctx);
    (*ctx)->cfg = ina_mem_alloc(sizeof(iarray_config_t));
    INA_FAIL_IF((*ctx)->cfg == NULL);
    ina_mem_cpy((*ctx)->cfg, cfg, sizeof(iarray_config_t));
    if (!(cfg->eval_flags & IARRAY_EXPR_EVAL_ITERBLOCK)
        && !(cfg->eval_flags & IARRAY_EXPR_EVAL_ITERCHUNK)) {
        // The default is iterating by blocks
        (*ctx)->cfg->eval_flags |= IARRAY_EXPR_EVAL_ITERBLOCK;
    }
    INA_FAIL_IF_ERROR(ina_mempool_new(_IARRAY_MEMPOOL_EVAL, NULL, INA_MEM_DYNAMIC, &(*ctx)->mp));
    INA_FAIL_IF_ERROR(ina_mempool_new(_IARRAY_MEMPOOL_OP_CHUNKS, NULL, INA_MEM_DYNAMIC, &(*ctx)->mp_op));
    INA_FAIL_IF_ERROR(ina_mempool_new(_IARRAY_MEMPOOL_EVAL_TMP, NULL, INA_MEM_DYNAMIC, &(*ctx)->mp_tmp_out));

    return INA_SUCCESS;

fail:
    iarray_context_free(ctx);
    return ina_err_get_rc();
}

INA_API(void) iarray_context_free(iarray_context_t **ctx)
{
    INA_VERIFY_FREE(ctx);
    ina_mempool_free(&(*ctx)->mp_tmp_out);
    ina_mempool_free(&(*ctx)->mp_op);
    ina_mempool_free(&(*ctx)->mp);
    INA_MEM_FREE_SAFE((*ctx)->cfg);
    INA_MEM_FREE_SAFE(*ctx);
}
