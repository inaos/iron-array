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

int32_t get_nearest_power2(int64_t value)
{
    int64_t power2 = 2;
    while (power2 < value && power2 < INT32_MAX) {
        power2 *= 2;
    }
    power2 /= 2;
    return power2;
}

// Given a shape, offer advice on the partition size
INA_API(ina_rc_t) iarray_partition_advice(iarray_context_t *ctx,
                                          iarray_data_type_t dtype,
                                          const int ndim,
                                          const int64_t *shape,
                                          int32_t *pshape)
{
    int64_t _pshape[IARRAY_DIMENSION_MAX];  // initially pshape values need to be int64_t
    INA_UNUSED(ctx);  // we could use context in the future
    /* Use INAC to determine L3 cache size */
    const int L3 = 4 * 1024 * 1024;
    // High value should allow to hold (2x operand, 1x temporary, 1x reserve) in L3
    const int64_t high = L3 / 4;
    const int64_t low = 128 * 1024;
    int itemsize = 0;
    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            itemsize = 8;
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            itemsize = 4;
            break;
        default:
            return INA_ERR_ERROR;
    }

    for (int i = 0; i < ndim; i++) {
        _pshape[i] = get_nearest_power2(shape[i]);
    }

    // Shrink partition until we get its size into the (low, high] boundaries
    int64_t psize = 0;
    do {
        for (int i = 0; i < ndim; i++) {
            // The size of the partition so far
            psize = itemsize;
            for (int j = 0; j < ndim; j++) {
                psize *= _pshape[j];
            }
            if (psize <= high) {
                break;
            }
            else if (psize < low) {
                _pshape[i] = shape[i];
                break;
            }
            _pshape[i] /= 2;
        }
    } while (psize > high);

    // If some pshape axis is too close to the original shape, split it again, but not too much
    if (psize > low) {
        for (int i = 0; i < ndim; i++) {
            if (((float) (shape[i] - _pshape[i]) / (float) _pshape[i]) < 0.1) {
                _pshape[i] = _pshape[i] / 2 + _pshape[i] / 4;
            }
            psize = itemsize;
            for (int j = 0; j < ndim; j++) {
                psize *= _pshape[j];
            }
            if (psize < low) {
                break;
            }
        }
    }

    // Now that we are sure that all the axis in pshape are < INT32_MAX, copy to actual pshape
    for (int i = 0; i < ndim; i++) {
        pshape[i] = (int32_t)_pshape[i];
    }

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
