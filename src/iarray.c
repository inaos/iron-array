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
    return INA_SUCCESS;
}

INA_API(void) iarray_destroy()
{
    blosc_destroy();
    _blosc_inited = 0;
}

INA_API(ina_rc_t) iarray_partition_advice(iarray_data_type_t dtype, int *max_nelem, int *min_nelem)
{
    /* Use INAC to determine L3 cache size */
    // high = L3 / 4 (2x operand, 1x temporary, 1x reserve) / dtype
    //low = 4k (determine a better solution later)
    INA_UNUSED(dtype);
    INA_UNUSED(max_nelem);
    INA_UNUSED(min_nelem);
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
    if (!(cfg->eval_flags & IARRAY_EXPR_EVAL_BLOCK) && !(cfg->eval_flags & IARRAY_EXPR_EVAL_CHUNK)
        && !(cfg->eval_flags & IARRAY_EXPR_EVAL_ITERBLOCK)
        && !(cfg->eval_flags & IARRAY_EXPR_EVAL_ITERCHUNKPARA)
        &&!(cfg->eval_flags & IARRAY_EXPR_EVAL_ITERCHUNK)) {
        (*ctx)->cfg->eval_flags |= IARRAY_EXPR_EVAL_BLOCK;
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
