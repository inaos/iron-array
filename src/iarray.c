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
#include <stdlib.h>

#include <iarray_private.h>

#include <minjugg.h>

#if __linux__
#include <sys/sysinfo.h>
#include <sched.h>
#endif

static int _ina_inited = 0;
static int _blosc_inited = 0;
static int _jug_inited = 0;


static const char* __get_err_getsubject(int id) {
    switch (id) {
        case IARRAY_ES_CONTAINER:
            return "CONTAINER";
        case IARRAY_ES_DTSHAPE:
            return "DTSHAPE";
        case IARRAY_ES_SHAPE:
            return "SHAPE";
        case IARRAY_ES_CHUNKSHAPE:
            return "CHUNK SHAPE";
        case IARRAY_ES_NDIM:
            return "NUMBER OF DIMENSIONS";
        case IARRAY_ES_DTYPE:
            return "DATA TYPE";
        case IARRAY_ES_STORAGE:
            return "STORAGE";
        case IARRAY_ES_PERSISTENCY:
            return "PERSISTENCY";
        case IARRAY_ES_BUFFER:
            return "BUFFER";
        case IARRAY_ES_CATERVA:
            return "CATERVA";
        case IARRAY_ES_BLOSC:
            return "BLOSC";
        case IARRAY_ES_ASSERTION:
            return "ASSERTION";
        case IARRAY_ES_BLOCKSHAPE:
            return "BLOCK SHAPE";
        case IARRAY_ES_RNG_METHOD:
            return "RANDOM GENERATOR METHOD";
        case IARRAY_ES_RAND_METHOD:
            return "RANDOM METHOD";
        case IARRAY_ES_RAND_PARAM:
            return "RANDOM PARAM";
        case IARRAY_ES_ITER:
            return "ITERATOR";
        case IARRAY_ES_EVAL_METHOD:
            return "EVALUATION METHOD";
        case IARRAY_ES_EVAL_ENGINE:
            return "EVALUATION ENGINE";
        case IARRAY_ES_NCORES:
            return "NUMBER OF CORES";
        default:
            return "";
    }
}

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
    if (!_jug_inited) {
        jug_init();
        _jug_inited = 1;
    }

    ina_err_register_dict(__get_err_getsubject);

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
    jug_destroy();
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

// Return the number of (logical) cores in CPU
INA_API(ina_rc_t) iarray_get_ncores(int *ncores, int64_t max_ncores)
{
    *ncores = 1;
#ifdef INA_OS_OSX
    *ncores = (int)sysconf(_SC_NPROCESSORS_ONLN);
#else
    IARRAY_FAIL_IF_ERROR(ina_cpu_get_total_logical_count(ncores));
#endif
    if ((max_ncores > 0) && (*ncores > max_ncores)) {
        *ncores = max_ncores;
    }

    return INA_SUCCESS;

#ifndef INA_OS_OSX
fail:
    INA_TRACE1(iarray.error, "Cannot get the number of cores");
    return INA_ERROR(IARRAY_ERR_GET_NCORES);
#endif

}

// Partition optimized for C access
int32_t c_optim_partition(int64_t low, int64_t high, int ndim, const int64_t *shape, int64_t *partshape,
                          int itemsize) {
    for (int i = 0; i < ndim; i++) {
        partshape[i] = get_nearest_power2(shape[i]);
    }

    // Shrink chunk until we get its size into the [low, high] boundaries
    int64_t partsize = 0;
    do {
        for (int i = 0; i < ndim; i++) {
            // The size of the chunk so far
            partsize = itemsize;
            for (int j = 0; j < ndim; j++) {
                partsize *= partshape[j];
            }
            if (partsize <= high) {
                break;
            }
            else if (partsize < low) {
                partshape[i] = shape[i];
                break;
            }
            partshape[i] /= 2;
        }
    }
    while (partsize > high);

    return partsize;
}

// Given a shape, offer advice on the partition shapes (chunkshape and blockshape)
INA_API(ina_rc_t) iarray_partition_advice(iarray_context_t *ctx, iarray_dtshape_t *dtshape, iarray_storage_t *storage,
                                          int64_t low, int64_t high)
{
    INA_UNUSED(ctx);  // we could use context in the future
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(storage);

    if (storage->backend != IARRAY_STORAGE_BLOSC) {
        return INA_ERROR(IARRAY_ERR_INVALID_STORAGE);
    }
    if (high == 0) {
        size_t L3;
        IARRAY_RETURN_IF_FAILED(ina_cpu_get_l3_cache_size(&L3));
        // High value should allow to hold (2x operand, 1x temporary, 1x reserve) in L3
        high = L3 / 4;
    }
    if (low == 0) {
        size_t L2;
        IARRAY_RETURN_IF_FAILED(ina_cpu_get_l2_cache_size(&L2));
        low = L2 / 2;
    }

    if (low > high) {
        INA_TRACE1(iarray.error, "The low limit is greater than the high limit");
        return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
    }

    iarray_data_type_t dtype = dtshape->dtype;
    int ndim = dtshape->ndim;
    int64_t *shape = dtshape->shape;
    int64_t *chunkshape = storage->chunkshape;
    int64_t *blockshape = storage->blockshape;
    int itemsize;
    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            itemsize = 8;
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            itemsize = 4;
            break;
        default:
            INA_TRACE1(iarray.error, "The data type is invalid");
            return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }

    int64_t chunksize = c_optim_partition(low, high, ndim, shape, chunkshape, itemsize);
    if (chunksize > INT32_MAX) {
        INA_TRACE1(iarray.error, "The chunk size can not be larger than 2 GB");
        return INA_ERROR(IARRAY_ERR_INVALID_CHUNKSHAPE);
    }

    int64_t blocksize = c_optim_partition(low, high, ndim, shape, blockshape, itemsize);
    if (blocksize > INT32_MAX) {
        INA_TRACE1(iarray.error, "The block size can not be larger than 2 GB");
        return INA_ERROR(IARRAY_ERR_INVALID_CHUNKSHAPE);
    }

    return INA_SUCCESS;
}

// Given a matmul operation (C = A * B), provide advice on the blocks for iteration A and B
// A and B are supposed to have (M, K) and (K, N) dimensions respectively
// C is supposed to have a chunk size of (m, n)
// The hint for the blockshapes are going to be (m, k) and (k, n) respectively
INA_API(ina_rc_t) iarray_matmul_advice(iarray_context_t *ctx,
                                       iarray_container_t *a,
                                       iarray_container_t *b,
                                       iarray_container_t *c,
                                       int64_t *blockshape_a,
                                       int64_t *blockshape_b,
                                       int64_t low,
                                       int64_t high)
{
    INA_UNUSED(ctx);  // we could use context in the future
    INA_VERIFY_NOT_NULL(a);
    INA_VERIFY_NOT_NULL(b);
    INA_VERIFY_NOT_NULL(c);
    INA_VERIFY_NOT_NULL(blockshape_a);
    INA_VERIFY_NOT_NULL(blockshape_b);

    if (high == 0) {
        size_t L3;
        IARRAY_RETURN_IF_FAILED(ina_cpu_get_l3_cache_size(&L3));
        // High value should allow to hold (2x operand, 1x temporary, 1x reserve) in L3
        high = L3 / 4;
    }
    if (low == 0) {
        size_t L2;
        IARRAY_RETURN_IF_FAILED(ina_cpu_get_l2_cache_size(&L2));
        low = L2 / 2;
    }

    if (low > high) {
        INA_TRACE1(iarray.error, "The low limit is grater than the high limit");
        return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
    }

    // Take the dtype of the first array (we don't support mixing data types yet)
    iarray_data_type_t dtype = a->dtshape->dtype;
    int itemsize;
    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            itemsize = 8;
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            itemsize = 4;
            break;
        default:
            INA_TRACE1(iarray.error, "The data type is invalid");
            return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }
    // First, the m and n values *have* to be the same for the chunk of the output
    int64_t m_dim = c->storage->chunkshape[0];
    int64_t n_dim = c->storage->chunkshape[1];

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
    blockshape_a[0] = m_dim;
    blockshape_a[1] = k_dim;
    blockshape_b[0] = k_dim;
    blockshape_b[1] = n_dim;

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_context_new(iarray_config_t *cfg, iarray_context_t **ctx)
{
    if (!_ina_inited) {
        fprintf(stderr, "Error.  You need to call `iarray_init()` prior to any other iarray function call.");
        exit(1);
    }

    INA_VERIFY_NOT_NULL(ctx);
    *ctx = ina_mem_alloc(sizeof(iarray_context_t));

    INA_VERIFY_NOT_NULL(cfg);
    (*ctx)->cfg = ina_mem_alloc(sizeof(iarray_config_t));

    ina_mem_cpy((*ctx)->cfg, cfg, sizeof(iarray_config_t));

    IARRAY_RETURN_IF_FAILED(ina_mempool_new(_IARRAY_MEMPOOL_EVAL, NULL, INA_MEM_DYNAMIC, &(*ctx)->mp));
    IARRAY_RETURN_IF_FAILED(ina_mempool_new(_IARRAY_MEMPOOL_EVAL, NULL, INA_MEM_DYNAMIC, &(*ctx)->mp_chunk_cache));
    IARRAY_RETURN_IF_FAILED(ina_mempool_new(_IARRAY_MEMPOOL_OP_CHUNKS, NULL, INA_MEM_DYNAMIC, &(*ctx)->mp_op));
    IARRAY_RETURN_IF_FAILED(ina_mempool_new(_IARRAY_MEMPOOL_EVAL_TMP, NULL, INA_MEM_DYNAMIC, &(*ctx)->mp_tmp_out));

    (*ctx)->prefilter_fn = NULL;
    (*ctx)->prefilter_params = NULL;

    return INA_SUCCESS;
}


INA_API(void) iarray_context_free(iarray_context_t **ctx)
{
    INA_VERIFY_FREE(ctx);
    ina_mempool_free(&(*ctx)->mp_tmp_out);
    ina_mempool_free(&(*ctx)->mp_op);
    ina_mempool_free(&(*ctx)->mp_chunk_cache);
    ina_mempool_free(&(*ctx)->mp);
    INA_MEM_FREE_SAFE((*ctx)->cfg);
    INA_MEM_FREE_SAFE(*ctx);
    *ctx = NULL;
}


ina_rc_t iarray_create_blosc_cparams(blosc2_cparams *cparams,
                                     iarray_context_t *ctx,
                                     int8_t typesize,
                                     int32_t blocksize)
{
    cparams->pparams = ctx->prefilter_params;
    cparams->prefilter = ctx->prefilter_fn;
    int blosc_filter_idx = 0;
    cparams->compcode = ctx->cfg->compression_codec;
    cparams->use_dict = ctx->cfg->use_dict;
    cparams->clevel = (uint8_t)ctx->cfg->compression_level; /* Since its just a mapping, we know the cast is ok */
    cparams->blocksize = blocksize;
    cparams->typesize = typesize;
    cparams->nthreads = (uint16_t)ctx->cfg->max_num_threads; /* Since its just a mapping, we know the cast is ok */
    if ((ctx->cfg->filter_flags & IARRAY_COMP_TRUNC_PREC)) {
        cparams->filters[blosc_filter_idx] = BLOSC_TRUNC_PREC;
        cparams->filters_meta[blosc_filter_idx] = ctx->cfg->fp_mantissa_bits;
        blosc_filter_idx++;
    }
    if (ctx->cfg->filter_flags & IARRAY_COMP_BITSHUFFLE) {
        cparams->filters[blosc_filter_idx] = BLOSC_BITSHUFFLE;
        blosc_filter_idx++;
    }
    if (ctx->cfg->filter_flags & IARRAY_COMP_SHUFFLE) {
        cparams->filters[blosc_filter_idx] = BLOSC_SHUFFLE;
        blosc_filter_idx++;
    }
    if (ctx->cfg->filter_flags & IARRAY_COMP_DELTA) {
        cparams->filters[blosc_filter_idx] = BLOSC_DELTA;
        blosc_filter_idx++;
    }
    return INA_SUCCESS;
}


ina_rc_t iarray_create_caterva_cfg(iarray_config_t *cfg, void *(*alloc)(size_t), void (*free)(void *), caterva_config_t *cat_cfg) {
    cat_cfg->alloc = alloc;
    cat_cfg->free = free;

    cat_cfg->nthreads = cfg->max_num_threads;
    cat_cfg->compcodec = cfg->compression_codec;
    cat_cfg->complevel = cfg->compression_level;
    cat_cfg->usedict = cfg->use_dict;
    cat_cfg->prefilter = NULL;
    cat_cfg->pparams = NULL;

    int blosc_filter_idx = 0;
    if ((cfg->filter_flags & IARRAY_COMP_TRUNC_PREC)) {
        cat_cfg->filters[blosc_filter_idx] = BLOSC_TRUNC_PREC;
        cat_cfg->filtersmeta[blosc_filter_idx] = cfg->fp_mantissa_bits;
        blosc_filter_idx++;
    }
    if (cfg->filter_flags & IARRAY_COMP_BITSHUFFLE) {
        cat_cfg->filters[blosc_filter_idx] = BLOSC_BITSHUFFLE;
        blosc_filter_idx++;
    }
    if (cfg->filter_flags & IARRAY_COMP_SHUFFLE) {
        cat_cfg->filters[blosc_filter_idx] = BLOSC_SHUFFLE;
        blosc_filter_idx++;
    }
    if (cfg->filter_flags & IARRAY_COMP_DELTA) {
        cat_cfg->filters[blosc_filter_idx] = BLOSC_DELTA;
    }
    return INA_SUCCESS;
}


ina_rc_t iarray_create_caterva_params(iarray_dtshape_t *dtshape, caterva_params_t *cat_params) {
    cat_params->ndim = dtshape->ndim;
    cat_params->itemsize = dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE ? sizeof(double) : sizeof(float);
    for (int i = 0; i < cat_params->ndim; ++i) {
        cat_params->shape[i] = dtshape->shape[i];
    }
    return INA_SUCCESS;
}


ina_rc_t iarray_create_caterva_storage(iarray_dtshape_t *dtshape, iarray_storage_t *storage, caterva_storage_t *cat_storage) {
    cat_storage->backend = storage->backend == IARRAY_STORAGE_BLOSC ? CATERVA_STORAGE_BLOSC : CATERVA_STORAGE_PLAINBUFFER;
    switch (cat_storage->backend) {
        case CATERVA_STORAGE_BLOSC:
            cat_storage->properties.blosc.enforceframe = storage->enforce_frame;
            cat_storage->properties.blosc.filename = storage->filename;
            for (int i = 0; i < dtshape->ndim; ++i) {
                cat_storage->properties.blosc.chunkshape[i] = (int32_t) storage->chunkshape[i];
                cat_storage->properties.blosc.blockshape[i] = (int32_t) storage->blockshape[i];
            }
            break;
        case CATERVA_STORAGE_PLAINBUFFER:
            break;
    }
    return INA_SUCCESS;
}
