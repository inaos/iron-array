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

#include "iarray_private.h"
#include <libiarray/iarray.h>
#include <stdlib.h>
#include <hwloc.h>
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
        case IARRAY_ES_CACHE_SIZES:
            return "CACHE SIZES";
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


INA_API(const char *) iarray_err_strerror(ina_rc_t error) {
    return ina_err_strerror(error);
}


// Return the number of (logical) cores in CPU
INA_API(ina_rc_t) iarray_get_ncores(int *ncores, int64_t max_ncores)
{
    // Allocate, initialize, and perform topology detection
    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);
    // Get the number of logical cores (Processing Units)
    int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_PU);
    if (depth < 0) {
        IARRAY_TRACE1(iarray.error, "Can not get the number of cores");
        return INA_ERROR(IARRAY_ERR_GET_NCORES);
    }
    *ncores = (int)hwloc_get_nbobjs_by_depth(topology, depth);
    // ...and destroy topology
    hwloc_topology_destroy(topology);

    // See whether cap value should be used
    if ((max_ncores > 0) && (*ncores > max_ncores)) {
        *ncores = max_ncores;
    }

    return INA_SUCCESS;
}


int64_t get_nearest_power2(int64_t value)
{
    int64_t power2 = 2;
    while (power2 <= value && power2 < INT64_MAX) {
        power2 *= 2;
    }
    power2 /= 2;
    return power2;
}


// Return partition shapes whose elements are a power of 2, if possible, and as squared box as possible
ina_rc_t boxed_optim_partition(int ndim, const int64_t *shape, int64_t *partshape, int itemsize,
                               int64_t minsize, int64_t maxsize) {
    for (int i = 0; i < ndim; i++) {
        partshape[i] = get_nearest_power2(shape[i]);
    }

    // Shrink partition dimensions in succession until we get its size fitting into maxsize
    int64_t partsize;
    do {
        for (int i = 0; i < ndim; i++) {
            // The size of the partition so far
            partsize = itemsize;
            for (int j = 0; j < ndim; j++) {
                partsize *= partshape[j];
            }
            if (partsize < minsize) {
                goto out2;
            }
            if (partsize <= maxsize) {
                goto out;
            }
            // Dimension 1 cannot be splitted anymore
            if (partshape[i] == 1) {
                continue;
            }
            partshape[i] /= 2;
        }
    }
    while (true);

out:
    // Lastly, if some chunkshape axis is too close to the original shape, split it again
    for (int i = 0; i < ndim; i++) {
        partsize = itemsize;
        for (int j = 0; j < ndim; j++) {
            partsize *= partshape[j];
        }
        if (partsize < minsize) {
            break;
        }
        if (partsize <= maxsize / 2) {
            break;
        }
        if (partshape[i] == 1) {
            continue;
        }
        if (((float) (shape[i] - partshape[i]) / (float) partshape[i]) < 0.1) {
            partshape[i] = partshape[i] / 2;
        }
    }

out2:
    if (partsize > INT32_MAX) {
        INA_TRACE1(iarray.error, "A chunk or block can not be larger than 2 GB");
        return INA_ERROR(IARRAY_ERR_INVALID_CHUNKSHAPE);
    }
    if (partsize <= 0) {
        INA_TRACE1(iarray.error, "A chunk or block can not be less or equal than 0");
        return INA_ERROR(IARRAY_ERR_INVALID_CHUNKSHAPE);
    }

    return INA_SUCCESS;
}


// Given a shape, offer advice on the partition shapes (chunkshape and blockshape)
INA_API(ina_rc_t) iarray_partition_advice(iarray_context_t *ctx, iarray_dtshape_t *dtshape, iarray_storage_t *storage,
                                          int64_t min_chunksize, int64_t max_chunksize,
                                          int64_t min_blocksize, int64_t max_blocksize)
{
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(storage);
    iarray_config_t* cfg = ctx->cfg;

    // Allocate, initialize, and perform topology detection
    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    if (storage->backend != IARRAY_STORAGE_BLOSC) {
        return INA_ERROR(IARRAY_ERR_INVALID_STORAGE);
    }

    // Get reasonable defaults for max and mins for chunk and block sizes
    if (max_blocksize == 0) {
        // hwloc_obj_t L2_obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_L2CACHE, 0);
        // if (L2_obj == NULL) {
        //     IARRAY_TRACE1(iarray.error, "Can not get the L2 cache size");
        //     return INA_ERROR(IARRAY_ERR_GET_CACHE_SIZES);
        // }
        // uint64_t L2_size = L2_obj->attr->cache.size;
        // Should allow to hold (4x (3 operands + 1 result) * 2x temporaries = 8x) in L2
        // max_blocksize = L2_size / 8;
        //
        // The L2 reported by Apple M1 is shared, and in the most energy-efficient cpu cluster (4 MB)
        //
        // Because of this, probably our best bet is to assign a fixed amount for the blocksize.
        // After some experimentation with the i9-10940X, 256 KB is probably a good and balanced guess.
        switch (cfg->compression_favor) {
            case IARRAY_COMPRESSION_FAVOR_CRATIO:
                max_blocksize = 1024 * 1024;
                break;
            case IARRAY_COMPRESSION_FAVOR_SPEED:
                max_blocksize = 128 * 1024;
                break;
            case IARRAY_COMPRESSION_FAVOR_BALANCE:
            default:
                max_blocksize = 256 * 1024;
        }
    }
    if (min_blocksize == 0) {
        // 1 KB for blocksize sounds like a good minimum
        min_blocksize = 1024;
    }

    if (max_chunksize == 0) {
        // Apple M1 is not working here (it does not really has a L3).
        // As this is not really necessary even for x86, we disable L3 detection for now.
        // hwloc_obj_t L3_obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_L3CACHE, 0);
        // if (L3_obj == NULL) {
        //    IARRAY_TRACE1(iarray.error, "Can not get the L3 cache size");
        //    return INA_ERROR(IARRAY_ERR_GET_CACHE_SIZES);
        // }
        // uint64_t L3_size = L3_obj->attr->cache.size;
        // Should allow to hold (2x operand, 1x temporary, 1x reserve) in L3
        // max_chunksize = L3_size / 4;
        //
        // Experiments say that making the chunks in expression to fit in L3
        // is not too important.  It better pays off to provide room enough
        // for having a lot of different threads to work in parallel for
        // producing the chunk of the output.
        // Looks like 128 MB is a good figure for the i9-10940X processor.
        switch (cfg->compression_favor) {
            case IARRAY_COMPRESSION_FAVOR_CRATIO:
                max_chunksize = 128 * 1024 * 1024;
                break;
            case IARRAY_COMPRESSION_FAVOR_SPEED:
                max_chunksize = 128 * 1024 * 1024;
                break;
            case IARRAY_COMPRESSION_FAVOR_BALANCE:
            default:
                max_chunksize = 128 * 1024 * 1024;
        }
    }
    if (min_chunksize == 0) {
        // 256 KB for chunksize sounds like a good minimum
        min_chunksize = 256 * 1024;
    }

    // ...and destroy topology
    hwloc_topology_destroy(topology);

    iarray_data_type_t dtype = dtshape->dtype;
    int8_t ndim = dtshape->ndim;
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

    // Compute the chunkshape.
    // TODO: Only boxed partition algorithm is implement, but a C and Fortran order could be useful too
    IARRAY_RETURN_IF_FAILED(boxed_optim_partition(ndim, shape, chunkshape, itemsize,
                                                  min_chunksize, max_chunksize));

    int32_t chunksize = itemsize;
    for (int i = 0; i < ndim; i++) {
        chunksize *= chunkshape[i];
    }
    if (chunksize < max_blocksize) {
        max_blocksize = chunksize;
    }
    // Compute the blockshape
    IARRAY_RETURN_IF_FAILED(boxed_optim_partition(ndim, chunkshape, blockshape, itemsize,
                                                  min_blocksize, max_blocksize));

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_context_new(iarray_config_t *cfg, iarray_context_t **ctx)
{if (!_ina_inited) {
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
    cparams->preparams = ctx->prefilter_params;
    cparams->prefilter = ctx->prefilter_fn;
    int blosc_filter_idx = 0;
    cparams->compcode = ctx->cfg->compression_codec;
    cparams->use_dict = ctx->cfg->use_dict;
    cparams->clevel = (uint8_t)ctx->cfg->compression_level; /* Since its just a mapping, we know the cast is ok */
    cparams->blocksize = blocksize;
    cparams->typesize = (int32_t) typesize;
    cparams->nthreads = (uint16_t) ctx->cfg->max_num_threads; /* Since its just a mapping, we know the cast is ok */
    if ((ctx->cfg->filter_flags & IARRAY_COMP_TRUNC_PREC)) {
        cparams->filters[blosc_filter_idx] = BLOSC_TRUNC_PREC;
        cparams->filters_meta[blosc_filter_idx] = ctx->cfg->fp_mantissa_bits;
        blosc_filter_idx++;
    }
    if (ctx->cfg->filter_flags & IARRAY_COMP_DELTA) {
        cparams->filters[blosc_filter_idx] = BLOSC_DELTA;
        blosc_filter_idx++;
    }
    if (ctx->cfg->filter_flags & IARRAY_COMP_BITSHUFFLE) {
        cparams->filters[BLOSC2_MAX_FILTERS - 1] = BLOSC_BITSHUFFLE;
    }
    if (ctx->cfg->filter_flags & IARRAY_COMP_SHUFFLE) {
        cparams->filters[BLOSC2_MAX_FILTERS - 1] = BLOSC_SHUFFLE;
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
    if (cfg->filter_flags & IARRAY_COMP_DELTA) {
        cat_cfg->filters[blosc_filter_idx] = BLOSC_DELTA;
    }
    if (cfg->filter_flags & IARRAY_COMP_BITSHUFFLE) {
        cat_cfg->filters[BLOSC2_MAX_FILTERS - 1] = BLOSC_BITSHUFFLE;
    }
    if (cfg->filter_flags & IARRAY_COMP_SHUFFLE) {
        cat_cfg->filters[BLOSC2_MAX_FILTERS - 1] = BLOSC_SHUFFLE;
    }
    if (cfg->btune) {
        blosc2_btune *iabtune = malloc(sizeof(blosc2_btune));
        btune_config iabtune_config = BTUNE_CONFIG_DEFAULTS;
        switch (cfg->compression_favor) {
            case IARRAY_COMPRESSION_FAVOR_CRATIO:
                iabtune_config.comp_mode = BTUNE_COMP_HCR;
                break;
            case IARRAY_COMPRESSION_FAVOR_SPEED:
                iabtune_config.comp_mode = BTUNE_COMP_HSP;
                break;
            default:
                iabtune_config.comp_mode = BTUNE_COMP_BALANCED;
        }
        iabtune->btune_config = malloc(sizeof(btune_config));
        memcpy(iabtune->btune_config, &iabtune_config, sizeof(btune_config));
        iabtune->btune_init = iabtune_init;
        iabtune->btune_next_blocksize = iabtune_next_blocksize;
        iabtune->btune_next_cparams = iabtune_next_cparams;
        iabtune->btune_update = iabtune_update;
        iabtune->btune_free = iabtune_free;
        cat_cfg->udbtune = iabtune;
    }
    // cat_cfg->udbtune = NULL;

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


static int32_t serialize_meta(iarray_data_type_t dtype, uint8_t **smeta)
{
    if (smeta == NULL) {
        return -1;
    }
    if (dtype > IARRAY_DATA_TYPE_MAX) {
        return -1;
    }
    int32_t smeta_len = 3;  // the dtype should take less than 7-bit, so 1 byte is enough to store it
    *smeta = malloc((size_t)smeta_len);

    // version
    **smeta = 0;

    // dtype entry
    *(*smeta + 1) = (uint8_t)dtype;  // positive fixnum (7-bit positive integer)

    // flags (initialising all the entries to 0)
    *(*smeta + 2) = 0;  // positive fixnum (7-bit for flags)

    return smeta_len;
}


ina_rc_t iarray_create_caterva_storage(iarray_dtshape_t *dtshape, iarray_storage_t *storage, caterva_storage_t *cat_storage) {
    cat_storage->backend = storage->backend == IARRAY_STORAGE_BLOSC ? CATERVA_STORAGE_BLOSC : CATERVA_STORAGE_PLAINBUFFER;
    switch (cat_storage->backend) {
        case CATERVA_STORAGE_BLOSC:
            cat_storage->properties.blosc.sequencial = storage->enforce_frame;
            cat_storage->properties.blosc.urlpath = storage->urlpath;
            for (int i = 0; i < dtshape->ndim; ++i) {
                cat_storage->properties.blosc.chunkshape[i] = (int32_t) storage->chunkshape[i];
                cat_storage->properties.blosc.blockshape[i] = (int32_t) storage->blockshape[i];
            }
            uint8_t *smeta;
            int32_t smeta_len = serialize_meta(dtshape->dtype, &smeta);
            if (smeta_len < 0) {
                IARRAY_TRACE1(iarray.error, "Error serializing the meta-information");
                return INA_ERROR(INA_ERR_FAILED);
            }
            cat_storage->properties.blosc.nmetalayers = 1;
            caterva_metalayer_t *metalayer = &cat_storage->properties.blosc.metalayers[0];
            metalayer->name = strdup("iarray");
            metalayer->sdata = smeta;
            metalayer->size = smeta_len;
            break;
        case CATERVA_STORAGE_PLAINBUFFER:
            break;
    }
    return INA_SUCCESS;
}
