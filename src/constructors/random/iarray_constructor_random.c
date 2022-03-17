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


typedef struct {
    uint8_t itemsize;
    int32_t blocksize;
    int32_t chunksize;
    iarray_random_ctx_t *rng_ctx;
    iarray_random_method_fn random_method_fn;
    VSLStreamStatePtr *streams;
    uint8_t **buffers;
} iarray_constructor_random_info;

typedef struct {
    VSLStreamStatePtr stream;
    uint8_t *random_buffer;
} iarray_constructor_random_block_info;

int random_array_init_fn(iarray_constructor_array_info_t *array_info,
                         void **custom_array_info) {

    return 0;
}


int random_block_init_fn(iarray_constructor_array_info_t *array_info,
                         iarray_constructor_chunk_info_t *chunk_info,
                         iarray_constructor_block_info_t *block_info,
                         void *custom_array_info,
                         void *custom_chunk_info,
                         void **custom_block_info) {

    INA_UNUSED(array_info);
    INA_UNUSED(custom_chunk_info);
    iarray_constructor_random_info *random_array_info = custom_array_info;

    *custom_block_info = ina_mem_alloc(sizeof(iarray_constructor_random_block_info));
    iarray_constructor_random_block_info *random_block_info = *custom_block_info;
    // random_block_info->stream = random_array_info->streams[block_info->tid];

    int vsl_error;
    INA_TRACE1(iarray.trace, "Start copying the stream");
    vsl_error = vslCopyStream(&random_block_info->stream, random_array_info->rng_ctx->stream);
    if (vsl_error != VSL_STATUS_OK) {
        INA_TRACE1(iarray.error, "The stream copy failed(%d)", vsl_error);
        return -1;
    }
    MKL_UINT64 nskip[1];
    nskip[0] = (MKL_UINT64) block_info->index_in_chunk_flat * array_info->a->catarr->blocknitems +
            chunk_info->index_flat * array_info->a->catarr->chunknitems;
    vsl_error = vslSkipAheadStreamEx(random_block_info->stream, (MKL_INT) 1, nskip);
    if (vsl_error != VSL_STATUS_OK) {
        INA_TRACE1(iarray.error, "The stream skip ahead %lld elements failed (%d)", nskip[0], vsl_error);
        return -1;
    }
    INA_TRACE1(iarray.trace, "the stream copy is finished");

    vsl_error = random_array_info->random_method_fn(random_array_info->rng_ctx,
                                                 random_block_info->stream,
                                                 random_array_info->itemsize,
                                                 random_array_info->blocksize,
                                                 random_array_info->buffers[block_info->tid]);
    if (vsl_error != VSL_ERROR_OK) {
        IARRAY_TRACE1(iarray.error, "The random generator method failed");
        return -1;
    }

    return 0;
}

int random_block_destroy_fn(iarray_constructor_array_info_t *array_info,
                            iarray_constructor_chunk_info_t *chunk_info,
                            iarray_constructor_block_info_t *block_info,
                            void *custom_info,
                            void *custom_chunk_info,
                            void **custom_block_info) {
    INA_UNUSED(array_info);
    INA_UNUSED(chunk_info);
    INA_UNUSED(block_info);
    INA_UNUSED(custom_info);
    INA_UNUSED(custom_chunk_info);

    iarray_constructor_random_block_info *random_block_info = *custom_block_info;
    vslDeleteStream(&random_block_info->stream);

    INA_MEM_FREE_SAFE(random_block_info);

    return 0;
}

int random_generator_fn(uint8_t *dest,
                        iarray_constructor_array_info_t *array_info,
                        iarray_constructor_chunk_info_t *chunk_info,
                        iarray_constructor_block_info_t *block_info,
                        void *custom_array_info,
                        void *custom_chunk_info,
                        void *custom_block_info,
                        iarray_constructor_item_fn item_fn) {
    INA_UNUSED(chunk_info);
    INA_UNUSED(custom_chunk_info);
    INA_UNUSED(custom_block_info);
    INA_UNUSED(item_fn);

    iarray_constructor_random_info *random_array_info = custom_array_info;

    iarray_container_t *a = array_info->a;

    uint8_t ndim = array_info->ndim;
    uint8_t itemsize = array_info->itemsize;

    int64_t src_start[CATERVA_MAX_DIM] = {0};
    int64_t dst_start[CATERVA_MAX_DIM] = {0};
    int64_t dst_shape[CATERVA_MAX_DIM] = {0};
    for (int i = 0; i < ndim; ++i) {
        dst_shape[i] = a->catarr->blockshape[i];
    }

    caterva_copy_buffer(ndim,
                        itemsize,
                        random_array_info->buffers[block_info->tid],
                        block_info->shape,
                        src_start,
                        block_info->shape,
                        dest,
                        dst_shape,
                        dst_start
                        );

    return 0;
}

INA_API(ina_rc_t) iarray_random_prefilter(iarray_context_t *ctx,
                                          iarray_dtshape_t *dtshape,
                                          iarray_random_ctx_t *random_ctx,
                                          iarray_random_method_fn random_method_fn,
                                          iarray_storage_t *storage,
                                          iarray_container_t **container)
{
    iarray_constructor_random_info random_info;
    random_info.rng_ctx = random_ctx;
    random_info.random_method_fn = random_method_fn;
    random_info.itemsize = dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE ? 8 : 4;
    random_info.blocksize = 1;
    random_info.chunksize = 1;
    for (int i = 0; i < dtshape->ndim; ++i) {
        random_info.blocksize *= (int32_t) storage->blockshape[i];
        random_info.chunksize *= (int32_t) storage->chunkshape[i];
    }

    int nthreads = mkl_get_max_threads();
    mkl_set_num_threads(1);

    random_info.streams = ina_mem_alloc(sizeof(VSLStreamStatePtr) * ctx->cfg->max_num_threads);
    random_info.buffers = ina_mem_alloc(sizeof(uint8_t *) * ctx->cfg->max_num_threads);
    for (int i = 0; i < ctx->cfg->max_num_threads; ++i) {
        // vslCopyStream(&random_info.streams[i], random_ctx->stream);
        // vslSkipAheadStream(random_info.streams[i], i * random_info.blocksize);
        random_info.buffers[i] = ina_mem_alloc(random_info.blocksize * random_info.itemsize);
    }

    iarray_constructor_block_params_t block_params = IARRAY_CONSTRUCTOR_BLOCK_PARAMS_DEFAULT;
    block_params.block_init_fn = random_block_init_fn;
    block_params.block_destroy_fn = random_block_destroy_fn;
    block_params.generator_fn = random_generator_fn;
    block_params.constructor_info = &random_info;

    INA_RETURN_IF_FAILED(iarray_constructor_block(ctx, dtshape, &block_params, storage, container));

    for (int i = 0; i < ctx->cfg->max_num_threads; ++i) {
        // vslDeleteStream(&random_info.streams[i]);
        INA_MEM_FREE_SAFE(random_info.buffers[i]);
    }
    INA_MEM_FREE_SAFE(random_info.streams);
    INA_MEM_FREE_SAFE(random_info.buffers);

    mkl_set_num_threads(nthreads);

    return INA_SUCCESS;
}
