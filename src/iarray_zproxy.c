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
#include "caterva_utils.h"
#include <libiarray/iarray.h>
#include <stdio.h>


typedef struct {
    char *zproxy_urlpath;
    //!< urlpath to zarr array with data.
    int64_t shape[IARRAY_DIMENSION_MAX];
    //!< Shape of original data.
    int64_t extshape[IARRAY_DIMENSION_MAX];
    //!< Extended shape of original data.
    int64_t chunkshape[IARRAY_DIMENSION_MAX];
    //!< Shape of each chunk.
    int64_t extchunkshape[IARRAY_DIMENSION_MAX];
    //!< Extended shape of each chunk.
    int32_t blockshape[IARRAY_DIMENSION_MAX];
    //!< Shape of each block.
    uint8_t ndim;
    //!< Data dimensions.
    zhandler_ptr zhandler;
    //!< Function pointer to zarr handler
} zproxy_postparams_udata;


ina_rc_t zproxy_postfilter(blosc2_postfilter_params *postparams)
{
    zproxy_postparams_udata *udata = postparams->user_data;
    int64_t *extshape = udata->extshape;
    int64_t *extchunkshape = udata->extchunkshape;
    int64_t *chunkshape = udata->chunkshape;
    int32_t *blockshape = udata->blockshape;
    uint8_t ndim = udata->ndim;

    int64_t chunks_in_array[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < ndim; ++i) {
        chunks_in_array[i] = extshape[i] / chunkshape[i];
    }
    int64_t blocks_in_chunk[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < ndim; ++i) {
        blocks_in_chunk[i] = extchunkshape[i] / blockshape[i];
    }

    // Get coordinates of chunk
    int64_t nchunk = postparams->nchunk;
    int64_t nchunk_ndim[IARRAY_DIMENSION_MAX];
    index_unidim_to_multidim((int8_t)ndim, chunks_in_array, nchunk, nchunk_ndim);
    // Get coordinates of block
    int64_t nblock = postparams->nblock;
    int64_t nblock_ndim[IARRAY_DIMENSION_MAX];
    index_unidim_to_multidim((int8_t)ndim, blocks_in_chunk, nblock, nblock_ndim);
    // Get start element coordinates from the corresponding block
    int64_t start_elem_ndim[IARRAY_DIMENSION_MAX];
    int64_t stop_elem_ndim[IARRAY_DIMENSION_MAX];
    int64_t slice_shape[IARRAY_DIMENSION_MAX];
    int64_t size = postparams->typesize;
    for (int i = 0; i < ndim; ++i) {
        start_elem_ndim[i] = nchunk_ndim[i] * chunkshape[i] + nblock_ndim[i] * blockshape[i];
        if (start_elem_ndim[i] >= udata->shape[i]) {
            // This block does not contain any data because of the padding
            return INA_SUCCESS;
        }
        stop_elem_ndim[i] = nchunk_ndim[i] * chunkshape[i] + (nblock_ndim[i] + 1) * blockshape[i];
        // The stop may include the padding due to the blockshape
        if (nblock_ndim[i] == (blocks_in_chunk[i] - 1)) {
            stop_elem_ndim[i] -= (extchunkshape[i] - udata->chunkshape[i]);
        }
        // The stop may include the padding due to the chunkshape
        if (stop_elem_ndim[i] > udata->shape[i]) {
            stop_elem_ndim[i] = udata->shape[i];
        }
        slice_shape[i] = stop_elem_ndim[i] - start_elem_ndim[i];
        size *= slice_shape[i];
    }

    udata->zhandler(udata->zproxy_urlpath, start_elem_ndim, stop_elem_ndim, postparams->out);

    // Realloc data since there may be padding between elements
    uint8_t *aux = malloc(size);
    memcpy(aux, postparams->out, size);
    memset(postparams->out, 0, postparams->size);
    int64_t slice_start[IARRAY_DIMENSION_MAX] = {0};
    int64_t blockshape_i64[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < ndim; ++i) {
        blockshape_i64[i] = blockshape[i];
    }
    caterva_copy_buffer(ndim, (uint8_t)postparams->typesize,
                        aux, slice_shape, slice_start, slice_shape,
                        postparams->out, blockshape_i64, slice_start);
    free(aux);

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_add_zproxy_postfilter(iarray_container_t *src, char *zarr_urlpath, zhandler_ptr zhandler)
{
    INA_VERIFY_NOT_NULL(src);

    // Create dparams
    blosc2_dparams *dparams;
    blosc2_schunk_get_dparams(src->catarr->sc, &dparams);

    dparams->postfilter = (blosc2_postfilter_fn)zproxy_postfilter;

    blosc2_postfilter_params *postparams = malloc(sizeof(blosc2_postfilter_params));
    zproxy_postparams_udata *zpostparams = malloc(sizeof(zproxy_postparams_udata));

    // Fill the user_data
    zpostparams->zproxy_urlpath = malloc(strlen(zarr_urlpath) + 1);
    strcpy(zpostparams->zproxy_urlpath, zarr_urlpath);

    zpostparams->ndim = src->catarr->ndim;
    for (int i = 0; i < zpostparams->ndim; ++i) {
        zpostparams->extshape[i] = src->catarr->extshape[i];
        zpostparams->shape[i] = src->catarr->shape[i];
        zpostparams->chunkshape[i] = src->catarr->chunkshape[i];
        zpostparams->extchunkshape[i] = src->catarr->extchunkshape[i];
        zpostparams->blockshape[i] = src->catarr->blockshape[i];
    }

    zpostparams->zhandler = zhandler;

    postparams->user_data = (void*)zpostparams;
    dparams->postparams = postparams;

    // Create new context since postparams is empty in the old one
    blosc2_free_ctx(src->catarr->sc->dctx);
    src->catarr->sc->dctx = blosc2_create_dctx(*dparams);
    free(dparams);

    return INA_SUCCESS;
}
