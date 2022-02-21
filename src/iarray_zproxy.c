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
    char *zproxy_urlpath;
    //!< urlpath to zarr array with data.
    int64_t extshape[IARRAY_DIMENSION_MAX];
    //!< Shape of original data.
    int64_t extchunkshape[IARRAY_DIMENSION_MAX];
    //!< Shape of each chunk.
    int32_t blockshape[IARRAY_DIMENSION_MAX];
    //!< Shape of each block.
    uint8_t ndim;
    //!< Data dimensions.
    zhandler_ptr zhandler;
    //!< Function pointer to zarr handler
} zproxy_postparams_udata;


ina_rc_t fill_zproxy_udata(char *zarr_urlpath,  iarray_container_t *c, zhandler_ptr zhandler)
{
    blosc2_postfilter_params *postparams = c->catarr->sc->dctx->postparams;
    zproxy_postparams_udata udata = {0};
    postparams->user_data = (void*)&udata;

    udata.zproxy_urlpath = zarr_urlpath;
    udata.ndim = c->catarr->ndim;

    for (int i = 0; i < udata.ndim; ++i) {
        udata.extshape[i] = c->catarr->extshape[i];
        udata.extchunkshape[i] = c->catarr->extchunkshape[i];
        udata.blockshape[i] = c->catarr->blockshape[i];
    }
    udata.zhandler = zhandler;

    return INA_SUCCESS;
}


ina_rc_t zproxy_postfilter(blosc2_postfilter_params *postparams)
{
    zproxy_postparams_udata *udata = postparams->user_data;
    int64_t *shape = udata->extshape;
    int64_t *chunkshape = udata->extchunkshape;
    int32_t *blockshape = udata->blockshape;
    uint8_t ndim = udata->ndim;

    int64_t chunks_in_array[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < ndim; ++i) {
        chunks_in_array[i] = shape[i] / chunkshape[i];
    }
    int64_t blocks_in_chunk[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < ndim; ++i) {
        blocks_in_chunk[i] = chunkshape[i] / blockshape[i];
    }

    // Get coordinates of chunk
    int64_t nchunk = postparams->nchunk;
    int64_t nchunk_ndim[IARRAY_DIMENSION_MAX] = {0};
    index_unidim_to_multidim(ndim, chunks_in_array, nchunk, nchunk_ndim);
    // Get coordinates of block
    int64_t nblock = postparams->nblock;
    int64_t nblock_ndim[IARRAY_DIMENSION_MAX] = {0};
    index_unidim_to_multidim(ndim, blocks_in_chunk, nblock, nblock_ndim);
    // Get start element coordinates from the corresponding block
    int64_t *start_elem_ndim = malloc(IARRAY_DIMENSION_MAX * sizeof(int64_t));
    int64_t stop_elem_ndim[IARRAY_DIMENSION_MAX] = {0};
    for (int i = 0; i < ndim; ++i) {
        start_elem_ndim[i] = nchunk_ndim[i] * chunkshape[i] + nblock_ndim[i] * blockshape[i];
        stop_elem_ndim[i] = nchunk_ndim[i] * chunkshape[i] + (nblock_ndim[i] + 1) * blockshape[i] + 1;
    }
    udata->zhandler(udata->zproxy_urlpath, start_elem_ndim, stop_elem_ndim, postparams->out);

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_add_zproxy_postfilter(iarray_container_t *src, char *zarr_urlpath, zhandler_ptr zhandler)
{
    INA_VERIFY_NOT_NULL(src);

    // Create dparams
    blosc2_dparams *dparams;
    blosc2_schunk_get_dparams(src->catarr->sc, &dparams);
    dparams->postfilter = (blosc2_postfilter_fn)zproxy_postfilter;

    // We need to zero the contents of the postparams
    blosc2_postfilter_params postparams = {0};
    zproxy_postparams_udata zpostparams = {0};
    postparams.user_data = (void*)&zpostparams;
    dparams->postparams = &postparams;

    // Create new context since postparams is empty in the old one
    blosc2_context *dctx = blosc2_create_dctx(*dparams);
    free(src->catarr->sc->dctx);
    src->catarr->sc->dctx = dctx;
    src->catarr->sc->dctx->postfilter = dparams->postfilter;

    IARRAY_RETURN_IF_FAILED(fill_zproxy_udata(zarr_urlpath, src, zhandler));

    free(dparams);

    return INA_SUCCESS;
}
