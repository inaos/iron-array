/*
 * Copyright ironArray SL 2021.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of ironArray SL
 * ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include "iarray_private.h"


typedef struct {
    void (*cast)(void *, void *, int32_t);
    //!< Cast function.
    blosc2_schunk *viewed_schunk;
    //!< Predecessor's schunk
} view_postparams_udata;


#define CAST (void (*)(void *, void *, int32_t))

#define D_I64_PARAMS const double *src, int64_t *dst, int32_t nelems
#define D_UI64_PARAMS const double *src, uint64_t *dst, int32_t nelems

#define F_D_PARAMS const float *src, double *dst, int32_t nelems
#define F_I64_PARAMS const float *src, int64_t *dst, int32_t nelems
#define F_I32_PARAMS const float *src, int32_t *dst, int32_t nelems
#define F_UI64_PARAMS const float *src, uint64_t *dst, int32_t nelems
#define F_UI32_PARAMS const float *src, uint32_t *dst, int32_t nelems

#define I64_D_PARAMS const int64_t *src, double *dst, int32_t nelems
#define I64_UI64_PARAMS const int64_t *src, uint64_t *dst, int32_t nelems

#define I32_D_PARAMS const int32_t *src, double *dst, int32_t nelems
#define I32_F_PARAMS const int32_t *src, float *dst, int32_t nelems
#define I32_I64_PARAMS const int32_t *src, int64_t *dst, int32_t nelems
#define I32_UI64_PARAMS const int32_t *src, uint64_t *dst, int32_t nelems
#define I32_UI32_PARAMS const int32_t *src, uint32_t *dst, int32_t nelems

#define I16_D_PARAMS const int16_t *src, double *dst, int32_t nelems
#define I16_F_PARAMS const int16_t *src, float *dst, int32_t nelems
#define I16_I64_PARAMS const int16_t *src, int64_t *dst, int32_t nelems
#define I16_I32_PARAMS const int16_t *src, int32_t *dst, int32_t nelems
#define I16_UI64_PARAMS const int16_t *src, uint64_t *dst, int32_t nelems
#define I16_UI32_PARAMS const int16_t *src, uint32_t *dst, int32_t nelems
#define I16_UI16_PARAMS const int16_t *src, uint16_t *dst, int32_t nelems

#define I8_D_PARAMS const int8_t *src, double *dst, int32_t nelems
#define I8_F_PARAMS const int8_t *src, float *dst, int32_t nelems
#define I8_I64_PARAMS const int8_t *src, int64_t *dst, int32_t nelems
#define I8_I32_PARAMS const int8_t *src, int32_t *dst, int32_t nelems
#define I8_I16_PARAMS const int8_t *src, int16_t *dst, int32_t nelems
#define I8_UI64_PARAMS const int8_t *src, uint64_t *dst, int32_t nelems
#define I8_UI32_PARAMS const int8_t *src, uint32_t *dst, int32_t nelems
#define I8_UI16_PARAMS const int8_t *src, uint16_t *dst, int32_t nelems
#define I8_UI8_PARAMS const int8_t *src, uint8_t *dst, int32_t nelems
#define I8_B_PARAMS const int8_t *src, bool *dst, int32_t nelems

#define UI64_D_PARAMS const uint64_t *src, double *dst, int32_t nelems
#define UI64_I64_PARAMS const uint64_t *src, int64_t *dst, int32_t nelems

#define UI32_D_PARAMS const uint32_t *src, double *dst, int32_t nelems
#define UI32_F_PARAMS const uint32_t *src, float *dst, int32_t nelems
#define UI32_I64_PARAMS const uint32_t *src, int64_t *dst, int32_t nelems
#define UI32_I32_PARAMS const uint32_t *src, int32_t *dst, int32_t nelems
#define UI32_UI64_PARAMS const uint32_t *src, uint64_t *dst, int32_t nelems

#define UI16_D_PARAMS const uint16_t *src, double *dst, int32_t nelems
#define UI16_F_PARAMS const uint16_t *src, float *dst, int32_t nelems
#define UI16_I64_PARAMS const uint16_t *src, int64_t *dst, int32_t nelems
#define UI16_I32_PARAMS const uint16_t *src, int32_t *dst, int32_t nelems
#define UI16_I16_PARAMS const uint16_t *src, int16_t *dst, int32_t nelems
#define UI16_UI64_PARAMS const uint16_t *src, uint64_t *dst, int32_t nelems
#define UI16_UI32_PARAMS const uint16_t *src, uint32_t *dst, int32_t nelems

#define UI8_D_PARAMS const uint8_t *src, double *dst, int32_t nelems
#define UI8_F_PARAMS const uint8_t *src, float *dst, int32_t nelems
#define UI8_I64_PARAMS const uint8_t *src, int64_t *dst, int32_t nelems
#define UI8_I32_PARAMS const uint8_t *src, int32_t *dst, int32_t nelems
#define UI8_I16_PARAMS const uint8_t *src, int16_t *dst, int32_t nelems
#define UI8_I8_PARAMS const uint8_t *src, int8_t *dst, int32_t nelems
#define UI8_UI64_PARAMS const uint8_t *src, uint64_t *dst, int32_t nelems
#define UI8_UI32_PARAMS const uint8_t *src, uint32_t *dst, int32_t nelems
#define UI8_UI16_PARAMS const uint8_t *src, uint16_t *dst, int32_t nelems
#define UI8_B_PARAMS const uint8_t *src, bool *dst, int32_t nelems

#define B_D_PARAMS const bool *src, double *dst, int32_t nelems
#define B_F_PARAMS const bool *src, float *dst, int32_t nelems
#define B_I64_PARAMS const bool *src, int64_t *dst, int32_t nelems
#define B_I32_PARAMS const bool *src, int32_t *dst, int32_t nelems
#define B_I16_PARAMS const bool *src, int16_t *dst, int32_t nelems
#define B_I8_PARAMS const bool *src, int8_t *dst, int32_t nelems
#define B_UI64_PARAMS const bool *src, uint64_t *dst, int32_t nelems
#define B_UI32_PARAMS const bool *src, uint32_t *dst, int32_t nelems
#define B_UI16_PARAMS const bool *src, uint16_t *dst, int32_t nelems
#define B_UI8_PARAMS const bool *src, uint8_t *dst, int32_t nelems


#define VIEW_CAST \
    for (int i = 0; i < nelems; ++i) { \
        *dst = *src;    \
        dst++;  \
        src++;  \
    }

static void d_i64_cast(D_I64_PARAMS) { VIEW_CAST }
static void d_ui64_cast(D_UI64_PARAMS) { VIEW_CAST }

static void f_d_cast(F_D_PARAMS) { VIEW_CAST }
static void f_i64_cast(F_I64_PARAMS) { VIEW_CAST }
static void f_i32_cast(F_I32_PARAMS) { VIEW_CAST }
static void f_ui64_cast(F_UI64_PARAMS) { VIEW_CAST }
static void f_ui32_cast(F_UI32_PARAMS) { VIEW_CAST }

static void i64_d_cast(I64_D_PARAMS) { VIEW_CAST }
static void i64_ui64_cast(I64_UI64_PARAMS) { VIEW_CAST }

static void i32_d_cast(I32_D_PARAMS) { VIEW_CAST }
static void i32_f_cast(I32_F_PARAMS) { VIEW_CAST }
static void i32_i64_cast(I32_I64_PARAMS) { VIEW_CAST }
static void i32_ui64_cast(I32_UI64_PARAMS) { VIEW_CAST }
static void i32_ui32_cast(I32_UI32_PARAMS) { VIEW_CAST }

static void i16_d_cast(I16_D_PARAMS) { VIEW_CAST }
static void i16_f_cast(I16_F_PARAMS) { VIEW_CAST }
static void i16_i64_cast(I16_I64_PARAMS) { VIEW_CAST }
static void i16_i32_cast(I16_I32_PARAMS) { VIEW_CAST }
static void i16_ui64_cast(I16_UI64_PARAMS) { VIEW_CAST }
static void i16_ui32_cast(I16_UI32_PARAMS) { VIEW_CAST }
static void i16_ui16_cast(I16_UI16_PARAMS) { VIEW_CAST }

static void i8_d_cast(I8_D_PARAMS) { VIEW_CAST }
static void i8_f_cast(I8_F_PARAMS) { VIEW_CAST }
static void i8_i64_cast(I8_I64_PARAMS) { VIEW_CAST }
static void i8_i32_cast(I8_I32_PARAMS) { VIEW_CAST }
static void i8_i16_cast(I8_I16_PARAMS) { VIEW_CAST }
static void i8_ui64_cast(I8_UI64_PARAMS) { VIEW_CAST }
static void i8_ui32_cast(I8_UI32_PARAMS) { VIEW_CAST }
static void i8_ui16_cast(I8_UI16_PARAMS) { VIEW_CAST }
static void i8_ui8_cast(I8_UI8_PARAMS) { VIEW_CAST }
static void i8_b_cast(I8_B_PARAMS) { VIEW_CAST }

static void ui64_d_cast(UI64_D_PARAMS) { VIEW_CAST }
static void ui64_i64_cast(UI64_I64_PARAMS) { VIEW_CAST }

static void ui32_d_cast(UI32_D_PARAMS) { VIEW_CAST }
static void ui32_f_cast(UI32_F_PARAMS) { VIEW_CAST }
static void ui32_i64_cast(UI32_I64_PARAMS) { VIEW_CAST }
static void ui32_i32_cast(UI32_I32_PARAMS) { VIEW_CAST }
static void ui32_ui64_cast(UI32_UI64_PARAMS) { VIEW_CAST }

static void ui16_d_cast(UI16_D_PARAMS) { VIEW_CAST }
static void ui16_f_cast(UI16_F_PARAMS) { VIEW_CAST }
static void ui16_i64_cast(UI16_I64_PARAMS) { VIEW_CAST }
static void ui16_i32_cast(UI16_I32_PARAMS) { VIEW_CAST }
static void ui16_i16_cast(UI16_I16_PARAMS) { VIEW_CAST }
static void ui16_ui64_cast(UI16_UI64_PARAMS) { VIEW_CAST }
static void ui16_ui32_cast(UI16_UI32_PARAMS) { VIEW_CAST }

static void ui8_d_cast(UI8_D_PARAMS) { VIEW_CAST }
static void ui8_f_cast(UI8_F_PARAMS) { VIEW_CAST }
static void ui8_i64_cast(UI8_I64_PARAMS) { VIEW_CAST }
static void ui8_i32_cast(UI8_I32_PARAMS) { VIEW_CAST }
static void ui8_i16_cast(UI8_I16_PARAMS) { VIEW_CAST }
static void ui8_i8_cast(UI8_I8_PARAMS) { VIEW_CAST }
static void ui8_ui64_cast(UI8_UI64_PARAMS) { VIEW_CAST }
static void ui8_ui32_cast(UI8_UI32_PARAMS) { VIEW_CAST }
static void ui8_ui16_cast(UI8_UI16_PARAMS) { VIEW_CAST }
static void ui8_b_cast(UI8_B_PARAMS) { VIEW_CAST }

static void b_d_cast(B_D_PARAMS) { VIEW_CAST }
static void b_f_cast(B_F_PARAMS) { VIEW_CAST }
static void b_i64_cast(B_I64_PARAMS) { VIEW_CAST }
static void b_i32_cast(B_I32_PARAMS) { VIEW_CAST }
static void b_i16_cast(B_I16_PARAMS) { VIEW_CAST }
static void b_i8_cast(B_I8_PARAMS) { VIEW_CAST }
static void b_ui64_cast(B_UI64_PARAMS) { VIEW_CAST }
static void b_ui32_cast(B_UI32_PARAMS) { VIEW_CAST }
static void b_ui16_cast(B_UI16_PARAMS) { VIEW_CAST }
static void b_ui8_cast(B_UI8_PARAMS) { VIEW_CAST }


ina_rc_t prop1_view_postfilter(blosc2_postfilter_params *postparams)
{
    view_postparams_udata *udata = postparams->user_data;

    uint8_t *chunk;
    bool needs_free;
    int csize = blosc2_schunk_get_lazychunk(udata->viewed_schunk, postparams->nchunk, &chunk,
                                            &needs_free);
    if (csize < 0) {
        IARRAY_TRACE1(iarray.tracing, "Error getting lazy chunk");
        return IARRAY_ERR_BLOSC_FAILED;
    }

    blosc2_dparams dparams = {.schunk = udata->viewed_schunk,
                              .postfilter=udata->viewed_schunk->dctx->postfilter,
                              .postparams=udata->viewed_schunk->dctx->postparams};
    blosc2_context *dctx = blosc2_create_dctx(dparams);

    int block_nitems = postparams->size / postparams->typesize;
    uint8_t *aux = malloc(postparams->size  / 1);
    int bsize = blosc2_getitem_ctx(dctx, chunk, csize,
                                   postparams->nblock * block_nitems,
                                   block_nitems, aux, postparams->size / 1);
    if (needs_free) {
        free(chunk);
    }
    if (bsize < 0) {
        blosc2_free_ctx(dctx);
        IARRAY_TRACE1(iarray.tracing, "Error getting block");
        return IARRAY_ERR_BLOSC_FAILED;
    }
    udata->cast(aux, postparams->out, block_nitems);

    free(aux);
    blosc2_free_ctx(dctx);

    return INA_SUCCESS;
}

ina_rc_t prop2_view_postfilter(blosc2_postfilter_params *postparams)
{
    view_postparams_udata *udata = postparams->user_data;

    uint8_t *chunk;
    bool needs_free;
    int csize = blosc2_schunk_get_lazychunk(udata->viewed_schunk, postparams->nchunk, &chunk,
                                            &needs_free);
    if (csize < 0) {
        IARRAY_TRACE1(iarray.tracing, "Error getting lazy chunk");
        return IARRAY_ERR_BLOSC_FAILED;
    }

    blosc2_dparams dparams = {.schunk = udata->viewed_schunk,
                              .postfilter=udata->viewed_schunk->dctx->postfilter,
                              .postparams=udata->viewed_schunk->dctx->postparams};
    blosc2_context *dctx = blosc2_create_dctx(dparams);

    int block_nitems = postparams->size / postparams->typesize;
    uint8_t *aux = malloc(postparams->size  / 2);
    int bsize = blosc2_getitem_ctx(dctx, chunk, csize,
                                   postparams->nblock * block_nitems,
                                   block_nitems, aux, postparams->size / 2);
    if (needs_free) {
        free(chunk);
    }
    if (bsize < 0) {
        blosc2_free_ctx(dctx);
        IARRAY_TRACE1(iarray.tracing, "Error getting block");
        return IARRAY_ERR_BLOSC_FAILED;
    }
    udata->cast(aux, postparams->out, block_nitems);

    free(aux);
    blosc2_free_ctx(dctx);

    return INA_SUCCESS;
}

ina_rc_t prop4_view_postfilter(blosc2_postfilter_params *postparams)
{
    view_postparams_udata *udata = postparams->user_data;

    uint8_t *chunk;
    bool needs_free;
    int csize = blosc2_schunk_get_lazychunk(udata->viewed_schunk, postparams->nchunk, &chunk,
                                            &needs_free);
    if (csize < 0) {
        IARRAY_TRACE1(iarray.tracing, "Error getting lazy chunk");
        return IARRAY_ERR_BLOSC_FAILED;
    }

    blosc2_dparams dparams = {.schunk = udata->viewed_schunk,
                              .postfilter=udata->viewed_schunk->dctx->postfilter,
                              .postparams=udata->viewed_schunk->dctx->postparams};
    blosc2_context *dctx = blosc2_create_dctx(dparams);

    int block_nitems = postparams->size / postparams->typesize;
    uint8_t *aux = malloc(postparams->size  / 4);
    int bsize = blosc2_getitem_ctx(dctx, chunk, csize,
                                   postparams->nblock * block_nitems,
                                   block_nitems, aux, postparams->size / 4);
    if (needs_free) {
        free(chunk);
    }
    if (bsize < 0) {
        blosc2_free_ctx(dctx);
        IARRAY_TRACE1(iarray.tracing, "Error getting block");
        return IARRAY_ERR_BLOSC_FAILED;
    }
    udata->cast(aux, postparams->out, block_nitems);

    free(aux);
    blosc2_free_ctx(dctx);

    return INA_SUCCESS;
}

ina_rc_t prop8_view_postfilter(blosc2_postfilter_params *postparams)
{
    view_postparams_udata *udata = postparams->user_data;

    uint8_t *chunk;
    bool needs_free;
    int csize = blosc2_schunk_get_lazychunk(udata->viewed_schunk, postparams->nchunk, &chunk,
                                            &needs_free);
    if (csize < 0) {
        IARRAY_TRACE1(iarray.tracing, "Error getting lazy chunk");
        return IARRAY_ERR_BLOSC_FAILED;
    }

    blosc2_dparams dparams = {.schunk = udata->viewed_schunk,
                              .postfilter=udata->viewed_schunk->dctx->postfilter,
                              .postparams=udata->viewed_schunk->dctx->postparams};
    blosc2_context *dctx = blosc2_create_dctx(dparams);

    int block_nitems = postparams->size / postparams->typesize;
    uint8_t *aux = malloc(postparams->size  / 8);
    int bsize = blosc2_getitem_ctx(dctx, chunk, csize,
                                   postparams->nblock * block_nitems,
                                   block_nitems, aux, postparams->size / 8);
    if (needs_free) {
        free(chunk);
    }
    if (bsize < 0) {
        blosc2_free_ctx(dctx);
        IARRAY_TRACE1(iarray.tracing, "Error getting block");
        return IARRAY_ERR_BLOSC_FAILED;
    }
    udata->cast(aux, postparams->out, block_nitems);

    free(aux);
    blosc2_free_ctx(dctx);

    return INA_SUCCESS;
}

ina_rc_t slice_view_postfilter(blosc2_postfilter_params *postparams)
{
    view_postparams_udata *udata = postparams->user_data;

    uint8_t *chunk;
    bool needs_free;
    int csize = blosc2_schunk_get_lazychunk(udata->viewed_schunk, postparams->nchunk, &chunk,
                                            &needs_free);
    if (csize < 0) {
        IARRAY_TRACE1(iarray.tracing, "Error getting lazy chunk");
        return IARRAY_ERR_BLOSC_FAILED;
    }

    blosc2_dparams dparams = {.schunk = udata->viewed_schunk,
                              .postfilter=udata->viewed_schunk->dctx->postfilter,
                              .postparams=udata->viewed_schunk->dctx->postparams};
    blosc2_context *dctx = blosc2_create_dctx(dparams);

    int block_nitems = udata->viewed_schunk->blocksize / postparams->typesize;
    int bsize = blosc2_getitem_ctx(dctx, chunk, csize,
                                   postparams->nblock * block_nitems,
                                   block_nitems, postparams->out, postparams->size);
    if (needs_free) {
        free(chunk);
    }
    if (bsize < 0) {
        blosc2_free_ctx(dctx);
        IARRAY_TRACE1(iarray.tracing, "Error getting block");
        return IARRAY_ERR_BLOSC_FAILED;
    }
    blosc2_free_ctx(dctx);

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_add_view_postfilter(iarray_container_t *view)
{
    INA_VERIFY_NOT_NULL(view);
    // Create params
    blosc2_dparams *dparams;
    blosc2_schunk_get_dparams(view->catarr->sc, &dparams);

    blosc2_postfilter_params *postparams = malloc(sizeof(blosc2_postfilter_params));
    view_postparams_udata *view_postparams = malloc(sizeof(view_postparams_udata));

    view_postparams->viewed_schunk = view->container_viewed->catarr->sc;

    if (view->dtshape->dtype == view->container_viewed->dtshape->dtype) {
        dparams->postfilter = (blosc2_postfilter_fn) slice_view_postfilter;
        view_postparams->cast = NULL;
    }
    else {
        int prop = view->dtshape->dtype_size / view->container_viewed->dtshape->dtype_size;

        switch (prop) {
            case 1:
                dparams->postfilter = (blosc2_postfilter_fn) prop1_view_postfilter;
                break;
            case 2:
                dparams->postfilter = (blosc2_postfilter_fn) prop2_view_postfilter;
                break;
            case 4:
                dparams->postfilter = (blosc2_postfilter_fn) prop4_view_postfilter;
                break;
            case 8:
                dparams->postfilter = (blosc2_postfilter_fn) prop8_view_postfilter;
                break;
            default:
                return IARRAY_ERR_INVALID_DTYPE;
        }
        switch (view->container_viewed->dtshape->dtype) {
            case IARRAY_DATA_TYPE_DOUBLE: {
                switch (view->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_INT64:
                        view_postparams->cast = CAST d_i64_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        view_postparams->cast = CAST d_ui64_cast;
                        break;
                    default:
                        goto fail;
                }
                break;
            }
            case IARRAY_DATA_TYPE_FLOAT: {
                switch (view->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        view_postparams->cast = CAST f_d_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        view_postparams->cast = CAST f_i64_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT32:
                        view_postparams->cast = CAST f_i32_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        view_postparams->cast = CAST f_ui64_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT32:
                        view_postparams->cast = CAST f_ui32_cast;
                        break;
                    default:
                        goto fail;
                }
                break;
            }
            case IARRAY_DATA_TYPE_INT64: {
                switch (view->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE: {
                        view_postparams->cast = CAST i64_d_cast;
                        break;
                    }
                    case IARRAY_DATA_TYPE_UINT64:
                        view_postparams->cast = CAST i64_ui64_cast;
                        break;
                    default:
                        goto fail;
                }
                break;
            }
            case IARRAY_DATA_TYPE_INT32: {
                switch (view->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        view_postparams->cast = CAST i32_d_cast;
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        view_postparams->cast = CAST i32_f_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        view_postparams->cast = CAST i32_i64_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        view_postparams->cast = CAST i32_ui64_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT32:
                        view_postparams->cast = CAST i32_ui32_cast;
                        break;
                    default:
                        goto fail;
                }
                break;
            }
            case IARRAY_DATA_TYPE_INT16: {
                switch (view->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        view_postparams->cast = CAST i16_d_cast;
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        view_postparams->cast = CAST i16_f_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        view_postparams->cast = CAST i16_i64_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT32:
                        view_postparams->cast = CAST i16_i32_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        view_postparams->cast = CAST i16_ui64_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT32:
                        view_postparams->cast = CAST i16_ui32_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT16:
                        view_postparams->cast = CAST i16_ui16_cast;
                        break;
                    default:
                        goto fail;
                }
                break;
            }
            case IARRAY_DATA_TYPE_INT8: {
                switch (view->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        view_postparams->cast = CAST i8_d_cast;
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        view_postparams->cast = CAST i8_f_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        view_postparams->cast = CAST i8_i64_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT32:
                        view_postparams->cast = CAST i8_i32_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT16:
                        view_postparams->cast = CAST i8_i16_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        view_postparams->cast = CAST i8_ui64_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT32:
                        view_postparams->cast = CAST i8_ui32_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT16:
                        view_postparams->cast = CAST i8_ui16_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT8:
                        view_postparams->cast = CAST i8_ui8_cast;
                        break;
                    case IARRAY_DATA_TYPE_BOOL:
                        view_postparams->cast = CAST i8_b_cast;
                        break;
                    default:
                        goto fail;
                }
                break;
            }
            case IARRAY_DATA_TYPE_UINT64: {
                switch (view->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        view_postparams->cast = CAST ui64_d_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        view_postparams->cast = CAST ui64_i64_cast;
                        break;
                    default:
                        goto fail;
                }
                break;
            }
            case IARRAY_DATA_TYPE_UINT32: {
                switch (view->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        view_postparams->cast = CAST ui32_d_cast;
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        view_postparams->cast = CAST ui32_f_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        view_postparams->cast = CAST ui32_i64_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT32:
                        view_postparams->cast = CAST ui32_i32_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        view_postparams->cast = CAST ui32_ui64_cast;
                        break;
                    default:
                        goto fail;
                }
                break;
            }
            case IARRAY_DATA_TYPE_UINT16: {
                switch (view->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        view_postparams->cast = CAST ui16_d_cast;
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        view_postparams->cast = CAST ui16_f_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        view_postparams->cast = CAST ui16_i64_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT32:
                        view_postparams->cast = CAST ui16_i32_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT16:
                        view_postparams->cast = CAST ui16_i16_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        view_postparams->cast = CAST ui16_ui64_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT32:
                        view_postparams->cast = CAST ui16_ui32_cast;
                        break;
                    default:
                        goto fail;
                }
                break;
            }
            case IARRAY_DATA_TYPE_UINT8: {
                switch (view->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        view_postparams->cast = CAST ui8_d_cast;
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        view_postparams->cast = CAST ui8_f_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        view_postparams->cast = CAST ui8_i64_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT32:
                        view_postparams->cast = CAST ui8_i32_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT16:
                        view_postparams->cast = CAST ui8_i16_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT8:
                        view_postparams->cast = CAST ui8_i8_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        view_postparams->cast = CAST ui8_ui64_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT32:
                        view_postparams->cast = CAST ui8_ui32_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT16:
                        view_postparams->cast = CAST ui8_ui16_cast;
                        break;
                    case IARRAY_DATA_TYPE_BOOL:
                        view_postparams->cast = CAST ui8_b_cast;
                        break;
                    default:
                        goto fail;
                }
                break;
            }
            case IARRAY_DATA_TYPE_BOOL: {
                switch (view->dtshape->dtype) {
                    case IARRAY_DATA_TYPE_DOUBLE:
                        view_postparams->cast = CAST b_d_cast;
                        break;
                    case IARRAY_DATA_TYPE_FLOAT:
                        view_postparams->cast = CAST b_f_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT64:
                        view_postparams->cast = CAST b_i64_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT32:
                        view_postparams->cast = CAST b_i32_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT16:
                        view_postparams->cast = CAST b_i16_cast;
                        break;
                    case IARRAY_DATA_TYPE_INT8:
                        view_postparams->cast = CAST b_i8_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT64:
                        view_postparams->cast = CAST b_ui64_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT32:
                        view_postparams->cast = CAST b_ui32_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT16:
                        view_postparams->cast = CAST b_ui16_cast;
                        break;
                    case IARRAY_DATA_TYPE_UINT8:
                        view_postparams->cast = CAST b_ui8_cast;
                        break;
                    default:
                        goto fail;
                }
                break;
            }
            default:
                goto fail;
        }
    }

    postparams->user_data = (void*)view_postparams;
    dparams->postparams = postparams;

    // Create new context since postparams is empty in the old one
    blosc2_free_ctx(view->catarr->sc->dctx);
    view->catarr->sc->dctx = blosc2_create_dctx(*dparams);
    free(dparams);

    return INA_SUCCESS;

fail:
    IARRAY_TRACE1(iarray.error, "Only upcasting is supported");
    return IARRAY_ERR_INVALID_DTYPE;

}
