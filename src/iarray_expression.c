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
#include <minjugg.h>
#include <caterva_blosc.h>

#if defined(_OPENMP)
#include <omp.h>
#endif


typedef struct _iarray_jug_var_s {
    const char *var;
    iarray_container_t *c;
} _iarray_jug_var_t;


typedef enum iarray_expr_input_class_e {
    IARRAY_EXPR_EQ = 0u,  // Same chunkshape/blockshape
    IARRAY_EXPR_EQ_NCOMP = 1u, // Same chunkshape/blockshape and no-compressed data
    IARRAY_EXPR_NEQ = 2u  // Different chunkshape/blockshape
} iarray_expr_input_class_t;


struct iarray_expression_s {
    iarray_context_t *ctx;
    ina_str_t expr;
    int32_t typesize;
    int64_t nbytes;
    int nvars;
    int32_t max_out_len;
    jug_expression_t *jug_expr;
    uint64_t jug_expr_func;
    iarray_dtshape_t *out_dtshape;
    iarray_storage_t *out_store_properties;
    iarray_container_t *out;
    _iarray_jug_var_t vars[IARRAY_EXPR_OPERANDS_MAX];
};

// Struct to be used as info container for dealing with the expression
typedef struct iarray_expr_pparams_s {
    bool compressed_inputs;
    int ninputs;  // number of data inputs
    iarray_expr_input_class_t input_class[IARRAY_EXPR_OPERANDS_MAX];  // whether the inputs are compressed or not
    uint8_t* inputs[IARRAY_EXPR_OPERANDS_MAX];  // the data inputs
    int32_t input_csizes[IARRAY_EXPR_OPERANDS_MAX];  // the compressed size of data inputs
    int32_t input_typesizes[IARRAY_EXPR_OPERANDS_MAX];  // the typesizes for data inputs
    iarray_expression_t *e;
    iarray_iter_write_block_value_t out_value;
} iarray_expr_pparams_t;

// Struct to be used as argument to the evaluation function
typedef struct iarray_eval_pparams_s {
    int ninputs;  // number of data inputs
    uint8_t* inputs[IARRAY_EXPR_OPERANDS_MAX];  // the data inputs
    int32_t input_typesizes[IARRAY_EXPR_OPERANDS_MAX];  // the typesizes for data inputs
    void *user_data;  // a pointer to an iarray_expr_pparams_t struct
    uint8_t *out;  // the output buffer
    int32_t out_size;  // the size of output buffer (in bytes)
    int32_t out_typesize;  // the typesize of output
    int8_t ndim;  // the number of dimensions for inputs / output arrays
    int32_t *window_shape;  // the shape of the window for the input arrays (NULL if not available)
    int64_t *window_start; // the start coordinates for the window shape (NULL if not available)
    int32_t *window_strides; // the strides for the window shape (NULL if not available)
} iarray_eval_pparams_t;

typedef int (*iarray_eval_fn)(iarray_eval_pparams_t *params);

INA_API(ina_rc_t) iarray_expr_new(iarray_context_t *ctx, iarray_expression_t **e)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(e);
    *e = ina_mem_alloc(sizeof(iarray_expression_t));
    INA_RETURN_IF_NULL(e);
    (*e)->ctx = ctx;
    (*e)->expr = NULL;
    (*e)->nvars = 0;
    (*e)->max_out_len = 0;   // helper for leftovers
    ina_mem_set(&(*e)->vars, 0, sizeof(_iarray_jug_var_t) * IARRAY_EXPR_OPERANDS_MAX);
    jug_expression_new(&(*e)->jug_expr);
    return INA_SUCCESS;
}

INA_API(void) iarray_expr_free(iarray_context_t *ctx, iarray_expression_t **e)
{
    INA_VERIFY_FREE(e);
    if ((*e)->jug_expr != NULL) {
        jug_expression_free(&(*e)->jug_expr);
    }
    for (int nvar=0; nvar < (*e)->nvars; nvar++) {
        free((void*)((*e)->vars[nvar].var));
    }
    ina_mempool_reset(ctx->mp);  // FIXME: should be ina_mempool_free(), but it currently crashes
    ina_mempool_reset(ctx->mp_op);  // FIXME: ditto
    ina_mempool_reset(ctx->mp_tmp_out);  // FIXME: ditto
    ina_str_free((*e)->expr);
    INA_MEM_FREE_SAFE(*e);
}

INA_API(ina_rc_t) iarray_expr_bind(iarray_expression_t *e, const char *var, iarray_container_t *val)
{
    INA_VERIFY_NOT_NULL(e);
    INA_VERIFY_NOT_NULL(var);
    INA_VERIFY_NOT_NULL(val);

    e->vars[e->nvars].var = strdup(var);   // yes, we want a copy here!
    e->vars[e->nvars].c = val;
    e->nvars++;
    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_expr_bind_out_properties(iarray_expression_t *e, iarray_dtshape_t *dtshape, iarray_storage_t *store)
{
    e->out_dtshape = ina_mem_alloc(sizeof(iarray_dtshape_t));
    ina_mem_cpy(e->out_dtshape, dtshape, sizeof(iarray_dtshape_t));

    if (store == NULL) {
        e->out_store_properties = NULL;
    }
    else {
        e->out_store_properties = ina_mem_alloc(sizeof(iarray_storage_t));
        ina_mem_cpy(e->out_store_properties, store, sizeof(iarray_storage_t));
        if (store->urlpath != NULL) {
            e->out_store_properties->urlpath = strdup(store->urlpath);
        } else {
            e->out_store_properties->urlpath = NULL;
        }
    }
    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_expr_bind_scalar_float(iarray_expression_t *e, const char *var, float val)
//{
//  iarray_container_t *c = ina_mempool_dalloc(e->mp, sizeof(iarray_container_t));
//  c->dtshape = ina_mempool_dalloc(e->mp, sizeof(iarray_dtshape_t));
//  c->dtshape->ndim = 0;
//  c->dtshape->dims = NULL;
//  c->dtshape->dtype = IARRAY_DATA_TYPE_FLOAT;
//  c->scalar_value.f = val;
//  return INA_SUCCESS;
//}
{
    INA_UNUSED(e);
    INA_UNUSED(var);
    INA_UNUSED(val);
    return INA_ERROR(INA_ERR_NOT_IMPLEMENTED);
}

INA_API(ina_rc_t) iarray_expr_bind_scalar_double(iarray_expression_t *e, const char *var, double val)
//{
//    iarray_container_t *c = ina_mempool_dalloc(e->ctx->mp, sizeof(iarray_container_t));
//    c->dtshape = ina_mempool_dalloc(e->ctx->mp, sizeof(iarray_dtshape_t));
//    c->dtshape->ndim = 0;
//    c->dtshape->dtype = IARRAY_DATA_TYPE_DOUBLE;
//    c->scalar_value.d = val;
//    e->vars[e->nvars].var = var;
//    e->vars[e->nvars].c = c;
//    e->nvars++;
//    return INA_SUCCESS;
//}
{
    INA_UNUSED(e);
    INA_UNUSED(var);
    INA_UNUSED(val);
    return INA_ERROR(INA_ERR_NOT_IMPLEMENTED);
}


static ina_rc_t _iarray_expr_prepare(iarray_expression_t *e)
{
    uint32_t eval_method = e->ctx->cfg->eval_method & 0x3u;

    if (eval_method == IARRAY_EVAL_METHOD_ITERBLOSC) {
        if (e->out_store_properties->backend == IARRAY_STORAGE_PLAINBUFFER) {
            return INA_ERROR(IARRAY_ERR_INVALID_EVAL_METHOD);
        }
    }
    if (eval_method == IARRAY_EVAL_METHOD_AUTO) {
        if (e->out_store_properties->backend == IARRAY_STORAGE_PLAINBUFFER) {
            eval_method = IARRAY_EVAL_METHOD_ITERCHUNK;
        } else {
            eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
        }
    }

    e->ctx->cfg->eval_method = eval_method;

    switch (e->out_dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            e->typesize = sizeof(double);
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            e->typesize = sizeof(float);
            break;
        default:
            return INA_ERROR(IARRAY_ERR_INVALID_DTYPE);
    }

    size_t size;
    IARRAY_RETURN_IF_FAILED(iarray_shape_size(e->out_dtshape, &size));
    e->nbytes = size * e->typesize;


    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_expr_compile_udf(
    iarray_expression_t *e,
    int llvm_bc_len,
    const char *llvm_bc,
    const char* name)
{
    INA_VERIFY_NOT_NULL(e);
    INA_VERIFY_NOT_NULL(llvm_bc);

    IARRAY_RETURN_IF_FAILED(_iarray_expr_prepare(e));

    IARRAY_RETURN_IF_FAILED(
        jug_udf_compile(e->jug_expr, llvm_bc_len, llvm_bc, name, &e->jug_expr_func)
    );

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_expr_compile(iarray_expression_t *e, const char *expr)
{
    INA_VERIFY_NOT_NULL(e);
    INA_VERIFY_NOT_NULL(expr);

    e->expr = ina_str_new_fromcstr(expr);

    IARRAY_RETURN_IF_FAILED(_iarray_expr_prepare(e));

    jug_te_variable *jug_vars = ina_mempool_dalloc(e->ctx->mp, e->nvars * sizeof(jug_te_variable));
    memset(jug_vars, 0, e->nvars * sizeof(jug_te_variable));
    for (int nvar = 0; nvar < e->nvars; nvar++) {
        jug_vars[nvar].name = e->vars[nvar].var;
    }

    IARRAY_RETURN_IF_FAILED(jug_expression_compile(e->jug_expr, ina_str_cstr(e->expr), e->nvars,
                                                      jug_vars, e->typesize, &e->jug_expr_func));

    return INA_SUCCESS;
}

int prefilter_func(blosc2_prefilter_params *pparams)
{
    iarray_expr_pparams_t *expr_pparams = (iarray_expr_pparams_t*)pparams->user_data;
    struct iarray_expression_s *e = expr_pparams->e;
    int ninputs = expr_pparams->ninputs;
    // Populate the eval_pparams
    iarray_eval_pparams_t eval_pparams = {0};
    eval_pparams.ninputs = ninputs;
    memcpy(eval_pparams.input_typesizes, expr_pparams->input_typesizes, ninputs * sizeof(int32_t));
    eval_pparams.user_data = expr_pparams;
    eval_pparams.out = pparams->out;
    eval_pparams.out_size = pparams->out_size;
    eval_pparams.out_typesize = pparams->out_typesize;
    eval_pparams.ndim = expr_pparams->e->out_dtshape->ndim;
    int32_t blocksize = pparams->out_size;
    int32_t typesize = pparams->out_typesize;

    int8_t ndim = e->out->dtshape->ndim;

    // Element strides (in elements)
    int32_t strides[IARRAY_DIMENSION_MAX];
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0 ; --i) {
        strides[i] = strides[i+1] * e->out->catarr->blockshape[i+1];
    }

    // Block strides (in blocks)
    int32_t strides_block[IARRAY_DIMENSION_MAX];
    strides_block[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0 ; --i) {
        strides_block[i] = strides_block[i+1] * (int32_t) (e->out->catarr->extchunkshape[i+1] / e->out->catarr->blockshape[i+1]);
    }

    // Flattened block number
    int32_t nblock = pparams->out_offset / pparams->out_size;

    // Multidimensional block number
    int32_t nblock_ndim[IARRAY_DIMENSION_MAX];
    for (int i = ndim - 1; i >= 0; --i) {
        if (i != 0) {
            nblock_ndim[i] = (nblock % strides_block[i-1]) / strides_block[i];
        } else {
            nblock_ndim[i] = (nblock % (e->out->catarr->extchunknitems / e->out->catarr->blocknitems)) / strides_block[i];
        }
    }

    // Position of the first element of the block (inside current chunk)
    int64_t start_in_chunk[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < ndim; ++i) {
        start_in_chunk[i] = nblock_ndim[i] * e->out->catarr->blockshape[i];
    }

    // Position of the first element of the block (inside container)
    int64_t start_in_container[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < ndim; ++i) {
        start_in_container[i] = start_in_chunk[i] + expr_pparams->out_value.block_index[i] * e->out->catarr->chunkshape[i];
    }

    // Check if the block is out of bounds
    bool out_of_bounds = false;
    for (int i = 0; i < ndim; ++i) {
        if (start_in_container[i] > e->out->catarr->shape[i]) {
            out_of_bounds = true;
            break;
        }
    }

    // Shape of the current block
    int32_t shape[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < ndim; ++i) {
        if (out_of_bounds) {
            shape[i] = 0;
        } else if (start_in_container[i] + e->out->catarr->blockshape[i] > e->out->catarr->shape[i]) {
            shape[i] = (int32_t) (e->out->catarr->shape[i] - start_in_container[i]);
        } else if (start_in_chunk[i] + e->out->catarr->blockshape[i] > e->out->catarr->chunkshape[i]) {
            shape[i] = (int32_t) (e->out->catarr->chunkshape[i] - start_in_chunk[i]);
        } else {
            shape[i] = e->out->catarr->blockshape[i];
        }
    }

    // We can only set the visible shape of the output for the ITERBLOSC eval method.
    eval_pparams.window_shape = shape;
    eval_pparams.window_start = start_in_container;
    eval_pparams.window_strides = strides;

    // The code below only works for the case where inputs and output have the same typesize.
    // More love is needed in the future, where we would want to allow mixed types in expressions.

    bool inputs_malloced[IARRAY_DIMENSION_MAX];
    for (int i = 0; i < ninputs; i++) {
        inputs_malloced[i] = false;
        switch (expr_pparams->input_class[i]) {
            case IARRAY_EXPR_EQ_NCOMP:
                eval_pparams.inputs[i] = expr_pparams->inputs[i] + BLOSC_EXTENDED_HEADER_LENGTH + pparams->out_offset;
                break;
            case IARRAY_EXPR_EQ:
                eval_pparams.inputs[i] = ina_mem_alloc_aligned(64, blocksize);
                inputs_malloced[i] = true;
                int64_t nitems = blocksize / typesize;
                int64_t offset_index = pparams->out_offset / typesize;
                blosc2_dparams dparams = {.nthreads = 1, .schunk = e->vars[i].c->catarr->sc};
                blosc2_context *dctx = blosc2_create_dctx(dparams);
                int64_t rbytes = blosc2_getitem_ctx(dctx, expr_pparams->inputs[i], expr_pparams->input_csizes[i],
                                                    (int) offset_index, (int) nitems,
                                                    eval_pparams.inputs[i], blocksize);
                blosc2_free_ctx(dctx);
                if (rbytes != blocksize) {
                    fprintf(stderr, "Read from inputs failed inside pipeline\n");
                    return -1;
                }
                break;
            case IARRAY_EXPR_NEQ:
                eval_pparams.inputs[i] = expr_pparams->inputs[i] + pparams->out_offset;
                break;
            default:
                return -1;
        }
    }

    // Eval the expression for this chunk
    int ret;

    ret = ((iarray_eval_fn) e->jug_expr_func)(&eval_pparams);
    switch (ret) {
        case 0:
            // 0 means success
            break;
        case 1:
            IARRAY_TRACE1(iarray.error, "Out of bounds in LLVM eval engine");
            return -2;
        default:
            IARRAY_TRACE1(iarray.error, "Error in executing LLVM eval engine");
            return -3;
    }

    for (int i = 0; i < ninputs; i++) {
        if (inputs_malloced[i]) {
            INA_MEM_FREE_SAFE(eval_pparams.inputs[i]);
        }
    }

    return 0;
}


int postfilter_func(blosc2_postfilter_params *pparams) {
    // pparams is private for every thread when it arrives here
    iarray_expr_pparams_t *expr_pparams = (iarray_expr_pparams_t *) pparams->user_data;
    struct iarray_expression_s *e = expr_pparams->e;
    int ninputs = expr_pparams->ninputs;
    // Populate the eval_pparams
    iarray_eval_pparams_t eval_pparams = {0};
    // A postfilter only accepts one input for now
    eval_pparams.ninputs = 1;
    eval_pparams.inputs[0] = (uint8_t*)pparams->in;
    memcpy(eval_pparams.input_typesizes, expr_pparams->input_typesizes, ninputs * sizeof(int32_t));
    eval_pparams.out = pparams->out;
    eval_pparams.out_size = pparams->size;
    eval_pparams.out_typesize = eval_pparams.input_typesizes[0];
    eval_pparams.ndim = expr_pparams->e->out_dtshape->ndim;

    eval_pparams.user_data = expr_pparams;

    // Do the actual evaluation
    int ret = ((iarray_eval_fn) e->jug_expr_func)(&eval_pparams);
    switch (ret) {
        case 0:
            // 0 means success
            break;
        case 1:
            IARRAY_TRACE1(iarray.error, "Out of bounds in LLVM eval engine");
            return -2;
        default:
            IARRAY_TRACE1(iarray.error, "Error in executing LLVM eval engine");
            return -3;
    }

    return 0;
}


INA_API(ina_rc_t) iarray_expr_attach_postfilter(iarray_expression_t *e, iarray_container_t *c) {
    blosc2_context* dctx = c->catarr->sc->dctx;
    dctx->postfilter = (blosc2_postfilter_fn)postfilter_func;
    blosc2_postfilter_params *pparams = dctx->postparams;
    if (pparams == NULL) {
        // postparams not initialized yet
        pparams = calloc(1, sizeof(blosc2_postfilter_params));
    }
    iarray_expr_pparams_t* expr_pparams = calloc(1, sizeof(iarray_expr_pparams_t));
    expr_pparams->e = e;
    expr_pparams->ninputs = 1;
    expr_pparams->input_typesizes[0] = (int32_t) c->catarr->itemsize;
    pparams->user_data = (void *) expr_pparams;
    dctx->postparams = pparams;

    return INA_SUCCESS;
}


ina_rc_t iarray_eval_cleanup(iarray_expression_t *e, int64_t nitems_written)
{
    ina_mempool_reset(e->ctx->mp);
    ina_mempool_reset(e->ctx->mp_op);
    ina_mempool_reset(e->ctx->mp_tmp_out);

    int64_t nitems_in_schunk = e->nbytes / e->typesize;
    if (nitems_written != nitems_in_schunk) {
        IARRAY_TRACE1(iarray.error, "The number of items written is different from items in final container");
        return INA_ERROR(INA_ERR_NOT_COMPLETE);
    }

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_eval_iterchunk(iarray_expression_t *e, iarray_container_t *ret, int64_t *out_chunkshape)
{
    int nvars = e->nvars;
    int64_t nitems_written = 0;


    // Create and initialize an iterator per variable
    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_t *ctx = NULL;
    iarray_context_new(&cfg, &ctx);
    iarray_iter_read_block_t **iter_var = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_t));
    iarray_iter_read_block_value_t *iter_value = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_value_t));

    for (int nvar = 0; nvar < nvars; nvar++) {
        iarray_container_t *var = e->vars[nvar].c;
        IARRAY_RETURN_IF_FAILED(iarray_iter_read_block_new(ctx, &iter_var[nvar], var, out_chunkshape, &iter_value[nvar],
                                                           false));
    }

    // Write iterator for output
    iarray_iter_write_block_t *iter_out;
    iarray_iter_write_block_value_t out_value;
    IARRAY_RETURN_IF_FAILED(iarray_iter_write_block_new(ctx, &iter_out, ret, out_chunkshape, &out_value, false));

    // Create expr pparams
    iarray_expr_pparams_t expr_pparams = {0};
    expr_pparams.e = e;
    expr_pparams.ninputs = nvars;

    // Create eval pparams
    iarray_eval_pparams_t eval_pparams = {0};
    eval_pparams.ninputs = nvars;
    eval_pparams.out_typesize = (int32_t) e->out->catarr->itemsize;
    eval_pparams.ndim = e->out->dtshape->ndim;
    eval_pparams.user_data = &expr_pparams;
    for (int i = 0; i < nvars; ++i) {
        eval_pparams.input_typesizes[i] = (int32_t) e->vars[i].c->catarr->itemsize;
        expr_pparams.input_class[i] = IARRAY_EXPR_NEQ;
    }

    // Evaluate the expression for all the chunks in variables
    while (INA_SUCCEED(iarray_iter_write_block_has_next(iter_out))) {
        IARRAY_RETURN_IF_FAILED(iarray_iter_write_block_next(iter_out, NULL, 0));

        int32_t out_items = (int32_t)(iter_out->cur_block_size);

        // Decompress chunks in variables into temporaries
        for (int nvar = 0; nvar < nvars; nvar++) {
            IARRAY_RETURN_IF_FAILED(iarray_iter_read_block_next(iter_var[nvar], NULL, 0));

            eval_pparams.inputs[nvar] = iter_value[nvar].block_pointer;
            expr_pparams.inputs[nvar] = iter_value[nvar].block_pointer;
        }

        // Eval the expression for this chunk
        e->max_out_len = out_items;  // so as to prevent operating beyond the limits
        eval_pparams.out = out_value.block_pointer;
        eval_pparams.out_size = out_value.block_size * e->typesize;
        expr_pparams.out_value = out_value;

        int32_t shape[IARRAY_DIMENSION_MAX];
        int64_t start[IARRAY_DIMENSION_MAX];
        int32_t strides[IARRAY_DIMENSION_MAX];

        strides[ret->dtshape->ndim - 1] = 1;
        for (int i = ret->dtshape->ndim - 1; i >= 0; --i) {
            shape[i] = out_value.block_shape[i];
            start[i] = out_value.elem_index[i];
            if (i != ret->dtshape->ndim - 1)
                strides[i] = strides[i+1] * shape[i+1];
        }
        eval_pparams.window_shape = shape;
        eval_pparams.window_start = start;
        eval_pparams.window_strides = strides;

        int err = ((iarray_eval_fn) e->jug_expr_func)(&eval_pparams);
        if (err != 0) {
            return INA_ERROR(IARRAY_ERR_EVAL_ENGINE_FAILED);
        }
        nitems_written += out_items;
    }

    IARRAY_ITER_FINISH();
    for (int nvar = 0; nvar < nvars; nvar++) {
        iarray_iter_read_block_free(&(iter_var[nvar]));
    }
    iarray_iter_write_block_free(&iter_out);

    INA_MEM_FREE_SAFE(iter_var);
    INA_MEM_FREE_SAFE(iter_value);
    iarray_context_free(&ctx);

    IARRAY_RETURN_IF_FAILED(iarray_eval_cleanup(e, nitems_written));

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_eval_iterblosc(iarray_expression_t *e, iarray_container_t *ret, int64_t *out_chunkshape)
{
    int64_t nitems_written = 0;
    int nvars = e->nvars;

    if (ret->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        IARRAY_TRACE1(iarray.error, "ITERBLOSC eval can't be used with a plainbuffer output container");
        return INA_ERROR(IARRAY_ERR_INVALID_STORAGE);
    }

    // Setup a new cparams with a prefilter
    blosc2_prefilter_params pparams = {0};
    iarray_expr_pparams_t expr_pparams = {0};
    expr_pparams.e = e;
    expr_pparams.ninputs = nvars;
    pparams.user_data = (void *) &expr_pparams;

    // Initialize the typesize for each variable
    for (int nvar = 0; nvar < nvars; nvar++) {
        iarray_container_t *var = e->vars[nvar].c;
        expr_pparams.input_typesizes[nvar] = (int32_t) var->catarr->itemsize;
    }

    // Determine the class of each container
    iarray_container_t *out = e->out;
    for (int nvar = 0; nvar < nvars; ++nvar) {
        iarray_container_t *var = e->vars[nvar].c;
        bool eq = true;
        if (var->view) {
            eq = false;
        }else {
            for (int i = 0; i < var->dtshape->ndim; ++i) {
                if (out->storage->chunkshape[i] != var->storage->chunkshape[i]) {
                    eq = false;
                    break;
                }
                if (out->storage->blockshape[i] != var->storage->blockshape[i]) {
                    eq = false;
                    break;
                }
            }
        }
        if (eq == false) {
            expr_pparams.input_class[nvar] = IARRAY_EXPR_NEQ;
        } else {
            expr_pparams.input_class[nvar] = IARRAY_EXPR_EQ;
        }
    }
    iarray_context_t *ctx = e->ctx;

    // Need for not compatible containers
    iarray_iter_read_block_t **iter_var = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_t));
    iarray_iter_read_block_value_t *iter_value = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_value_t));
    uint8_t **external_buffers = ina_mem_alloc(nvars * sizeof(void *));
    for (int nvar = 0; nvar < nvars; nvar++) {
        if (expr_pparams.input_class[nvar] == IARRAY_EXPR_NEQ) {
            iarray_container_t *var = e->vars[nvar].c;
            IARRAY_RETURN_IF_FAILED(iarray_iter_read_block_new(ctx, &iter_var[nvar], var, out_chunkshape, &iter_value[nvar],
                                                               false));
            external_buffers[nvar] = ina_mem_alloc(ret->catarr->extchunknitems * ret->catarr->itemsize);
            iter_var[nvar]->padding = true;
        }
    }

    // Need for compatible containers
    uint8_t **var_chunks = malloc(nvars * sizeof(void*));
    bool *var_needs_free = malloc(nvars * sizeof(bool));

    // Write iterator for output
    ctx->prefilter_fn = (blosc2_prefilter_fn)prefilter_func;
    ctx->prefilter_params = &pparams;

    iarray_iter_write_block_t *iter_out;
    iarray_iter_write_block_value_t out_value;
    int32_t external_buffer_size = (int32_t) (ret->catarr->extchunknitems * ret->catarr->sc->typesize + BLOSC_MAX_OVERHEAD);
    void *external_buffer = NULL;  // to inform the iterator that we are passing an external buffer
    IARRAY_RETURN_IF_FAILED(iarray_iter_write_block_new(ctx, &iter_out, ret, out_chunkshape, &out_value, true));

    // Evaluate the expression for all the chunks in variables
    int32_t nchunk = 0;
    while (INA_SUCCEED(iarray_iter_write_block_has_next(iter_out))) {
        // The external buffer is needed *inside* the write iterator because
        // this will end as a (realloc'ed) compressed chunk of a final container
        // (we do so in order to avoid copies as much as possible)
        external_buffer = malloc(external_buffer_size);

        IARRAY_RETURN_IF_FAILED(iarray_iter_write_block_next(iter_out, external_buffer, external_buffer_size));

        int32_t out_items = (int32_t)(iter_out->cur_block_size);  // TODO: add a protection against cur_block_size > 2**31

        // Get the chunk for each variable
        for (int nvar = 0; nvar < nvars; nvar++) {
            if (expr_pparams.input_class[nvar] != IARRAY_EXPR_NEQ) {
                blosc2_schunk *schunk = e->vars[nvar].c->catarr->sc;
                int csize = blosc2_schunk_get_lazychunk(schunk, nchunk, &var_chunks[nvar], &var_needs_free[nvar]);
                if (csize < 0) {
                    IARRAY_TRACE1(iarray.error, "Error in retrieving chunk from schunk");
                    return INA_ERROR(INA_ERR_NOT_SUPPORTED);
                }
                bool memcpyed = *(var_chunks[nvar] + 2) & (uint8_t)BLOSC_MEMCPYED;
                if (memcpyed) {
                    expr_pparams.input_class[nvar] = IARRAY_EXPR_EQ_NCOMP;
                } else {
                    expr_pparams.input_class[nvar] = IARRAY_EXPR_EQ;
                }
                expr_pparams.inputs[nvar] = var_chunks[nvar];
                expr_pparams.input_csizes[nvar] = csize;
            } else {
                IARRAY_RETURN_IF_FAILED(iarray_iter_read_block_next(iter_var[nvar], NULL, 0));
                IARRAY_ERR_CATERVA(caterva_blosc_array_repart_chunk((int8_t *) external_buffers[nvar],
                                                                    ret->catarr->extchunknitems * ret->catarr->itemsize,
                                                                    iter_value[nvar].block_pointer,
                                                                    ret->catarr->chunknitems * ret->catarr->itemsize,
                                                                    ret->catarr));

                expr_pparams.inputs[nvar] = external_buffers[nvar];
            }
        }

        // Eval the expression for this chunk
        expr_pparams.out_value = out_value;  // useful for the prefilter function
        blosc2_cparams cparams = {0};
        IARRAY_RETURN_IF_FAILED(iarray_create_blosc_cparams(&cparams, ctx, ret->catarr->itemsize,
                                ret->catarr->itemsize * ret->catarr->blocknitems));

        blosc2_context *cctx = blosc2_create_cctx(cparams);  // we need it here to propagate pparams.inputs
        int csize = blosc2_compress_ctx(cctx, NULL, ret->catarr->extchunknitems * e->typesize,
                                        out_value.block_pointer,
                                        ret->catarr->extchunknitems * e->typesize + BLOSC_MAX_OVERHEAD);
        if (csize <= 0) {
            IARRAY_TRACE1(iarray.error, "Error compressing a blosc chunk");
            return INA_ERROR(IARRAY_ERR_BLOSC_FAILED);
        }
        blosc2_free_ctx(cctx);

        // Free temporary chunks
        for (int nvar = 0; nvar < e->nvars; nvar++) {
            if (var_needs_free[nvar] && expr_pparams.input_class[nvar] != IARRAY_EXPR_NEQ) {
                free(var_chunks[nvar]);
            }
        }

        iter_out->compressed_chunk_buffer = true;
        nitems_written += out_items;
        nchunk += 1;
    }

    IARRAY_ITER_FINISH();
    iarray_iter_write_block_free(&iter_out);

    // Free initialized iterators
    for (int nvar = 0; nvar < nvars; ++nvar) {
        if (expr_pparams.input_class[nvar] == IARRAY_EXPR_NEQ) {
            iarray_iter_read_block_free(&iter_var[nvar]);
            ina_mem_free(external_buffers[nvar]);
        }
    }
    ina_mem_free(external_buffers);
    free(var_chunks);
    free(var_needs_free);

    INA_MEM_FREE_SAFE(iter_var);
    INA_MEM_FREE_SAFE(iter_value);

    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_eval(iarray_expression_t *e, iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(e);
    INA_VERIFY_NOT_NULL(container);

    int flags = e->out_store_properties->urlpath ? IARRAY_CONTAINER_PERSIST : 0;
    IARRAY_RETURN_IF_FAILED(iarray_container_new(e->ctx, e->out_dtshape, e->out_store_properties, flags, container));
    e->out = *container;
    iarray_container_t *ret = *container;

    int64_t out_chunkshape[IARRAY_DIMENSION_MAX];
    if (ret->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        int32_t nelems = e->out->catarr->chunknitems;
        for (int i = ret->dtshape->ndim - 1; i >= 0; i--) {
            int32_t chunkshapei = nelems < ret->dtshape->shape[i] ? nelems : (int32_t) ret->dtshape->shape[i];
            out_chunkshape[i] = chunkshapei;
            nelems = nelems / chunkshapei;
        }
    } else {
        for (int i = 0; i < ret->dtshape->ndim; ++i) {
            out_chunkshape[i] = ret->storage->chunkshape[i];
        }
    }

    uint32_t eval_method = e->ctx->cfg->eval_method & 0x3u;

    switch (eval_method) {
        case IARRAY_EVAL_METHOD_ITERCHUNK:
            IARRAY_RETURN_IF_FAILED( iarray_eval_iterchunk(e, ret, out_chunkshape));
            break;
        case IARRAY_EVAL_METHOD_ITERBLOSC:
            IARRAY_RETURN_IF_FAILED(iarray_eval_iterblosc(e, ret, out_chunkshape));
            break;
        default:
            IARRAY_TRACE1(iarray.error, "Invalid eval method");
            return INA_ERROR(IARRAY_ERR_INVALID_EVAL_METHOD);
    }
    return INA_SUCCESS;
}


ina_rc_t iarray_shape_size(iarray_dtshape_t *dtshape, size_t *size)
{
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(size);

    *size = 1;
    for (int i = 0; i < dtshape->ndim; ++i) {
        *size *= dtshape->shape[i];
    }
    return INA_SUCCESS;
}


INA_API(ina_rc_t) iarray_expr_get_mp(iarray_expression_t *e, ina_mempool_t **mp)
{
    INA_VERIFY_NOT_NULL(e);
    INA_VERIFY_NOT_NULL(mp);
    *mp = e->ctx->mp;
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_expr_get_nthreads(iarray_expression_t *e, int *nthreads)
{
    INA_VERIFY_NOT_NULL(e);
    INA_VERIFY_NOT_NULL(nthreads);
    *nthreads = e->ctx->cfg->max_num_threads;
    return INA_SUCCESS;
}
