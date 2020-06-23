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
#include <contribs/tinyexpr/tinyexpr.h>
#include <minjugg.h>
#include <caterva_blosc.h>

#if defined(_OPENMP)
#include <omp.h>
#endif


typedef struct _iarray_tinyexpr_var_s {
    const char *var;
    iarray_container_t *c;
} _iarray_tinyexpr_var_t;

struct iarray_expression_s {
    iarray_context_t *ctx;
    ina_str_t expr;
    int32_t nchunks;
    int32_t blocksize;
    int32_t typesize;
    int32_t chunksize;
    int64_t nbytes;
    int nvars;
    int32_t max_out_len;
    te_expr *texpr;
    jug_expression_t *jug_expr;
    uint64_t jug_expr_func;
    iarray_temporary_t **temp_vars;
    iarray_dtshape_t *out_dtshape;
    iarray_storage_t *out_store_properties;
    iarray_container_t *out;
    _iarray_tinyexpr_var_t vars[IARRAY_EXPR_OPERANDS_MAX];
};

// Struct to be used as info container for dealing with the expression
typedef struct iarray_expr_pparams_s {
    bool compressed_inputs;  // whether the inputs are compressed or not (should be the first to avoid crashes, WTF??)
    int ninputs;  // number of data inputs
    uint8_t* inputs[IARRAY_EXPR_OPERANDS_MAX];  // the data inputs
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
    ina_mem_set(&(*e)->vars, 0, sizeof(_iarray_tinyexpr_var_t) * IARRAY_EXPR_OPERANDS_MAX);
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
    INA_MEM_FREE_SAFE((*e)->temp_vars);
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
    memcpy(e->out_dtshape, dtshape, sizeof(iarray_dtshape_t));

    e->out_store_properties = ina_mem_alloc(sizeof(iarray_storage_t));
    memcpy(e->out_store_properties, store, sizeof(iarray_storage_t));

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
    uint32_t eval_method = e->ctx->cfg->eval_flags & 0x3u;
    uint32_t eval_engine = (e->ctx->cfg->eval_flags & 0x38u) >> 3u;

    if (eval_method == IARRAY_EVAL_METHOD_AUTO) {
        iarray_storage_type_t backend = IARRAY_STORAGE_BLOSC;
        bool equal_pshape = true;
        bool equal_bshape = true;

        if (e->out_store_properties->backend == IARRAY_STORAGE_PLAINBUFFER) {
            backend = IARRAY_STORAGE_PLAINBUFFER;
        } else {
            for (int i = 0; i < e->nvars; ++i) {
                iarray_container_t *c = e->vars[i].c;
                if (c->storage->backend == IARRAY_STORAGE_PLAINBUFFER) {
                    backend = IARRAY_STORAGE_PLAINBUFFER;
                    break;
                }
                if (equal_pshape) {
                    for (int j = 0; j < c->dtshape->ndim; ++j) {
                        if (c->storage->chunkshape[j] != e->out_store_properties->chunkshape[j]) {
                            equal_pshape = false;
                            break;
                        }
                    }
                }
                if (equal_bshape) {
                    for (int j = 0; j < c->dtshape->ndim; ++j) {
                        if (c->storage->blockshape[j] != e->out_store_properties->blockshape[j]) {
                            equal_bshape = false;
                            break;
                        }
                    }
                }
            }
        }

        if (backend == IARRAY_STORAGE_PLAINBUFFER) {
           eval_method = IARRAY_EVAL_METHOD_ITERCHUNK;
        } else {
            if (!equal_pshape) {
                eval_method = IARRAY_EVAL_METHOD_ITERBLOSC;
            } else {
                // Add new method for equal blockshape
                eval_method = IARRAY_EVAL_METHOD_ITERBLOSC2;
            }
        }
    }

    if (eval_engine == IARRAY_EVAL_ENGINE_AUTO) {
        if (eval_method == IARRAY_EVAL_METHOD_ITERCHUNK) {
            eval_engine = IARRAY_EVAL_ENGINE_INTERPRETER;
        } else {
            eval_engine = IARRAY_EVAL_ENGINE_COMPILER;
        }
    }

    e->ctx->cfg->eval_flags = eval_method | (eval_engine << 3u);

    e->temp_vars = ina_mem_alloc(e->nvars * sizeof(iarray_temporary_t *));
    caterva_array_t *catarr = e->vars[0].c->catarr;

    e->typesize = catarr->itemsize;
    int64_t size = 1;
    for (int i = 0; i < e->vars[0].c->dtshape->ndim; ++i) {
        size *= e->vars[0].c->dtshape->shape[i];
    }

    e->nbytes = size * e->typesize;
    if (catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        // Somewhat arbitrary values follows
        e->blocksize = 1024 * e->typesize;
        e->chunksize = 16 * e->blocksize;
    }
    else {
        blosc2_schunk *schunk = catarr->sc;
        if (eval_method == IARRAY_EVAL_METHOD_ITERBLOSC2) {
            uint8_t *chunk;
            bool needs_free;
            int retcode = blosc2_schunk_get_chunk(schunk, 0, &chunk, &needs_free);
            if (retcode < 0) {
                if (chunk != NULL) {
                    free(chunk);
                }
                IARRAY_TRACE1(iarray.error, "Error getting  chunk from a blosc schunk");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
            }

            size_t chunksize, cbytes, blocksize;
            blosc_cbuffer_sizes(chunk, &chunksize, &cbytes, &blocksize);
            if (needs_free) {
                free(chunk);
            }
            e->chunksize = (int32_t) chunksize;
            e->blocksize = (int32_t) blocksize;
        }
        else if (eval_method == IARRAY_EVAL_METHOD_ITERCHUNK ||
                 eval_method == IARRAY_EVAL_METHOD_ITERBLOSC) {
            e->chunksize = schunk->chunksize;
        }
        else {
            IARRAY_TRACE1(iarray.error, "Flag is not supported in evaluator");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_NOT_SUPPORTED));
        }
    }

    e->nchunks = (int32_t)(e->nbytes / e->chunksize);
    if (e->nchunks * e->chunksize < e->nbytes) {
        e->nchunks += 1;
    }

    // Create temporaries for initial variables.
    // We don't need the temporaries to be conformant with chunkshape; only the buffer
    // size needs to the same.
    iarray_dtshape_t dtshape_var = {0};  // initialize to 0s
    dtshape_var.ndim = 1;
    int32_t temp_var_dim0 = 0;
    if (eval_method == IARRAY_EVAL_METHOD_ITERBLOSC2) {
        temp_var_dim0 = e->blocksize / e->typesize;
    } else if (eval_method == IARRAY_EVAL_METHOD_ITERCHUNK ||
               eval_method == IARRAY_EVAL_METHOD_ITERBLOSC) {
        temp_var_dim0 = e->chunksize / e->typesize;
        e->blocksize = 0;
    } else {
        IARRAY_TRACE1(iarray.error, "Flag is not supported in evaluator");
        IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_NOT_SUPPORTED));
    }
    dtshape_var.shape[0] = temp_var_dim0;
    dtshape_var.dtype = e->vars[0].c->dtshape->dtype;

    for (int nvar = 0; nvar < e->nvars; nvar++) {
        // Allocate different buffers for each thread too
        INA_FAIL_IF_ERROR(iarray_temporary_new(e, e->vars[nvar].c, &dtshape_var, &e->temp_vars[nvar]));
    }

    return INA_SUCCESS;

fail:
    INA_MEM_FREE_SAFE(e->temp_vars);
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_expr_compile_udf(
    iarray_expression_t *e,
    int llvm_bc_len,
    const char *llvm_bc,
    const char* name)
{
    INA_VERIFY_NOT_NULL(e);
    INA_VERIFY_NOT_NULL(llvm_bc);

    ina_rc_t rc = _iarray_expr_prepare(e);
    if (rc != INA_SUCCESS) {
        return rc;
    }

    INA_FAIL_IF_ERROR(
        jug_udf_compile(e->jug_expr, llvm_bc_len, llvm_bc, name, &e->jug_expr_func)
    );

    return INA_SUCCESS;

fail:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_expr_compile(iarray_expression_t *e, const char *expr)
{
    INA_VERIFY_NOT_NULL(e);
    INA_VERIFY_NOT_NULL(expr);

    e->expr = ina_str_new_fromcstr(expr);

    ina_rc_t rc = _iarray_expr_prepare(e);
    if (rc != INA_SUCCESS) {
        return rc;
    }

    te_variable *te_vars = ina_mempool_dalloc(e->ctx->mp, e->nvars * sizeof(te_variable));
    jug_te_variable *jug_vars = ina_mempool_dalloc(e->ctx->mp, e->nvars * sizeof(jug_te_variable));
    memset(jug_vars, 0, e->nvars * sizeof(jug_te_variable));
    for (int nvar = 0; nvar < e->nvars; nvar++) {
        te_vars[nvar].name = e->vars[nvar].var;
        te_vars[nvar].type = TE_VARIABLE;
        te_vars[nvar].context = NULL;
        jug_vars[nvar].name = e->vars[nvar].var;

        // Allocate different buffers for each thread too
        te_vars[nvar].address = *(e->temp_vars + nvar);
    }

    int err = 0;
    uint32_t eval_engine = (e->ctx->cfg->eval_flags & 0x38u) >> 3u;
    if (eval_engine == IARRAY_EVAL_ENGINE_INTERPRETER) {
        if (e->ctx->cfg->max_num_threads > 1) {
            // tinyexpr engine does not support multi-threading, so disable it silently
            IARRAY_TRACE1(iarray.warning, "tinyexpr does not support multithreading: fall back to use 1 thread");
            e->ctx->cfg->max_num_threads = 1;
        }
        e->texpr = te_compile(e, ina_str_cstr(e->expr), te_vars, e->nvars, &err);
        if (e->texpr == 0) {
            IARRAY_TRACE1(iarray.error, "Error compiling the expression with tinyexpr");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_EVAL_ENGINE_NOT_COMPILED));
        }
    }
    else if (eval_engine == IARRAY_EVAL_ENGINE_COMPILER) {
        ina_rc_t ina_err = jug_expression_compile(e->jug_expr, ina_str_cstr(e->expr), e->nvars,
                                              jug_vars, e->typesize, &e->jug_expr_func);
        if (INA_FAILED(ina_err)) {
            IARRAY_TRACE1(iarray.error, "Error compiling the expression with juggernaut");
            INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_EVAL_ENGINE_NOT_COMPILED));
        }
    }
    else {
        return IARRAY_ERR_INVALID_EVAL_ENGINE;
    }
    return INA_SUCCESS;

fail:
    INA_MEM_FREE_SAFE(e->temp_vars);
    return ina_err_get_rc();
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
    int32_t bsize = pparams->out_size;
    int32_t typesize = pparams->out_typesize;
    int64_t nitems = bsize / typesize;
    int64_t offset_index = pparams->out_offset / typesize;

    int8_t ndim = e->out->dtshape->ndim;

    // Element strides (in elements)
    int32_t strides[IARRAY_DIMENSION_MAX];
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0 ; --i) {
        strides[i] = strides[i+1] * e->out->catarr->blockshape[i];
    }

    // Block strides (in blocks)
    int32_t strides_block[IARRAY_DIMENSION_MAX];
    strides_block[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0 ; --i) {
        strides_block[i] = strides_block[i+1] * e->out->catarr->extchunkshape[i] / e->out->catarr->blockshape[i];
    }

    // Flattened block number
    int32_t nblock = pparams->out_offset / pparams->out_size;

    // Multidimensional block number
    int32_t nblock_ndim[IARRAY_DIMENSION_MAX];
    for (int i = ndim - 1; i >= 0; --i) {
        if (i != 0) {
            nblock_ndim[i] = (nblock % strides_block[i-1]) / strides_block[i];
        } else {
            nblock_ndim[i] = (nblock % (e->out->catarr->extchunksize / e->out->catarr->blocksize)) / strides_block[i];
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
            shape[i] = e->out->catarr->shape[i] - start_in_container[i];
        } else if (start_in_chunk[i] + e->out->catarr->blockshape[i] > e->out->catarr->chunkshape[i]) {
            shape[i] = e->out->catarr->chunkshape[i] - start_in_chunk[i];
        } else {
            shape[i] = e->out->catarr->blockshape[i];
        }
    }

    // Element strides (using bytes)
    for (int i = 0; i < ndim; ++i) {
        strides[i] *= typesize;
    }

    unsigned int eval_method = e->ctx->cfg->eval_flags & 0x7u;
    if (eval_method != IARRAY_EVAL_METHOD_ITERCHUNK) {
        // We can only set the visible shape of the output for the ITERBLOSC eval method.
        eval_pparams.window_shape = shape;
        eval_pparams.window_start = start_in_container;
        eval_pparams.window_strides = strides;
    } else {
        // eval_pparams is initialized to {0} above, but better be explicit.
        eval_pparams.window_shape = NULL;
        eval_pparams.window_start = NULL;
        eval_pparams.window_strides = NULL;
    }

    // The code below only works for the case where inputs and output have the same typesize.
    // More love is needed in the future, where we would want to allow mixed types in expressions.

    int avail_space = pparams->ttmp_nbytes;
    INA_UNUSED(avail_space);  // Fix build warning
    int used_space = 0;
    int ninputs_malloced = 0;
    for (int i = 0; i < ninputs; i++) {
        if (expr_pparams->compressed_inputs) {
            if (false && (used_space + bsize) <= avail_space) {
                // Unfortunately, we cannot re-use temporaries in threads because SVML refuse to vectorize operations
                // We have an available ttmp block that can be used for this operand
                eval_pparams.inputs[i] = pparams->ttmp + used_space;
                used_space += bsize;
            }
            else {
                // eval_pparams.inputs[i] = malloc(bsize);
                // ina_mem_alloc_aligned has the same performance than regular malloc, but allows to specify alignment
                // and alignment can be important for e.g. AVX-512 (64 bytes)
                eval_pparams.inputs[i] = ina_mem_alloc_aligned(64, bsize);
                ninputs_malloced++;
            }
            int64_t rbytes = blosc_getitem(expr_pparams->inputs[i], (int) offset_index, (int) nitems, eval_pparams.inputs[i]);
            if (rbytes != bsize) {
                fprintf(stderr, "Read from inputs failed inside pipeline\n");
                return -1;
            }
        }
        else {
            eval_pparams.inputs[i] = expr_pparams->inputs[i] + pparams->out_offset;
        }
    }

    for (int i = 0; i < ninputs; i++) {
        e->temp_vars[i]->data = eval_pparams.inputs[i];
    }

    // Eval the expression for this chunk
    int ret;
    uint32_t eval_engine = (e->ctx->cfg->eval_flags & 0x38u) >> 3u;
    switch (eval_engine) {
        case IARRAY_EVAL_ENGINE_INTERPRETER:
            e->max_out_len = pparams->out_size / pparams->out_typesize;  // so as to prevent operating beyond the limits
            const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
            memcpy(pparams->out, (uint8_t*)expr_out->data, pparams->out_size);
            break;
        case IARRAY_EVAL_ENGINE_COMPILER:
            ret = ((iarray_eval_fn)e->jug_expr_func)(&eval_pparams);
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
            break;
        default:
            IARRAY_TRACE1(iarray.error, "Invalid eval engine");
            return -4;
    }

    if (expr_pparams->compressed_inputs) {
        for (int i = (ninputs - ninputs_malloced); i < ninputs; i++) {
            // free(eval_pparams.inputs[i]);
            INA_MEM_FREE_SAFE(eval_pparams.inputs[i]);
        }
    }

    return 0;
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

INA_API(ina_rc_t) iarray_eval_iterchunk(iarray_expression_t *e, iarray_container_t *ret, int64_t *out_pshape)
{
    ina_rc_t rc;
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
        if (INA_FAILED(iarray_iter_read_block_new(ctx, &iter_var[nvar], var, out_pshape, &iter_value[nvar], false))) {
            goto fail_iterchunk;
        }
    }

    // Write iterator for output
    iarray_iter_write_block_t *iter_out;
    iarray_iter_write_block_value_t out_value;
    if (INA_FAILED(iarray_iter_write_block_new(ctx, &iter_out, ret, out_pshape, &out_value, false))) {
        goto fail_iterchunk;
    }

    // Evaluate the expression for all the chunks in variables
    while (INA_SUCCEED(iarray_iter_write_block_has_next(iter_out))) {
        if (INA_FAILED(iarray_iter_write_block_next(iter_out, NULL, 0))) {
            goto fail_iterchunk;
        }
        int32_t out_items = (int32_t)(iter_out->cur_block_size);

        // Decompress chunks in variables into temporaries
        for (int nvar = 0; nvar < nvars; nvar++) {
            if INA_FAILED(iarray_iter_read_block_next(iter_var[nvar], NULL, 0)) {
                goto fail_iterchunk;
            }
            e->temp_vars[nvar]->data = iter_value[nvar].block_pointer;
        }

        // Eval the expression for this chunk
        uint32_t eval_engine = (e->ctx->cfg->eval_flags & 0x38u) >> 3u;
        if (eval_engine == IARRAY_EVAL_ENGINE_COMPILER) {
            IARRAY_TRACE1(iarray.error, "LLVM engine cannot be used with iterchunk");
            return INA_ERROR(IARRAY_ERR_INVALID_EVAL_ENGINE);
        }
        e->max_out_len = out_items;  // so as to prevent operating beyond the limits
        const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
        memcpy((char*)out_value.block_pointer, (uint8_t*)expr_out->data, out_items * e->typesize);
        nitems_written += out_items;
        ina_mempool_reset(e->ctx->mp_tmp_out);
    }

    if (ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0)) {
        goto fail_iterchunk;
    }

    for (int nvar = 0; nvar < nvars; nvar++) {
        iarray_iter_read_block_free(&(iter_var[nvar]));
    }
    iarray_iter_write_block_free(&iter_out);
    INA_MEM_FREE_SAFE(iter_var);
    INA_MEM_FREE_SAFE(iter_value);
    iarray_context_free(&ctx);

    rc = iarray_eval_cleanup(e, nitems_written);
    return rc;

    fail_iterchunk:
    return ina_err_get_rc();
}

INA_API(ina_rc_t) iarray_eval_iterblosc(iarray_expression_t *e, iarray_container_t *ret, int64_t *out_pshape)
{
    ina_rc_t rc;
    int64_t nitems_written = 0;
    int nvars = e->nvars;

    if (ret->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        IARRAY_TRACE1(iarray.error, "ITERBLOSC eval can't be used with a plainbuffer output container");
        INA_ERROR(IARRAY_ERR_INVALID_STORAGE);
        goto fail;
    }


    blosc2_prefilter_fn prefilter = (blosc2_prefilter_fn)prefilter_func;
    blosc2_prefilter_params pparams = {0};
    iarray_expr_pparams_t expr_pparams = {0};
    expr_pparams.e = e;
    expr_pparams.ninputs = nvars;
    expr_pparams.compressed_inputs = false;
    pparams.user_data = (void *) &expr_pparams;

    // Create and initialize an iterator per variable
    iarray_context_t *ctx = e->ctx;
    ctx->prefilter_fn = prefilter;
    ctx->prefilter_params = &pparams;

    iarray_iter_read_block_t **iter_var = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_t));
    iarray_iter_read_block_value_t *iter_value = ina_mem_alloc(nvars * sizeof(iarray_iter_read_block_value_t));
    for (int nvar = 0; nvar < nvars; nvar++) {
        iarray_container_t *var = e->vars[nvar].c;
        if (INA_FAILED(iarray_iter_read_block_new(ctx, &iter_var[nvar], var, out_pshape, &iter_value[nvar], false))) {
            goto fail;
        }
        expr_pparams.input_typesizes[nvar] = var->catarr->sc->typesize;
    }

    // Write iterator for output
    iarray_iter_write_block_t *iter_out;
    iarray_iter_write_block_value_t out_value;

    int32_t external_buffer_size = ret->catarr->chunksize * ret->catarr->sc->typesize + BLOSC_MAX_OVERHEAD;
    void *external_buffer;  // for informing the iterator that we are passing an external buffer

    if (INA_FAILED(iarray_iter_write_block_new(ctx, &iter_out, ret, out_pshape, &out_value, true))) {
        goto fail;
    }
    uint8_t **external_buffers = ina_mem_alloc(nvars * sizeof(void *));
    for (int i = 0; i < nvars; ++i) {
        external_buffers[i] = ina_mem_alloc(ret->catarr->extchunksize * ret->catarr->itemsize);
    }

    // Evaluate the expression for all the chunks in variables
    while (INA_SUCCEED(iarray_iter_write_block_has_next(iter_out))) {
        // The external buffer is needed *inside* the write iterator because
        // this will end as a (realloc'ed) compressed chunk of a final container
        // (we do so in order to avoid copies as much as possible)
        external_buffer = malloc(external_buffer_size);

        if (INA_FAILED(iarray_iter_write_block_next(iter_out, external_buffer, external_buffer_size))) {
            goto fail;
        }

        // Update the external buffer with freshly allocated memory
        int64_t out_items = iter_out->cur_block_size;

        // Decompress chunks in variables into temporaries
        for (int nvar = 0; nvar < nvars; nvar++) {
            if (INA_FAILED(iarray_iter_read_block_next(iter_var[nvar], NULL, 0))) {
                goto fail;
            }
            caterva_blosc_array_repart_chunk((int8_t *) external_buffers[nvar],
                              ret->catarr->extchunksize * ret->catarr->itemsize,
                                             iter_value[nvar].block_pointer,
                                             ret->catarr->chunksize * ret->catarr->itemsize,
                                             ret->catarr);
            e->temp_vars[nvar]->data = external_buffers[nvar];
            expr_pparams.inputs[nvar] = external_buffers[nvar];
        }

        // Eval the expression for this chunk
        expr_pparams.out_value = out_value;  // useful for the prefilter function
        blosc2_cparams cparams = {0};
        iarray_create_blosc_cparams(&cparams, ctx, ret->catarr->itemsize,
                                    ret->catarr->itemsize * ret->catarr->blocksize);
        blosc2_context *cctx = blosc2_create_cctx(cparams);  // we need it here to propagate pparams.inputs
        int csize = blosc2_compress_ctx(cctx, ret->catarr->extchunksize * e->typesize,
                                        NULL, out_value.block_pointer,
                                        ret->catarr->extchunksize * e->typesize + BLOSC_MAX_OVERHEAD);
        blosc2_free_ctx(cctx);
        if (csize <= 0) {
            IARRAY_TRACE1(iarray.error, "Error compressing a blosc chunk");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
        }
        iter_out->compressed_chunk_buffer = true;
        nitems_written += out_items;
    }

    for (int i = 0; i < nvars; ++i) {
        ina_mem_free(external_buffers[i]);
    }
    ina_mem_free(external_buffers);

    if (ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0)) {
        goto fail;
    }

    for (int nvar = 0; nvar < nvars; nvar++) {
        iarray_iter_read_block_free(&iter_var[nvar]);
    }
    iarray_iter_write_block_free(&iter_out);
    INA_MEM_FREE_SAFE(iter_var);
    INA_MEM_FREE_SAFE(iter_value);

    rc = iarray_eval_cleanup(e, nitems_written);
    return rc;

    fail:
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_eval_iterblosc2(iarray_expression_t *e, iarray_container_t *ret, int64_t *out_pshape)
{
    ina_rc_t rc;
    int64_t nitems_written = 0;
    int nvars = e->nvars;

    if (ret->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        IARRAY_TRACE1(iarray.error, "ITERBLOSC2 eval can't be used with a plainbuffer output container");
        INA_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_STORAGE));
    }

    // Setup a new cparams with a prefilter
    blosc2_prefilter_params pparams = {0};
    iarray_expr_pparams_t expr_pparams = {0};
    expr_pparams.e = e;
    expr_pparams.ninputs = nvars;
    expr_pparams.compressed_inputs = true;
    pparams.user_data = (void *) &expr_pparams;

    // Initialize the typesize for each variable
    for (int nvar = 0; nvar < nvars; nvar++) {
        iarray_container_t *var = e->vars[nvar].c;
        expr_pparams.input_typesizes[nvar] = var->catarr->sc->typesize;
    }
    uint8_t **var_chunks = malloc(nvars * sizeof(void*));
    bool *var_needs_free = malloc(nvars * sizeof(bool));

    // Write iterator for output
    iarray_context_t *ctx = e->ctx;
    ctx->prefilter_fn = (blosc2_prefilter_fn)prefilter_func;
    ctx->prefilter_params = &pparams;

    iarray_iter_write_block_t *iter_out;
    iarray_iter_write_block_value_t out_value;
    int32_t external_buffer_size = ret->catarr->extchunksize * ret->catarr->sc->typesize + BLOSC_MAX_OVERHEAD;
    void *external_buffer = NULL;  // to inform the iterator that we are passing an external buffer
    INA_FAIL_IF_ERROR(iarray_iter_write_block_new(ctx, &iter_out, ret, out_pshape, &out_value, true));

    // Evaluate the expression for all the chunks in variables
    int32_t nchunk = 0;
    while (INA_SUCCEED(iarray_iter_write_block_has_next(iter_out))) {
        // The external buffer is needed *inside* the write iterator because
        // this will end as a (realloc'ed) compressed chunk of a final container
        // (we do so in order to avoid copies as much as possible)
        external_buffer = malloc(external_buffer_size);

        INA_FAIL_IF_ERROR(INA_FAILED(iarray_iter_write_block_next(iter_out, external_buffer, external_buffer_size)));

        int32_t out_items = (int32_t)(iter_out->cur_block_size);  // TODO: add a protection against cur_block_size > 2**31

        // Get the uncompressed chunks for each variable
        for (int nvar = 0; nvar < nvars; nvar++) {
            blosc2_schunk *schunk = e->vars[nvar].c->catarr->sc;
            int csize = blosc2_schunk_get_chunk(schunk, nchunk, &var_chunks[nvar], &var_needs_free[nvar]);
            if (csize < 0) {
                IARRAY_TRACE1(iarray.error, "Error in retrieving chunk from schunk");
                IARRAY_FAIL_IF_ERROR(INA_ERROR(INA_ERR_NOT_SUPPORTED));
            }
            expr_pparams.inputs[nvar] = var_chunks[nvar];
        }

        // Eval the expression for this chunk
        expr_pparams.out_value = out_value;  // useful for the prefilter function
        blosc2_cparams cparams = {0};
        iarray_create_blosc_cparams(&cparams, ctx, ret->catarr->itemsize,
                                    ret->catarr->itemsize * ret->catarr->blocksize);
        blosc2_context *cctx = blosc2_create_cctx(cparams);  // we need it here to propagate pparams.inputs
        int csize = blosc2_compress_ctx(cctx, ret->catarr->extchunksize * e->typesize,
                                        NULL, out_value.block_pointer,
                                        ret->catarr->extchunksize * e->typesize + BLOSC_MAX_OVERHEAD);
        if (csize <= 0) {
            IARRAY_TRACE1(iarray.error, "Error compressing a blosc chunk");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_BLOSC_FAILED));
        }
        blosc2_free_ctx(cctx);
        for (int nvar = 0; nvar < e->nvars; nvar++) {
            if (var_needs_free[nvar]) {
                free(var_chunks[nvar]);
            }
        }

        iter_out->compressed_chunk_buffer = true;
        nitems_written += out_items;
        nchunk += 1;
    }

    if (ina_err_get_rc() != INA_RC_PACK(IARRAY_ERR_END_ITER, 0)) {
        goto fail;
    }

    iarray_iter_write_block_free(&iter_out);
    free(var_chunks);
    free(var_needs_free);

    rc = iarray_eval_cleanup(e, nitems_written);
    return rc;

    fail:
    return ina_err_get_rc();
}


INA_API(ina_rc_t) iarray_eval(iarray_expression_t *e, iarray_container_t **container)
{
    INA_VERIFY_NOT_NULL(e);
    INA_VERIFY_NOT_NULL(container);

    int flags = e->out_store_properties->filename ? IARRAY_CONTAINER_PERSIST : 0;
    iarray_container_new(e->ctx, e->out_dtshape, e->out_store_properties, flags, container);
    e->out = *container;
    iarray_container_t *ret = *container;

    int64_t out_pshape[IARRAY_DIMENSION_MAX];
    if (ret->catarr->storage == CATERVA_STORAGE_PLAINBUFFER) {
        // Compute a decent chunkshape for a plainbuffer output
        int32_t nelems = e->chunksize / e->typesize;
        for (int i = ret->dtshape->ndim - 1; i >= 0; i--) {
            int32_t pshapei = nelems < ret->dtshape->shape[i] ? nelems : (int32_t) ret->dtshape->shape[i];
            out_pshape[i] = pshapei;
            nelems = nelems / pshapei;
        }
    } else {
        for (int i = 0; i < ret->dtshape->ndim; ++i) {
            out_pshape[i] = ret->storage->chunkshape[i];
        }
    }

    uint32_t eval_method = e->ctx->cfg->eval_flags & 0x3u;

    switch (eval_method) {
        case IARRAY_EVAL_METHOD_ITERCHUNK:
            return iarray_eval_iterchunk(e, ret, out_pshape);
        case IARRAY_EVAL_METHOD_ITERBLOSC:
            return iarray_eval_iterblosc(e, ret, out_pshape);
        case IARRAY_EVAL_METHOD_ITERBLOSC2:
            return iarray_eval_iterblosc2(e, ret, out_pshape);
        default:
            IARRAY_TRACE1(iarray.error, "Invalid eval method");
            return INA_ERROR(IARRAY_ERR_INVALID_EVAL_METHOD);
    }
}


ina_rc_t iarray_shape_size(iarray_dtshape_t *dtshape, size_t *size)
{
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(size);

    size_t type_size = 0;
    switch (dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            type_size = sizeof(double);
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            type_size = sizeof(float);
            break;
        default:
            IARRAY_TRACE1(iarray.error, "The data type is invalid");
            IARRAY_FAIL_IF_ERROR(INA_ERROR(IARRAY_ERR_INVALID_DTYPE));
    }
    for (int i = 0; i < dtshape->ndim; ++i) {
        *size += dtshape->shape[i] * type_size;
    }
    return INA_SUCCESS;
    fail:
    return ina_err_get_rc();
}

ina_rc_t iarray_temporary_new(iarray_expression_t *expr, iarray_container_t *c, iarray_dtshape_t *dtshape,
        iarray_temporary_t **temp)
{
    INA_VERIFY_NOT_NULL(expr);
    INA_VERIFY_NOT_NULL(dtshape);
    INA_VERIFY_NOT_NULL(temp);

    // When c == NULL means a temporary for output, which should go to its own memory pool for being
    // able to reset it during each block/chunk evaluation
    ina_mempool_t *mempool = (c != NULL) ? expr->ctx->mp : expr->ctx->mp_tmp_out;
    *temp = ina_mempool_dalloc(mempool, sizeof(iarray_temporary_t));
    (*temp)->dtshape = ina_mempool_dalloc(mempool, sizeof(iarray_dtshape_t));
    ina_mem_cpy((*temp)->dtshape, dtshape, sizeof(iarray_dtshape_t));
    size_t typesize = dtshape->dtype == IARRAY_DATA_TYPE_DOUBLE ? 8 : 4;
    size_t size = expr->max_out_len * typesize;
    (*temp)->size = size;
    if (c != NULL) {
        // FIXME: support float values too
        ina_mem_cpy(&(*temp)->scalar_value, &c->scalar_value, sizeof(double));
    }
    if (size > 0) {
        (*temp)->data = ina_mempool_dalloc(mempool, size);
    }

    return INA_SUCCESS;
}

iarray_temporary_t* _iarray_func(iarray_expression_t *expr, iarray_temporary_t *operand1,
                                 iarray_temporary_t *operand2, iarray_functype_t func)
{
    if (expr == NULL) {
        goto fail;
    }
    if (operand1 == NULL) {
        goto fail;
    }
    if (operand2 != NULL && (
        (operand1->dtshape->ndim != operand2->dtshape->ndim) ||
        (operand1->size != operand2->size))) {
        printf("The 2 operands do not match dims or sizes");
        goto fail;
    }

    iarray_dtshape_t dtshape = {0};  // initialize to 0s
    iarray_temporary_t *out;
    bool scalar = true;
    if (operand1->dtshape->ndim > 0) {
        scalar = false;
        dtshape.dtype = operand1->dtshape->dtype;
        dtshape.ndim = operand1->dtshape->ndim;
        memcpy(dtshape.shape, operand1->dtshape->shape, sizeof(int64_t) * dtshape.ndim);
    }

    // Creating the temporary means interacting with the INA memory allocator, which is not thread-safe.
    // We should investigate on how to overcome this syncronization point (if possible at all).
    ina_rc_t err;

    err = iarray_temporary_new(expr, NULL, &dtshape, &out);
    IARRAY_FAIL_IF_ERROR(err);

    switch (dtshape.dtype) {
        case IARRAY_DATA_TYPE_DOUBLE: {
            double *operand1_pointer;
            double *operand2_pointer = NULL;
            double *out_pointer;
            int32_t len;
            if (scalar) {
                len = 1;
                operand1_pointer = &operand1->scalar_value.d;
                out_pointer = &out->scalar_value.d;
            }
            else {
                len = expr->max_out_len == 0 ? (int32_t)(out->size / sizeof(double)) : expr->max_out_len;
                operand1_pointer = operand1->data;
                if (operand2 != NULL) operand2_pointer = operand2->data;
                out_pointer = out->data;
            }
            switch (func) {
                case IARRAY_FUNC_ABS:
                    vdAbs(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_ACOS:
                    vdAcos(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_ASIN:
                    vdAsin(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_ATAN:
                    vdAtan(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_ATAN2:
                    vdAtan2(len, operand1_pointer, operand2_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_CEIL:
                    vdCeil(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_COS:
                    vdCos(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_COSH:
                    vdCosh(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_EXP:
                    vdExp(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_FLOOR:
                    vdFloor(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_LN:
                    vdLn(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_LOG10:
                    vdLog10(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_NEGATE:
                    for (int i = 0; i < len; i++) {
                        out_pointer[i] = -operand1_pointer[i];
                    }
                    break;
                case IARRAY_FUNC_POW:
                    vdPow(len, operand1_pointer, operand2_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_SIN:
                    vdSin(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_SINH:
                    vdSinh(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_SQRT:
                    vdSqrt(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_TAN:
                    vdTan(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_TANH:
                    vdTanh(len, operand1_pointer, out_pointer);
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Operation not supported yet");
                    goto fail;
            }
        }
        break;
        case IARRAY_DATA_TYPE_FLOAT: {
            int32_t len;
            float *operand1_pointer;
            float *operand2_pointer = NULL;
            float *out_pointer;
            if (scalar) {
                len = 1;
                operand1_pointer = &operand1->scalar_value.f;
                out_pointer = &out->scalar_value.f;
            }
            else {
                len = expr->max_out_len == 0 ? (int32_t)(out->size / sizeof(float)) : expr->max_out_len;
                operand1_pointer = operand1->data;
                if (operand2 != NULL) operand2_pointer = operand2->data;
                out_pointer = out->data;
            }
            switch (func) {
                case IARRAY_FUNC_ABS:
                    vsAbs(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_ACOS:
                    vsAcos(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_ASIN:
                    vsAsin(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_ATAN:
                    vsAtan(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_ATAN2:
                    vsAtan2(len, operand1_pointer, operand2_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_CEIL:
                    vsCeil(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_COS:
                    vsCos(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_COSH:
                    vsCosh(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_EXP:
                    vsExp(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_FLOOR:
                    vsFloor(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_LN:
                    vsLn(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_LOG10:
                    vsLog10(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_NEGATE:
                    for (int i = 0; i < len; i++) {
                        out_pointer[i] = -operand1_pointer[i];
                    }
                    break;
                case IARRAY_FUNC_POW:
                    vsPow(len, operand1_pointer, operand2_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_SIN:
                    vsSin(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_SINH:
                    vsSinh(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_SQRT:
                    vsSqrt(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_TAN:
                    vsTan(len, operand1_pointer, out_pointer);
                    break;
                case IARRAY_FUNC_TANH:
                    vsTanh(len, operand1_pointer, out_pointer);
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Operation not supported yet");
                    goto fail;
            }
        }
        break;
        default:
            IARRAY_TRACE1(iarray.error, "Operation not supported yet");
            goto fail;
    }

    return out;

    fail:
    // TODO: Free temporary
    return NULL;
}

static iarray_temporary_t* _iarray_op(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_optype_t op)
{
    if (expr == NULL) {
        goto fail;
    }
    if (lhs == NULL) {
        goto fail;
    }

    if (rhs == NULL) {
        goto fail;
    }

    bool scalar = false;
    bool scalar_vector = false;
    bool vector_vector = false;

    iarray_dtshape_t dtshape = {0};  // initialize to 0s
    iarray_temporary_t *scalar_tmp = NULL;
    iarray_temporary_t *scalar_lhs = NULL;
    iarray_temporary_t *out;

    if (lhs->dtshape->ndim == 0 && rhs->dtshape->ndim == 0) {   /* scalar-scalar */
        dtshape.dtype = rhs->dtshape->dtype;
        dtshape.ndim = rhs->dtshape->ndim;
        memcpy(dtshape.shape, rhs->dtshape->shape, sizeof(int64_t) * dtshape.ndim);
        scalar = true;
    }
    else if (lhs->dtshape->ndim == 0 || rhs->dtshape->ndim == 0) {   /* scalar-vector */
        if (lhs->dtshape->ndim == 0) {
            dtshape.dtype = rhs->dtshape->dtype;
            dtshape.ndim = rhs->dtshape->ndim;
            ina_mem_cpy(dtshape.shape, rhs->dtshape->shape, sizeof(int64_t) * dtshape.ndim);
            scalar_tmp = lhs;
            scalar_lhs = rhs;
        }
        else {
            dtshape.dtype = lhs->dtshape->dtype;
            dtshape.ndim = lhs->dtshape->ndim;
            ina_mem_cpy(dtshape.shape, lhs->dtshape->shape, sizeof(int64_t) * dtshape.ndim);
            scalar_tmp = rhs;
            scalar_lhs = lhs;
        }
        scalar_vector = true;
    }
    else if (lhs->dtshape->ndim == 1 && rhs->dtshape->ndim == 1) { /* vector-vector */
        dtshape.dtype = lhs->dtshape->dtype;
        dtshape.ndim = lhs->dtshape->ndim;
        ina_mem_cpy(dtshape.shape, lhs->dtshape->shape, sizeof(int64_t) * lhs->dtshape->ndim);
        vector_vector = true;
    }
    else {
        /* FIXME: matrix/vector and matrix/matrix addition */
    }

    // Creating the temporary means interacting with the INA memory allocator, which is not thread-safe.
    // We should investigate on how to overcome this syncronization point (if possible at all).

    ina_rc_t err;
#if defined(_OPENMP)
#pragma omp critical
#endif

    err = iarray_temporary_new(expr, NULL, &dtshape, &out);
    IARRAY_FAIL_IF_ERROR(err);

    switch (dtshape.dtype) {
        case IARRAY_DATA_TYPE_DOUBLE: {
            int32_t len = expr->max_out_len == 0 ? (int32_t)(out->size / sizeof(double)) : expr->max_out_len;
            if (scalar) {
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
                    out->scalar_value.d = lhs->scalar_value.d + rhs->scalar_value.d;
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
                    out->scalar_value.d = lhs->scalar_value.d - rhs->scalar_value.d;
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
                    out->scalar_value.d = lhs->scalar_value.d * rhs->scalar_value.d;
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
                    out->scalar_value.d = lhs->scalar_value.d / rhs->scalar_value.d;
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Operation not supported yet");
                    goto fail;
                }
            }
            else if (scalar_vector) {
                double dscalar = scalar_tmp->scalar_value.d;
                double *odata = (double*)out->data;
                double *ldata = (double*)scalar_lhs->data;
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] + dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] - dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] * dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] / dscalar;
                    }
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Operation not supported yet");
                    goto fail;
                }
            }
            else if (vector_vector) {
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
                    for (int i = 0; i < len; ++i) {
                        ((double*)out->data)[i] = ((double*)lhs->data)[i] + ((double*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
                    for (int i = 0; i < len; ++i) {
                        ((double*)out->data)[i] = ((double*)lhs->data)[i] - ((double*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
                    for (int i = 0; i < len; ++i) {
                        ((double*)out->data)[i] = ((double*)lhs->data)[i] * ((double*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
                    for (int i = 0; i < len; ++i) {
                        ((double*)out->data)[i] = ((double*)lhs->data)[i] / ((double*)rhs->data)[i];
                    }
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Operation not supported yet");
                    goto fail;
                }
            }
            else {
                IARRAY_TRACE1(iarray.error, "Dtshape combination not supported yet\n");
                goto fail;
            }
        }
        break;
        case IARRAY_DATA_TYPE_FLOAT: {
            int32_t len = expr->max_out_len == 0 ? (int32_t)(out->size / sizeof(float)) : expr->max_out_len;
            if (scalar) {
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
                    out->scalar_value.f = lhs->scalar_value.f + rhs->scalar_value.f;
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
                    out->scalar_value.f = lhs->scalar_value.f - rhs->scalar_value.f;
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
                    out->scalar_value.f = lhs->scalar_value.f * rhs->scalar_value.f;
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
                    out->scalar_value.f = lhs->scalar_value.f / rhs->scalar_value.f;
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Operation not supported yet");
                    goto fail;
                }
            }
            else if (scalar_vector) {
                float dscalar = (float)scalar_tmp->scalar_value.d;
                float *odata = (float*)out->data;
                float *ldata = (float*)scalar_lhs->data;
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] + dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] - dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] * dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] / dscalar;
                    }
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Operation not supported yet");
                    goto fail;
                }
            }
            else if (vector_vector) {
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
                    for (int i = 0; i < len; ++i) {
                        ((float*)out->data)[i] = ((float*)lhs->data)[i] + ((float*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
                    for (int i = 0; i < len; ++i) {
                        ((float*)out->data)[i] = ((float*)lhs->data)[i] - ((float*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
                    for (int i = 0; i < len; ++i) {
                        ((float*)out->data)[i] = ((float*)lhs->data)[i] * ((float*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
                    for (int i = 0; i < len; ++i) {
                        ((float*)out->data)[i] = ((float*)lhs->data)[i] / ((float*)rhs->data)[i];
                    }
                    break;
                default:
                    IARRAY_TRACE1(iarray.error, "Operation not supported yet");
                    goto fail;
                }
            }
            else {
                IARRAY_TRACE1(iarray.error, "Dtshape combination not supported yet\n");
                goto fail;
            }
        }
        break;
        default:  // switch (dtshape.dtype)
            IARRAY_TRACE1(iarray.error, "Data type not supported yet\n");
            goto fail;
    }

    return out;

    fail:
        // TODO: Free temporary
        return NULL;
}

iarray_temporary_t* _iarray_op_add(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs)
{
    return _iarray_op(expr, lhs, rhs, IARRAY_OPERATION_TYPE_ADD);
}

iarray_temporary_t* _iarray_op_sub(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs)
{
    return _iarray_op(expr, lhs, rhs, IARRAY_OPERATION_TYPE_SUB);
}

iarray_temporary_t* _iarray_op_mul(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs)
{
    return _iarray_op(expr, lhs, rhs, IARRAY_OPERATION_TYPE_MUL);
}

iarray_temporary_t* _iarray_op_divide(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs)
{
    return _iarray_op(expr, lhs, rhs, IARRAY_OPERATION_TYPE_DIVIDE);
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
