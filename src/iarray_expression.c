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

#define _IARRAY_EXPR_VAR_MAX      (128)

typedef struct _iarray_tinyexpr_var_s {
    const char *var;
    iarray_container_t *c;
} _iarray_tinyexpr_var_t;

struct iarray_expression_s {
    iarray_context_t *ctx;
    ina_str_t expr;
    size_t nchunks;
    size_t blocksize;
    size_t typesize;
    size_t chunksize;
    int nvars;
    te_expr *texpr;
    iarray_temporary_t **temp_vars;
    iarray_container_t *out;
    _iarray_tinyexpr_var_t vars[_IARRAY_EXPR_VAR_MAX];
};

INA_API(ina_rc_t) iarray_expr_new(iarray_context_t *ctx, iarray_expression_t **e)
{
    INA_VERIFY_NOT_NULL(ctx);
    INA_VERIFY_NOT_NULL(e);
    *e = ina_mem_alloc(sizeof(iarray_expression_t));
    INA_RETURN_IF_NULL(e);
    (*e)->ctx = ctx;
    (*e)->nvars = 0;
    (*e)->temp_vars = ina_mem_alloc(sizeof(iarray_temporary_t*)*_IARRAY_EXPR_VAR_MAX);
    ina_mem_set(&(*e)->vars, 0, sizeof(_iarray_tinyexpr_var_t)*_IARRAY_EXPR_VAR_MAX);
    return INA_SUCCESS;
}

INA_API(void) iarray_expr_free(iarray_context_t *ctx, iarray_expression_t **e)
{
    INA_ASSERT_NOT_NULL(ctx);
    INA_VERIFY_FREE(e);
    ina_mempool_reset(ctx->mp); // FIXME
    INA_MEM_FREE_SAFE((*e)->temp_vars);
    INA_MEM_FREE_SAFE(*e);
}

INA_API(ina_rc_t) iarray_expr_bind(iarray_expression_t *e, const char *var, iarray_container_t *val)
{
    if (val->dtshape->ndim > 2) {
        return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
    }
    e->vars[e->nvars].var = var;
    e->vars[e->nvars].c = val;
    e->nvars++;
    return INA_SUCCESS;
}

//INA_API(ina_rc_t) iarray_expr_bind_scalar_float(iarray_expression_t *e, const char *var, float val)
//{
//  iarray_container_t *c = ina_mempool_dalloc(e->mp, sizeof(iarray_container_t));
//  c->dtshape = ina_mempool_dalloc(e->mp, sizeof(iarray_dtshape_t));
//  c->dtshape->ndim = 0;
//  c->dtshape->dims = NULL;
//  c->dtshape->dtype = IARRAY_DATA_TYPE_FLOAT;
//  c->scalar_value.f = val;
//  return INA_SUCCESS;
//}

INA_API(ina_rc_t) iarray_expr_bind_scalar_double(iarray_expression_t *e, const char *var, double val)
{
    iarray_container_t *c = ina_mempool_dalloc(e->ctx->mp, sizeof(iarray_container_t));
    c->dtshape = ina_mempool_dalloc(e->ctx->mp, sizeof(iarray_dtshape_t));
    c->dtshape->ndim = 0;
    c->dtshape->dtype = IARRAY_DATA_TYPE_DOUBLE;
    c->scalar_value.d = val;
    e->vars[e->nvars].var = var;
    e->vars[e->nvars].c = c;
    e->nvars++;
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_expr_compile(iarray_expression_t *e, const char *expr)
{
    e->expr = ina_str_new_fromcstr(expr);
    te_variable *te_vars = ina_mempool_dalloc(e->ctx->mp, e->nvars * sizeof(te_variable));
    caterva_array_t *catarr = e->vars[0].c->catarr;
    blosc2_schunk *schunk = catarr->sc;
    int dim0 = 0;
    if (e->ctx->cfg->flags & IARRAY_EXPR_EVAL_BLOCK) {
        int typesize = schunk->typesize;
        int nchunks = schunk->nchunks;
        void *chunk;
        bool needs_free;
        int retcode = blosc2_schunk_get_chunk(schunk, 0, &chunk, &needs_free);
        size_t chunksize, cbytes, blocksize;
        blosc_cbuffer_sizes(chunk, &chunksize, &cbytes, &blocksize);
        if (needs_free) {
            free(chunk);
        }
        dim0 = (int)blocksize / typesize;
        e->nchunks = nchunks;
        e->chunksize = chunksize;
        e->blocksize = blocksize;
        e->typesize = typesize;
    }
    else if (e->ctx->cfg->flags & IARRAY_EXPR_EVAL_CHUNK) {
        dim0 = schunk->chunksize / schunk->typesize;
        e->nchunks = schunk->nchunks;
        e->chunksize = schunk->chunksize;
        e->typesize = schunk->typesize;
    }
    else {
        fprintf(stderr, "Flag %d is not supported\n", e->ctx->cfg->flags);
        return INA_ERR_NOT_SUPPORTED;
    }
    iarray_dtshape_t shape_var = {
        .ndim = 1,
        .shape = {dim0},
        .dtype = e->vars[0].c->dtshape->dtype,
    };
    for (int nvar = 0; nvar < e->nvars; nvar++) {
        iarray_temporary_new(e, e->vars[nvar].c, &shape_var, &e->temp_vars[nvar]);
        te_vars[nvar].name = e->vars[nvar].var;
        te_vars[nvar].address = &e->temp_vars[nvar];
        te_vars[nvar].type = TE_VARIABLE;
        te_vars[nvar].context = NULL;
    }
    int err = 0;
    e->texpr = te_compile(e, ina_str_cstr(e->expr), te_vars, e->nvars, &err);
    if (e->texpr == 0) {
        return INA_ERROR(INA_ERR_NOT_COMPILED);
    }
    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_eval(iarray_expression_t *e, iarray_container_t *ret)
{
    blosc2_schunk *schunk0 = e->vars[0].c->catarr->sc;  // get the super-chunk of the first variable
    size_t nitems_in_schunk = schunk0->nbytes / e->typesize;
    size_t nitems_in_chunk = e->chunksize / e->typesize;
    int nvars = e->nvars;
    caterva_update_shape(ret->catarr, *e->vars[0].c->shape);
    caterva_array_t out = *ret->catarr;

    if (e->ctx->cfg->flags & IARRAY_EXPR_EVAL_BLOCK) {
        int8_t *outbuf = ina_mem_alloc(e->chunksize);  // FIXME: this could benefit from using a mempool (probably not)
        size_t nitems = e->blocksize / e->typesize;
        void **var_chunks = ina_mem_alloc(nvars * sizeof(void*));
        bool *var_needs_free = ina_mem_alloc(nvars * sizeof(bool));
        // Allocate a buffer for every (compressed) chunk
        for (int nvar = 0; nvar < nvars; nvar++) {
            //var_chunks[nvar] = ina_mem_alloc(e->chunksize);  // FIXME: looks like this does not work correctly
            var_chunks[nvar] = malloc(e->chunksize);
            var_needs_free[nvar] = false;
        }
        for (size_t nchunk = 0; nchunk < e->nchunks; nchunk++) {
            size_t chunksize = (nchunk < e->nchunks - 1) ? e->chunksize : schunk0->nbytes - nchunk * e->chunksize;
            size_t nblocks_in_chunk = chunksize / e->blocksize;
            size_t corrected_blocksize = e->blocksize;
            size_t corrected_nitems = nitems;
            if (nblocks_in_chunk * e->blocksize < e->chunksize) {
                nitems_in_chunk = chunksize / e->typesize;
                nblocks_in_chunk += 1;
            }
            // Allocate a buffer for every chunk (specially useful for reading on-disk variables)
            for (int nvar = 0; nvar < nvars; nvar++) {
                blosc2_schunk *schunk = e->vars[nvar].c->catarr->sc;
                int retcode = blosc2_schunk_get_chunk(schunk, (int)nchunk, &var_chunks[nvar], &var_needs_free[nvar]);
            }
//#pragma omp parallel for schedule(dynamic)
            for (size_t nblock = 0; nblock < nblocks_in_chunk; nblock++) {
                if ((nblock + 1 == nblocks_in_chunk) && (nblock + 1) * e->blocksize > chunksize) {
                    corrected_blocksize = chunksize - nblock * e->blocksize;
                    corrected_nitems = (int)corrected_blocksize / e->typesize;
                }
                // Decompress blocks in variables into temporaries
                for (int nvar = 0; nvar < nvars; nvar++) {
                    int dsize = blosc_getitem(var_chunks[nvar], (int)(nblock * nitems), (int)corrected_nitems, e->temp_vars[nvar]->data);
                    if (dsize < 0) {
                        printf("Decompression error.  Error code: %d\n", dsize);
                        return INA_ERR_FAILED;
                    }
                }
                // Evaluate the expression for this block
                const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
                ina_mem_cpy(outbuf + nblock * e->blocksize, expr_out->data, corrected_blocksize);
            }
            blosc2_schunk_append_buffer(out.sc, outbuf, nitems_in_chunk * e->typesize);
        }
        for (int nvar = 0; nvar < nvars; nvar++) {
            if (var_needs_free[nvar]) {
                //ina_mem_free(var_chunks[nvar]);  // this raises an error (bug in the ina library?)
                free(var_chunks[nvar]);
            }
        }
        ina_mem_free(var_chunks);
        ina_mem_free(var_needs_free);
        ina_mem_free(outbuf);
    }
    else {
        // Evaluate the expression for all the chunks in variables
        for (size_t nchunk = 0; nchunk < e->nchunks; nchunk++) {
            // Decompress chunks in variables into temporaries
            for (int nvar = 0; nvar < nvars; nvar++) {
                blosc2_schunk *schunk = e->vars[nvar].c->catarr->sc;
                int dsize = blosc2_schunk_decompress_chunk(schunk, (int)nchunk, e->temp_vars[nvar]->data, e->chunksize);
                if (dsize < 0) {
                    printf("Decompression error.  Error code: %d\n", dsize);
                    return INA_ERR_FAILED;
                }
            }
            const iarray_temporary_t *expr_out = te_eval(e, e->texpr);
            // Correct the number of items in last chunk
            nitems_in_chunk = (nchunk < e->nchunks - 1) ? nitems_in_chunk : nitems_in_schunk - nchunk * nitems_in_chunk;
            blosc2_schunk_append_buffer(out.sc, expr_out->data, nitems_in_chunk * e->typesize);
        }
    }
    return INA_SUCCESS;
}

ina_rc_t iarray_shape_size(iarray_dtshape_t *dtshape, size_t *size)
{
    size_t type_size = 0;
    switch (dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            type_size = sizeof(double);
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            type_size = sizeof(float);
            break;
    }
    for (int i = 0; i < dtshape->ndim; ++i) {
        *size += dtshape->shape[i] * type_size;
    }
    return INA_SUCCESS;
}

ina_rc_t iarray_temporary_new(iarray_expression_t *expr, iarray_container_t *c, iarray_dtshape_t *dtshape,
        iarray_temporary_t **temp)
{
    *temp = ina_mempool_dalloc(expr->ctx->mp, sizeof(iarray_temporary_t));
    (*temp)->dtshape = ina_mempool_dalloc(expr->ctx->mp, sizeof(iarray_dtshape_t));
    ina_mem_cpy((*temp)->dtshape, dtshape, sizeof(iarray_dtshape_t));
    size_t size = 0;
    iarray_shape_size(dtshape, &size);
    (*temp)->size = size;
    if (c != NULL) {
        // FIXME: support float values too
        ina_mem_cpy(&(*temp)->scalar_value, &c->scalar_value, sizeof(double));
    }
    if (size > 0) {
        (*temp)->data = ina_mempool_dalloc(expr->ctx->mp, size);
    }

    return INA_SUCCESS;
}

static iarray_temporary_t* _iarray_op(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_optype_t op)
{
    bool scalar = false;
    bool scalar_vector = false;
    bool vector_vector = false;
    iarray_dtshape_t dtshape;
    ina_mem_set(&dtshape, 0, sizeof(iarray_dtshape_t));
    iarray_blas_type_t op_type = IARRAY_OPERATION_TYPE_BLAS1;
    iarray_temporary_t *scalar_tmp = NULL;
    iarray_temporary_t *scalar_lhs = NULL;
    iarray_temporary_t *out;

    if (lhs->dtshape->ndim == 0 && rhs->dtshape->ndim == 0) {   /* scalar-scalar */
        dtshape.dtype = rhs->dtshape->dtype;
        dtshape.ndim = rhs->dtshape->ndim;
        memcpy(dtshape.shape, rhs->dtshape->shape, sizeof(int) * dtshape.ndim);
        scalar = true;
    }
    else if (lhs->dtshape->ndim == 0 || rhs->dtshape->ndim == 0) {   /* scalar-vector */
        if (lhs->dtshape->ndim == 0) {
            dtshape.dtype = rhs->dtshape->dtype;
            dtshape.ndim = rhs->dtshape->ndim;
            ina_mem_cpy(dtshape.shape, rhs->dtshape->shape, sizeof(int) * dtshape.ndim);
            scalar_tmp = lhs;
            scalar_lhs = rhs;
        }
        else {
            dtshape.dtype = lhs->dtshape->dtype;
            dtshape.ndim = lhs->dtshape->ndim;
            ina_mem_cpy(dtshape.shape, lhs->dtshape->shape, sizeof(int) * dtshape.ndim);
            scalar_tmp = rhs;
            scalar_lhs = lhs;
        }
        scalar_vector = true;
    }
    else if (lhs->dtshape->ndim == 1 && rhs->dtshape->ndim == 1) { /* vector-vector */
        dtshape.dtype = lhs->dtshape->dtype;
        dtshape.ndim = lhs->dtshape->ndim;
        ina_mem_cpy(dtshape.shape, lhs->dtshape->shape, sizeof(int)*lhs->dtshape->ndim);
        vector_vector = true;
    }
    else {
        /* FIXME: matrix/vector and matrix/matrix addition */
    }

    iarray_temporary_new(expr, NULL, &dtshape, &out);

    switch (dtshape.dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
        {
            int len = (int)out->size / sizeof(double);
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
                    printf("Operation not supported yet");
                }
            }
            else if (scalar_vector) {
                double dscalar = scalar_tmp->scalar_value.d;
                double *odata = (double*)out->data;
                double *ldata = (double*)scalar_lhs->data;
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] + dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] - dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] * dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] / dscalar;
                    }
                    break;
                default:
                    printf("Operation not supported yet");
                }
            }
            else if (vector_vector) {
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        ((double*)out->data)[i] = ((double*)lhs->data)[i] + ((double*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        ((double*)out->data)[i] = ((double*)lhs->data)[i] - ((double*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        ((double*)out->data)[i] = ((double*)lhs->data)[i] * ((double*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        ((double*)out->data)[i] = ((double*)lhs->data)[i] / ((double*)rhs->data)[i];
                    }
                    break;
                default:
                    printf("Operation not supported yet");
                }
            }
            else {
                printf("DTshape combination not supported yet\n");
                return NULL;
            }
        }
        break;
        case IARRAY_DATA_TYPE_FLOAT:
        {
            int len = (int)out->size / sizeof(float);
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
                    printf("Operation not supported yet");
                }
            }
            else if (scalar_vector) {
                float dscalar = (float)scalar_tmp->scalar_value.d;
                float *odata = (float*)out->data;
                float *ldata = (float*)scalar_lhs->data;
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] + dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] - dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] * dscalar;
                    }
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        odata[i] = ldata[i] / dscalar;
                    }
                    break;
                default:
                    printf("Operation not supported yet");
                }
            }
            else if (vector_vector) {
                switch(op) {
                case IARRAY_OPERATION_TYPE_ADD:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        ((float*)out->data)[i] = ((float*)lhs->data)[i] + ((float*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_SUB:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        ((float*)out->data)[i] = ((float*)lhs->data)[i] - ((float*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_MUL:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        ((float*)out->data)[i] = ((float*)lhs->data)[i] * ((float*)rhs->data)[i];
                    }
                    break;
                case IARRAY_OPERATION_TYPE_DIVIDE:
#pragma omp parallel for
                    for (int i = 0; i < len; ++i) {
                        ((float*)out->data)[i] = ((float*)lhs->data)[i] / ((float*)rhs->data)[i];
                    }
                    break;
                default:
                    printf("Operation not supported yet");
                }
            }
            else {
                printf("DTshape combination not supported yet\n");
                return NULL;
            }
        }
        break;
    }

    return out;
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
    *mp = e->ctx->mp;
    return INA_SUCCESS;
}
