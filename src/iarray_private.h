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

#ifndef _IARRAY_PRIVATE_H_
#define _IARRAY_PRIVATE_H_

#include <libiarray/iarray.h>

/* Dependencies */
#include <blosc.h>
#include <caterva.h>
#include <mkl.h>

 /* Sizes */
#define _IARRAY_SIZE_KB  (1024)
#define _IARRAY_SIZE_MB  (1024*_IARRAY_SIZE_KB)
#define _IARRAY_SIZE_GB  (1024*_IARRAY_SIZE_MB)

/* Mempools */
#define _IARRAY_MEMPOOL_OP_CHUNKS (8*1024*1024) /* FIXME: evaluate L3 during context init) */
#define _IARRAY_MEMPOOL_EVAL_SIZE (8*1024*1024)

typedef enum iarray_optype_e {
    IARRAY_OPERATION_TYPE_ADD,
    IARRAY_OPERATION_TYPE_SUB,
    IARRAY_OPERATION_TYPE_MUL,
    IARRAY_OPERATION_TYPE_DIVIDE,
    IARRAY_OPERATION_TYPE_NEGATE,
} iarray_optype_t;

typedef enum iarray_blas_type_e {
    IARRAY_OPERATION_TYPE_BLAS1,
    IARRAY_OPERATION_TYPE_BLAS2,
    IARRAY_OPERATION_TYPE_BLAS3
} iarray_blas_type_t;

struct iarray_context_s {
    iarray_config_t *cfg;
    ina_mempool_t *mp;
    ina_mempool_t *mp_op;
    /* FIXME: track expressions -> list */
};

typedef struct _iarray_container_store_s {
    ina_str_t id;
} _iarray_container_store_t;

struct iarray_container_s {
    iarray_dtshape_t *dtshape;
    blosc2_cparams *cparams;
    blosc2_dparams *dparams;
    caterva_dims_t *pshape;  // FIXME: is not this part of dtshape?
    caterva_dims_t *shape;  // FIXME: is not this part of dtshape?
    blosc2_frame *frame;
    caterva_array_t *catarr;
    _iarray_container_store_t *store;
    int transposed;
    union {
        float f;
        double d;
    } scalar_value;
};

typedef struct iarray_itr_s {
    iarray_container_t *container;
    uint8_t *part;
    void *pointer;
    uint64_t *index;
    uint64_t nelem;
    uint64_t cont;
} iarray_itr_t;

typedef struct iarray_itr_chunk_s {
    iarray_container_t *container;
    uint8_t *part;
    void *pointer;
    uint64_t part_size;
    uint64_t *index;
    uint64_t nelem;
    uint64_t cont;
} iarray_itr_chunk_t;

typedef struct iarray_itr_matmul_s {
    iarray_container_t *container1;
    iarray_container_t *container2;
    uint64_t nchunk1;
    uint64_t nchunk2;
    uint64_t cont;
} iarray_itr_matmul_t;

typedef struct iarray_variable_s {
    const char *name;
    const void *address;
    iarray_dtshape_t dtshape;
    void *context;
} iarray_variable_t;


typedef struct iarray_temporary_s {
    iarray_dtshape_t *dtshape;
    size_t size;
    void *data;
    union {
        float f;
        double d;
    } scalar_value;
} iarray_temporary_t;

typedef void(*_iarray_vml_fun_d_ab)(const MKL_INT n, const double a[], const double b[], double r[]);
typedef void(*_iarray_vml_fun_s_ab)(const MKL_INT n, const float a[], const float b[], float r[]);

typedef void(*_iarray_vml_fun_d_a)(const MKL_INT n, const double a[], double r[]);
typedef void(*_iarray_vml_fun_s_a)(const MKL_INT n, const float a[], float r[]);

ina_rc_t iarray_temporary_new(iarray_expression_t *expr, iarray_container_t *c, iarray_dtshape_t *dtshape, iarray_temporary_t **temp);

ina_rc_t iarray_shape_size(iarray_dtshape_t *dtshape, size_t *size);

/* FIXME: since we want to keep the changes to tinyexpr as little as possible we deviate from our usual function decls */
//static iarray_temporary_t* _iarray_op(iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_optype_t op);
iarray_temporary_t* _iarray_op_add(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs);
iarray_temporary_t* _iarray_op_sub(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs);
iarray_temporary_t* _iarray_op_mul(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs);
iarray_temporary_t* _iarray_op_divide(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs);


// Iterators
INA_API(ina_rc_t) iarray_itr_matmul_new(iarray_context_t *ctx, iarray_container_t *container1, iarray_container_t *container2, iarray_itr_matmul_t **itr);
INA_API(ina_rc_t) iarray_itr_matmul_free(iarray_context_t *ctx, iarray_itr_matmul_t *itr);
INA_API(ina_rc_t) iarray_itr_matmul_init(iarray_itr_matmul_t *itr);
INA_API(ina_rc_t) iarray_itr_matmul_next(iarray_itr_matmul_t *itr);
INA_API(int) iarray_itr_matmul_finished(iarray_itr_matmul_t *itr);

#endif