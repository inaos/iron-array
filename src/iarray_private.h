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
/* FIXME: do some serious benchmarking for finding the optimal values below
 * (decide this at runtime maybe?) */
#define _IARRAY_MEMPOOL_OP_CHUNKS (1024*1024)
#define _IARRAY_MEMPOOL_EVAL (1024*1024)
#define _IARRAY_MEMPOOL_EVAL_TMP (1024*1024)

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
    ina_mempool_t *mp_tmp_out;
    /* FIXME: track expressions -> list */
};

typedef struct _iarray_container_store_s {
    ina_str_t id;
} _iarray_container_store_t;

struct iarray_container_s {
    iarray_dtshape_t *dtshape;
    blosc2_cparams *cparams;
    blosc2_dparams *dparams;
    blosc2_frame *frame;
    caterva_array_t *catarr;
    _iarray_container_store_t *store;
    int transposed;
    union {
        float f;
        double d;
    } scalar_value;
};

typedef struct iarray_iter_write_s {
    iarray_context_t *ctx;
    iarray_container_t *container;
    uint64_t *i_shape;
    uint64_t *i_pshape;
    uint8_t *part;
    void *pointer;
    uint64_t *index;
    uint64_t nelem;
    uint64_t cont;
    uint64_t cont_part;
    uint64_t cont_part_elem;
    uint64_t *bshape;
    uint64_t bsize;
    uint64_t *part_index;
} iarray_iter_write_t;

typedef struct iarray_iter_write_part_s {
    iarray_context_t *ctx;
    iarray_container_t *container;
    uint8_t *part;
    void *pointer;
    uint64_t *part_shape;
    uint64_t part_size;
    uint64_t *part_index;
    uint64_t *elem_index;
    uint64_t cont;
} iarray_iter_write_part_t;

typedef struct iarray_iter_read_block_s {
    iarray_context_t *ctx;
    iarray_container_t *container;
    uint8_t *part;
    void *pointer;
    uint64_t *shape;
    uint64_t *block_shape;
    uint64_t block_size;
    uint64_t *block_index;
    uint64_t *elem_index;
    uint64_t cont;
} iarray_iter_read_block_t;

typedef struct iarray_iter_matmul_s {
    iarray_context_t *ctx;
    iarray_container_t *container1;
    iarray_container_t *container2;
    uint64_t B0;
    uint64_t B1;
    uint64_t B2;
    uint64_t M;
    uint64_t K;
    uint64_t N;
    uint64_t npart1;
    uint64_t npart2;
    uint64_t cont;
} iarray_iter_matmul_t;

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
ina_rc_t _iarray_iter_matmul_new(iarray_context_t *ctx, iarray_container_t *container1,
                                 iarray_container_t *container2, uint64_t *bshape_a,
                                 uint64_t *bshape_b, iarray_iter_matmul_t **itr);
void _iarray_iter_matmul_free(iarray_iter_matmul_t *itr);
void _iarray_iter_matmul_init(iarray_iter_matmul_t *itr);
void _iarray_iter_matmul_next(iarray_iter_matmul_t *itr);
int _iarray_iter_matmul_finished(iarray_iter_matmul_t *itr);

// Utilities
bool _iarray_file_exists(const char * filename);

ina_rc_t _iarray_get_slice_buffer(iarray_context_t *ctx,
                                  iarray_container_t *c,
                                  int64_t *start,
                                  int64_t *stop,
                                  uint64_t *pshape,
                                  void *buffer,
                                  uint64_t buflen);
#endif
