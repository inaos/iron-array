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

typedef struct iarray_temporary_s {
    iarray_dtshape_t *dtshape;
    size_t size;
    void *data;
    union {
        float f;
        double d;
    } scalar_value;
} iarray_temporary_t;

ina_rc_t iarray_temporary_new(iarray_expression_t *expr, iarray_container_t *c, iarray_dtshape_t *dtshape, iarray_temporary_t **temp);

ina_rc_t iarray_shape_size(iarray_dtshape_t *dtshape, size_t *size);

/* FIXME: since we want to keep the changes to tinyexpr as little as possible we deviate from our usual function decls */
//static iarray_temporary_t* _iarray_op(iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_optype_t op);
iarray_temporary_t* _iarray_op_add(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs);
iarray_temporary_t* _iarray_op_sub(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs);
iarray_temporary_t* _iarray_op_mul(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs);
iarray_temporary_t* _iarray_op_divide(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs);
iarray_temporary_t* _iarray_matmul(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs);

#endif