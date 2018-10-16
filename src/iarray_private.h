#ifndef IARRAY_PRIVATE_H_
#define IARRAY_PRIVATE_H_

#include <stddef.h>
#include <libiarray/iarray.h>

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

typedef struct iarray_variable_s {
	const char *name;
	const void *address;
	iarray_dtshape_t dtshape;
	void *context;
} iarray_variable_t;

ina_rc_t iarray_temporary_new(iarray_expression_t *expr, iarray_container_t *c, iarray_dtshape_t *dtshape, iarray_temporary_t **temp);

ina_rc_t iarray_shape_size(iarray_dtshape_t *dtshape, size_t *size);

/* FIXME: since we want to keep the changes to tinyexpr as little as possible we deviate from our usual function decls */
//static iarray_temporary_t* _iarray_op(iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_optype_t op);
iarray_temporary_t* _iarray_op_add(iarray_temporary_t *lhs, iarray_temporary_t *rhs);
iarray_temporary_t* _iarray_op_sub(iarray_temporary_t *lhs, iarray_temporary_t *rhs);
iarray_temporary_t* _iarray_op_mul(iarray_temporary_t *lhs, iarray_temporary_t *rhs);
iarray_temporary_t* _iarray_op_divide(iarray_temporary_t *lhs, iarray_temporary_t *rhs);

#endif