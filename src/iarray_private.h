#ifndef IARRAY_PRIVATE_H_
#define IARRAY_PRIVATE_H_

#include <stddef.h>
#include <libiarray/iarray.h>

typedef enum iarray_operation_type_e {
	IARRAY_OPERATION_TYPE_BLAS1,
	IARRAY_OPERATION_TYPE_BLAS2,
	IARRAY_OPERATION_TYPE_BLAS3
} iarray_operation_type_t;

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
iarray_temporary_t* _iarray_op_add(iarray_temporary_t *lhs, iarray_temporary_t *rhs);

#endif