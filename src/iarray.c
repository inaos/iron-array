//
// Created by Francesc Alted on 11/09/2018.
//

#include <stdio.h>
#include <stdbool.h>
#include "iarray.h"
#include "iarray_private.h"
#include "tinyexpr.h"

struct iarray_context_s {
	iarray_config_t *cfg;
	/* FIXME: track expressions -> list */
};

struct iarray_expression_s {
	ina_mempool_t *mp;
};

struct iarray_container_s {
	iarray_dtshape_t *dtshape;
	union {
		float f;
		double d;
	} scalar_value;
};

static ina_rc_t _iarray_container_new(iarray_context_t *ctx, iarray_dtshape_t *shape, iarray_data_type_t dtype, iarray_container_t **c)
{
	*c = (iarray_container_t*)ina_mem_alloc(sizeof(iarray_container_t));
	INA_RETURN_IF_NULL(c);
	(*c)->dtshape = (iarray_dtshape_t*)ina_mem_alloc(sizeof(iarray_dtshape_t));
	ina_mem_cpy((*c)->dtshape, shape, sizeof(iarray_dtshape_t));
	/* FIXME: blosc init container */
	return INA_SUCCESS;
}

static ina_rc_t _iarray_container_fill_float(iarray_container_t *c, float value)
{
	/* FIXME: blosc set container */
	return INA_SUCCESS;
}

static ina_rc_t _iarray_container_fill_double(iarray_container_t *c, double value)
{
	/* FIXME: blosc set container */
	return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_ctx_new(iarray_config_t *cfg, iarray_context_t **ctx)
{
	INA_VERIFY_NOT_NULL(ctx);
	*ctx = ina_mem_alloc(sizeof(iarray_context_t));
	INA_RETURN_IF_NULL(ctx);
	(*ctx)->cfg = ina_mem_alloc(sizeof(iarray_config_t));
	ina_mem_cpy((*ctx)->cfg, cfg, sizeof(iarray_config_t));
	return INA_SUCCESS;
}

INA_API(void) iarray_ctx_free(iarray_context_t **ctx)
{
	INA_FREE_CHECK(ctx);
	INA_MEM_FREE_SAFE((*ctx)->cfg);
	INA_MEM_FREE_SAFE(ctx);
}

INA_API(ina_rc_t) iarray_arange(iarray_context_t *ctx, iarray_dtshape_t *dtshape, int start, int stop, int step, iarray_data_type_t dtype, iarray_container_t **container)
{
	INA_VERIFY_NOT_NULL(ctx);
	INA_VERIFY_NOT_NULL(dtshape);
	INA_VERIFY_NOT_NULL(container);

	_iarray_container_new(ctx, dtshape, dtype, container);
	/* implement arange */

	return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_zeros(iarray_context_t *ctx, iarray_dtshape_t *dtshape, iarray_data_type_t dtype, iarray_container_t **container)
{
	INA_VERIFY_NOT_NULL(ctx);
	INA_VERIFY_NOT_NULL(dtshape);
	INA_VERIFY_NOT_NULL(container);

	_iarray_container_new(ctx, dtshape, dtype, container);

	switch (dtype) {
		case IARRAY_DATA_TYPE_DOUBLE:
			_iarray_container_fill_double(*container, 0.0);
			break;
		case IARRAY_DATA_TYPE_FLOAT:
			_iarray_container_fill_float(*container, 0.0f);
			break;
	}

	return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_ones(iarray_context_t *ctx, iarray_dtshape_t *dtshape, iarray_data_type_t dtype, iarray_container_t **container)
{
	INA_VERIFY_NOT_NULL(ctx);
	INA_VERIFY_NOT_NULL(dtshape);
	INA_VERIFY_NOT_NULL(container);

	_iarray_container_new(ctx, dtshape, dtype, container);

	switch (dtype) {
	case IARRAY_DATA_TYPE_DOUBLE:
		_iarray_container_fill_double(*container, 1.0);
		break;
	case IARRAY_DATA_TYPE_FLOAT:
		_iarray_container_fill_float(*container, 1.0f);
		break;
	}

	return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_fill_float(iarray_context_t *ctx, iarray_dtshape_t *dtshape, float value, iarray_container_t **container)
{
	INA_VERIFY_NOT_NULL(ctx);
	INA_VERIFY_NOT_NULL(dtshape);
	INA_VERIFY_NOT_NULL(container);

	_iarray_container_new(ctx, dtshape, IARRAY_DATA_TYPE_FLOAT, container);

	_iarray_container_fill_float(*container, value);

	return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_fill_double(iarray_context_t *ctx, iarray_dtshape_t *dtshape, double value, iarray_container_t **container)
{
	INA_VERIFY_NOT_NULL(ctx);
	INA_VERIFY_NOT_NULL(dtshape);
	INA_VERIFY_NOT_NULL(container);

	_iarray_container_new(ctx, dtshape, IARRAY_DATA_TYPE_DOUBLE, container);

	_iarray_container_fill_double(*container, value);

	return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_rand(iarray_context_t *ctx, iarray_dtshape_t *dtshape, iarray_rng_t rng, iarray_data_type_t dtype, iarray_container_t **container)
{
	INA_VERIFY_NOT_NULL(ctx);
	INA_VERIFY_NOT_NULL(dtshape);
	INA_VERIFY_NOT_NULL(container);

	return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_expr_new(iarray_context_t *ctx, iarray_expression_t **e)
{
	INA_VERIFY_NOT_NULL(ctx);
	INA_VERIFY_NOT_NULL(e);
	*e = ina_mem_alloc(sizeof(iarray_expression_t));
	INA_RETURN_IF_NULL(e);
	return INA_SUCCESS;
}

INA_API(void) iarray_expr_free(iarray_context_t *ctx, iarray_expression_t **e)
{
	INA_VERIFY_NOT_NULL(ctx);
	INA_FREE_CHECK(e);
	INA_MEM_FREE_SAFE(e);
}

INA_API(ina_rc_t) iarray_expr_bind(iarray_expression_t *e, const char *var, iarray_container_t *val)
{
	if (val->dtshape->ndim > 2) {
		/* FIXME: raise error */
		return 1;
	}
	return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_expr_bind_scalar_float(iarray_expression_t *e, const char *var, float val)
{
	iarray_container_t *c = ina_mempool_dalloc(e->mp, sizeof(iarray_container_t));
	c->dtshape = ina_mempool_dalloc(e->mp, sizeof(iarray_dtshape_t));
	c->dtshape->ndim = 0;
	c->dtshape->dims = NULL;
	c->dtshape->dtype = IARRAY_DATA_TYPE_FLOAT;
	c->scalar_value.f = val;
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
		*size += dtshape->dims[i] * type_size;
	}
	return INA_SUCCESS;
}

ina_rc_t iarray_temporary_new(iarray_expression_t *expr, iarray_container_t *c, iarray_dtshape_t *dtshape, iarray_temporary_t **temp)
{
	*temp = ina_mempool_dalloc(expr->mp, sizeof(iarray_temporary_t));
	(*temp)->dtshape = ina_mempool_dalloc(expr->mp, sizeof(iarray_dtshape_t));
	memcpy((*temp)->dtshape, dtshape, sizeof(iarray_dtshape_t));
	size_t size = 0;
	iarray_shape_size(dtshape, &size);
	(*temp)->size = size;
	if (c != NULL) {
		memcpy(&(*temp)->scalar_value, &c->scalar_value, sizeof(double));
	}
	if (size > 0) {
		(*temp)->data = ina_mempool_dalloc(expr->mp, size);
	}

	return INA_SUCCESS;
}

iarray_temporary_t* _iarray_op_add(iarray_temporary_t *lhs, iarray_temporary_t *rhs)
{
	bool scalar = false;
	bool scalar_vector = false;
	iarray_dtshape_t dtshape;
	memset(&dtshape, 0, sizeof(iarray_dtshape_t));
	iarray_operation_type_t op_type = IARRAY_OPERATION_TYPE_BLAS1;
	iarray_temporary_t *scalar_tmp = NULL;
	iarray_temporary_t *scalar_lhs = NULL;
	iarray_temporary_t *out;
	iarray_expression_t expr; /* temp hack */
	memset(&expr, 0, sizeof(iarray_expression_t));

	if (lhs->dtshape->ndim == 0 && rhs->dtshape->ndim == 0) {   /* scalar-scalar */
		dtshape.dtype = rhs->dtshape->dtype;
		dtshape.ndim = rhs->dtshape->ndim;
		memcpy(dtshape.dims, rhs->dtshape->dims, sizeof(int) * dtshape.ndim);
		scalar = true;
	}
	else if (lhs->dtshape->ndim == 0 || rhs->dtshape->ndim == 0) {   /* scalar-vector */
		if (lhs->dtshape->ndim == 0) {
			dtshape.dtype = rhs->dtshape->dtype;
			dtshape.ndim = rhs->dtshape->ndim;
			memcpy(dtshape.dims, rhs->dtshape->dims, sizeof(int) * dtshape.ndim);
			scalar_tmp = lhs;
			scalar_lhs = rhs;
		}
		else {
			dtshape.dtype = lhs->dtshape->dtype;
			dtshape.ndim = lhs->dtshape->ndim;
			memcpy(dtshape.dims, lhs->dtshape->dims, sizeof(int) * dtshape.ndim);
			scalar_tmp = rhs;
			scalar_lhs = lhs;
		}
		scalar_vector = true;
	}
	else if (lhs->dtshape->ndim == 1 && rhs->dtshape->ndim == 1) { /* vector-vector */
		dtshape.dtype = lhs->dtshape->dtype;
		dtshape.ndim = lhs->dtshape->ndim;
		memcpy(dtshape.dims, lhs->dtshape->dims, sizeof(int)*lhs->dtshape->ndim);
	}
	else {
		/* FIXME: matrix/vector and matrix/matrix addition */
	}

	iarray_temporary_new(&expr, NULL, &dtshape, &out);

	switch (dtshape.dtype) {
		case IARRAY_DATA_TYPE_DOUBLE: 
		{
			int len = (int)out->size / sizeof(double);
			if (scalar) {
					out->scalar_value.d = lhs->scalar_value.d + rhs->scalar_value.d;
			}
			else if (scalar_vector) {
				for (int i = 0; i < len; ++i) {
					((double*)out->data)[i] = ((double*)scalar_lhs->data)[i] + scalar_tmp->scalar_value.d;
				}
			}
			else {
				for (int i = 0; i < len; ++i) {
					((double*)out->data)[i] = ((double*)lhs->data)[i] + ((double*)rhs->data)[i];
				}
			}
		}
		break;
		case IARRAY_DATA_TYPE_FLOAT:
		{
			int len = (int)out->size / sizeof(float);
			if (scalar) {
				out->scalar_value.f = lhs->scalar_value.f + rhs->scalar_value.f;
			}
			else if (scalar_vector) {
				for (int i = 0; i < len; ++i) {
					((float*)out->data)[i] = ((float*)scalar_lhs->data)[i] + scalar_tmp->scalar_value.f;
				}
			}
			else {
				for (int i = 0; i < len; ++i) {
					((float*)out->data)[i] = ((float*)lhs->data)[i] + ((float*)rhs->data)[i];
				}
			}
		}
		break;
	}

	return out;
}


int main(int argc, char **argv) {
	iarray_temporary_t *x1, *y1;
	iarray_expression_t iexpr;
	memset(&iexpr, 0, sizeof(iarray_expression_t));
	iarray_dtshape_t shape = {
			.ndim = 0,
			.dims = NULL,
			.dtype = IARRAY_DATA_TYPE_DOUBLE,
	};
	iarray_temporary_new(&iexpr, NULL, &shape, &x1);
	iarray_temporary_new(&iexpr, NULL, &shape, &y1);

	double var1 = 5;
	x1->scalar_value.d = var1;
	double var2 = 3.;
	y1->scalar_value.d = var2;

	/* Store variable names and pointers. */
	te_variable vars[] = {{"x", &x1}, {"y", &y1}};

	int err;
	/* Compile the expression with variables. */
	te_expr *expr = te_compile("x + y", vars, 2, &err);

	if (expr) {
		x1->scalar_value.d = 3;
		y1->scalar_value.d = 4;
		const iarray_temporary_t *h1 = te_eval(expr); /* Returns 5. */
		//printf("h1: %f\n", h1->size);

		x1->scalar_value.d = 5;
		y1->scalar_value.d = 12;
		const iarray_temporary_t *h2 = te_eval(expr); /* Returns 13. */
		//printf("h2: %f\n", h2);

		te_free(expr);
	} else {
		printf("Parse error at %d\n", err);
	}

	return 0;
}