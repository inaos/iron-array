//
// Created by Francesc Alted on 11/09/2018.
//

#include <libiarray/iarray.h>
#include <contribs/tinyexpr/tinyexpr.h>
#include "iarray_private.h"

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

INA_API(ina_rc_t) iarray_slice(iarray_context_t *ctx, iarray_container_t *c, iarray_slice_param_t *params, iarray_container_t **container)
{

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

//INA_API(ina_rc_t) iarray_expr_bind_scalar_float(iarray_expression_t *e, const char *var, float val)
//{
//	iarray_container_t *c = ina_mempool_dalloc(e->mp, sizeof(iarray_container_t));
//	c->dtshape = ina_mempool_dalloc(e->mp, sizeof(iarray_dtshape_t));
//	c->dtshape->ndim = 0;
//	c->dtshape->dims = NULL;
//	c->dtshape->dtype = IARRAY_DATA_TYPE_FLOAT;
//	c->scalar_value.f = val;
//	return INA_SUCCESS;
//}

INA_API(ina_rc_t) iarray_expr_bind_scalar_double(iarray_expression_t *e, const char *var, double val)
{
	iarray_container_t *c = ina_mempool_dalloc(e->mp, sizeof(iarray_container_t));
	c->dtshape = ina_mempool_dalloc(e->mp, sizeof(iarray_dtshape_t));
	c->dtshape->ndim = 0;
	c->dtshape->dims = NULL;
	c->dtshape->dtype = IARRAY_DATA_TYPE_DOUBLE;
	c->scalar_value.d = val;
	return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_expr_compile(iarray_expression_t *e, const char *expr)
{
	return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_eval(iarray_context_t *ctx, iarray_expression_t *e, iarray_container_t **ret)
{
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

ina_rc_t iarray_temporary_new(iarray_expression_t *expr, iarray_container_t *c, iarray_dtshape_t *dtshape,
		iarray_temporary_t **temp)
{
	*temp = ina_mempool_dalloc(expr->mp, sizeof(iarray_temporary_t));
	(*temp)->dtshape = ina_mempool_dalloc(expr->mp, sizeof(iarray_dtshape_t));
	ina_mem_cpy((*temp)->dtshape, dtshape, sizeof(iarray_dtshape_t));
	size_t size = 0;
	iarray_shape_size(dtshape, &size);
	(*temp)->size = size;
	if (c != NULL) {
        // FIXME: support float values too
	    ina_mem_cpy(&(*temp)->scalar_value, &c->scalar_value, sizeof(double));
	}
	if (size > 0) {
		(*temp)->data = ina_mempool_dalloc(expr->mp, size);
	}

	return INA_SUCCESS;
}

static iarray_temporary_t* _iarray_op(iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_optype_t op)
{
	bool scalar = false;
	bool scalar_vector = false;
	bool vector_vector = false;
	iarray_dtshape_t dtshape;
	iarray_operation_type_t op_type = IARRAY_OPERATION_TYPE_BLAS1;
	ina_mem_set(&dtshape, 0, sizeof(iarray_dtshape_t));
	iarray_blas_type_t op_type = IARRAY_OPERATION_TYPE_BLAS1;
	iarray_temporary_t *scalar_tmp = NULL;
	iarray_temporary_t *scalar_lhs = NULL;
	iarray_temporary_t *out;
	iarray_expression_t expr; /* temp hack */
	ina_mem_set(&expr, 0, sizeof(iarray_expression_t));

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
			ina_mem_cpy(dtshape.dims, rhs->dtshape->dims, sizeof(int) * dtshape.ndim);
			scalar_tmp = lhs;
			scalar_lhs = rhs;
		}
		else {
			dtshape.dtype = lhs->dtshape->dtype;
			dtshape.ndim = lhs->dtshape->ndim;
			ina_mem_cpy(dtshape.dims, lhs->dtshape->dims, sizeof(int) * dtshape.ndim);
			scalar_tmp = rhs;
			scalar_lhs = lhs;
		}
		scalar_vector = true;
	}
	else if (lhs->dtshape->ndim == 1 && rhs->dtshape->ndim == 1) { /* vector-vector */
		dtshape.dtype = lhs->dtshape->dtype;
		dtshape.ndim = lhs->dtshape->ndim;
		ina_mem_cpy(dtshape.dims, lhs->dtshape->dims, sizeof(int)*lhs->dtshape->ndim);
		vector_vector = true;
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

iarray_temporary_t* _iarray_op_add(iarray_temporary_t *lhs, iarray_temporary_t *rhs)
{
	return _iarray_op(lhs, rhs, IARRAY_OPERATION_TYPE_ADD);
}

iarray_temporary_t* _iarray_op_sub(iarray_temporary_t *lhs, iarray_temporary_t *rhs)
{
	return _iarray_op(lhs, rhs, IARRAY_OPERATION_TYPE_SUB);
}

iarray_temporary_t* _iarray_op_mul(iarray_temporary_t *lhs, iarray_temporary_t *rhs)
{
	return _iarray_op(lhs, rhs, IARRAY_OPERATION_TYPE_MUL);
}

iarray_temporary_t* _iarray_op_divide(iarray_temporary_t *lhs, iarray_temporary_t *rhs)
{
	return _iarray_op(lhs, rhs, IARRAY_OPERATION_TYPE_DIVIDE);
}

int scalar_scalar()
{
	iarray_temporary_t *x1, *y1;
	iarray_expression_t iexpr;  // FIXME
	memset(&iexpr, 0, sizeof(iarray_expression_t));
	iarray_dtshape_t xshape = {
			.ndim = 0,
			.dtype = IARRAY_DATA_TYPE_DOUBLE,
	};
	iarray_dtshape_t yshape = {
			.ndim = 0,
			.dtype = IARRAY_DATA_TYPE_DOUBLE,
	};
	iarray_temporary_new(&iexpr, NULL, &xshape, &x1);
	iarray_temporary_new(&iexpr, NULL, &yshape, &y1);

	x1->scalar_value.d = 5.;
	y1->scalar_value.d = 3.;

	/* Store variable names and pointers. */
	te_variable vars[] = {{"x", &x1}, {"y", &y1}};

	int err;
	/* Compile the expression with variables. */
	te_expr *expr = te_compile("x + y", vars, 2, &err);

	if (expr) {
		const iarray_temporary_t *h1 = te_eval(expr);
		printf("h1: %f\n", h1->scalar_value.d);

		x1->scalar_value.d = 10.;
		const iarray_temporary_t *h2 = te_eval(expr);
		printf("h2: %f\n", h2->scalar_value.d);

		te_free(expr);
	} else {
		printf("Parse error at %d\n", err);
	}

	return 0;
}

int scalar_vector()
{
	iarray_temporary_t *x1, *y1;
	iarray_expression_t iexpr;
	memset(&iexpr, 0, sizeof(iarray_expression_t));
	iarray_dtshape_t xshape = {
			.ndim = 0,
			.dtype = IARRAY_DATA_TYPE_DOUBLE,
	};
	iarray_dtshape_t yshape = {
			.ndim = 1,
			.dims = {100},
			.dtype = IARRAY_DATA_TYPE_DOUBLE,
	};
	iarray_temporary_new(&iexpr, NULL, &xshape, &x1);
	iarray_temporary_new(&iexpr, NULL, &yshape, &y1);

	x1->scalar_value.d = 5.;
	for (int i = 0; i < 100; i++) {
		((double*)y1->data)[i] = 3.;
	}

	/* Store variable names and pointers. */
	te_variable vars[] = {{"x", &x1}, {"y", &y1}};

	int err;
	/* Compile the expression with variables. */
	te_expr *expr = te_compile("x + y", vars, 2, &err);

	if (expr) {
		const iarray_temporary_t *h1 = te_eval(expr);
		printf("h1: %f, %f\n", ((double*)h1->data)[0], ((double*)h1->data)[99]);

		x1->scalar_value.d = 10.;
		const iarray_temporary_t *h2 = te_eval(expr);
		printf("h2: %f, %f\n", ((double*)h2->data)[0], ((double*)h2->data)[99]);

		te_free(expr);
	} else {
		printf("Parse error at %d\n", err);
	}

	return 0;
}

int vector_vector()
{
	iarray_temporary_t *x1, *y1;
	iarray_expression_t iexpr;
	memset(&iexpr, 0, sizeof(iarray_expression_t));
	iarray_dtshape_t xshape = {
			.ndim = 1,
			.dims = {100},
			.dtype = IARRAY_DATA_TYPE_DOUBLE,
	};
	iarray_dtshape_t yshape = {
			.ndim = 1,
			.dims = {100},
			.dtype = IARRAY_DATA_TYPE_DOUBLE,
	};
	iarray_temporary_new(&iexpr, NULL, &xshape, &x1);
	iarray_temporary_new(&iexpr, NULL, &yshape, &y1);

	double var1 = 5.;
	for (int i = 0; i < 100; i++) {
		((double*)x1->data)[i] = var1;
	}
	double var2 = 3.;
	for (int i = 0; i < 100; i++) {
		((double*)y1->data)[i] = var2;
	}

	/* Store variable names and pointers. */
	te_variable vars[] = {{"x", &x1}, {"y", &y1}};

	int err;
	/* Compile the expression with variables. */
	te_expr *expr = te_compile("x + y", vars, 2, &err);

	if (expr) {
		const iarray_temporary_t *h1 = te_eval(expr);
		printf("h1: %f, %f\n", ((double*)h1->data)[0], ((double*)h1->data)[99]);

		for (int i = 0; i < 100; i++) {
			((double*)x1->data)[i] = 10.;
		}
		const iarray_temporary_t *h2 = te_eval(expr);
		printf("h2: %f, %f\n", ((double*)h2->data)[0], ((double*)h2->data)[99]);

		te_free(expr);
	} else {
		printf("Parse error at %d\n", err);
	}

	return 0;
}

INA_API(ina_rc_t) iarray_eval_chunk(char* expr, iarray_variable_t vars[], int nvars, iarray_variable_t out,
		iarray_data_type_t dtype, int *err)
{
	// Get the super-chunk container for storing out values
	blosc2_schunk *sc_out = (blosc2_schunk*)out.address;

	// Allocate space for temporaries
	iarray_expression_t iexpr;
	memset(&iexpr, 0, sizeof(iarray_expression_t));
	iarray_temporary_t **temp_vars = ina_mempool_dalloc(iexpr.mp, (size_t)nvars * sizeof(void*));
	te_variable *te_vars = ina_mempool_dalloc(iexpr.mp, (size_t)nvars * sizeof(te_variable));
	for (int nvar = 0; nvar < nvars; nvar++) {
		blosc2_schunk *schunk = (blosc2_schunk*)vars[0].address;
		iarray_dtshape_t shape_var = {
				.ndim = 1,
				.dims = {schunk->chunksize / schunk->typesize},
				.dtype = dtype,
		};
		iarray_temporary_new(&iexpr, NULL, &shape_var, &temp_vars[nvar]);
		te_vars[nvar].name = vars[nvar].name;
		te_vars[nvar].address = &temp_vars[nvar];
		te_vars[nvar].type = TE_VARIABLE;
		te_vars[nvar].context = NULL;
	}

	// Create and compile the expression
	te_expr *texpr = te_compile(expr, te_vars, nvars, err);

	// Evaluate the expression for all the chunks in variables
	blosc2_schunk *first_schunk = (blosc2_schunk*)vars[0].address;  // get the super-chunk of the first variable
	size_t isize = (size_t)first_schunk->chunksize;
	for (int nchunk = 0; nchunk < first_schunk->nchunks; nchunk++) {
		// Decompress chunks in variables into temporaries
		for (int nvar = 0; nvar < nvars; nvar++) {
			blosc2_schunk *schunk = (blosc2_schunk*)vars[nvar].address;  // get the super-chunk of the first variable
			int dsize = blosc2_schunk_decompress_chunk(schunk, nchunk, temp_vars[nvar]->data, isize);
			if (dsize < 0) {
				printf("Decompression error.  Error code: %d\n", dsize);
				return dsize;
			}
		}
		const iarray_temporary_t *expr_out = te_eval(texpr);
		blosc2_schunk_append_buffer(sc_out, expr_out->data, isize);
	}
	free(temp_vars);  // FIXME: do a recursive free
	free(te_vars);
	return 0;
}


INA_API(ina_rc_t) iarray_eval_block(char* expr, iarray_variable_t vars[], int nvars, iarray_variable_t out,
		iarray_data_type_t dtype, int *err)
{
	// Get the super-chunk container for storing out values
	blosc2_schunk *sc_out = (blosc2_schunk*)out.address;
	// Get info about the blocksize and other info about chunks in super-chunk vars
	// FIXME: what happens when the different operands have different blocksizes?
	blosc2_schunk *first_schunk = (blosc2_schunk*)vars[0].address;  // get the super-chunk of the first variable
	int typesize = first_schunk->typesize;
	int nchunks = first_schunk->nchunks;
	size_t chunksize, cbytes, blocksize;
	blosc_cbuffer_sizes(first_schunk->data[0], &chunksize, &cbytes, &blocksize);

	// Allocate space for temporaries
	iarray_expression_t iexpr;
	memset(&iexpr, 0, sizeof(iarray_expression_t));
	iarray_temporary_t **temp_vars = ina_mempool_dalloc(iexpr.mp, (size_t)nvars * sizeof(void*));
	te_variable *te_vars = ina_mempool_dalloc(iexpr.mp, (size_t)nvars * sizeof(te_variable));
	for (int nvar = 0; nvar < nvars; nvar++) {
		iarray_dtshape_t shape_var = {
				.ndim = 1,
				.dims = {(int)blocksize / typesize},
				.dtype = dtype,
		};
		iarray_temporary_new(&iexpr, NULL, &shape_var, &temp_vars[nvar]);
		te_vars[nvar].name = vars[nvar].name;
		te_vars[nvar].address = &temp_vars[nvar];
		te_vars[nvar].type = TE_VARIABLE;
		te_vars[nvar].context = NULL;
	}

	// Create buffer for output chunk
	int8_t *outbuf = ina_mempool_dalloc(iexpr.mp, chunksize);

	// Create and compile the expression
	te_expr *texpr = te_compile(expr, te_vars, nvars, err);

	// Evaluate the expression for all the chunks in variables
	int nblocks_in_chunk = (int)chunksize / (int)blocksize;
	if (nblocks_in_chunk * blocksize < chunksize) {
		nblocks_in_chunk += 1;
	}
	int nitems = (int)blocksize / typesize;
	size_t corrected_blocksize = blocksize;
	int corrected_nitems = nitems;
	for (int nchunk = 0; nchunk < nchunks; nchunk++) {
		for (int nblock = 0; nblock < nblocks_in_chunk; nblock++) {
			if ((nblock + 1 == nblocks_in_chunk) && (nblock + 1) * blocksize > chunksize) {
				corrected_blocksize = chunksize - nblock * blocksize;
				corrected_nitems = (int)corrected_blocksize / typesize;
			}
			// Decompress blocks in variables into temporaries
			for (int nvar = 0; nvar < nvars; nvar++) {
				blosc2_schunk *schunk = (blosc2_schunk*)vars[nvar].address;
				uint8_t *chunk = schunk->data[nchunk];
				int dsize = blosc_getitem(chunk, nblock * nitems, corrected_nitems, temp_vars[nvar]->data);
				if (dsize < 0) {
					printf("Decompression error.  Error code: %d\n", dsize);
					return dsize;
				}
			}
			// Evaluate the expression for this block
			const iarray_temporary_t *expr_out = te_eval(texpr);
			memcpy(outbuf + nblock * blocksize, expr_out->data, corrected_blocksize);
		}
		blosc2_schunk_append_buffer(sc_out, outbuf, (size_t)chunksize);
	}

	free(outbuf);
	free(temp_vars);  // FIXME: do a recursive free
	free(te_vars);
	return 0;
}


int _main_(int argc, char **argv) {

	printf("** scalar-scalar:\n");
	int retcode = scalar_scalar();
	printf("** scalar-vector:\n");
	retcode = scalar_vector();
	printf("** vector-vector:\n");
	retcode = vector_vector();

	return retcode;
}