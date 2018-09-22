//
// Created by Francesc Alted on 11/09/2018.
//

#include <stdio.h>
#include "blosc.h"
#include "iarray.h"
#include "iarray_private.h"
#include "tinyexpr.h"

#define KB (1024)
#define MB (1024 * KB)
#define GB (1024 * MB)


#define NCHUNKS  50
#define CHUNKSIZE (200 * 1000)  // fits well in modern L3 caches
#define NELEM (NCHUNKS * CHUNKSIZE)  // multiple of CHUNKSIZE for now
#define NTHREADS  4

typedef void* ina_mempool_t;
#define ina_mempool_dalloc(mp, size)

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

INA_API(ina_rc_t) iarray_expr_bind_scalar_float(iarray_expression_t *e, const char *var, float val)
{
	//iarray_container_t *c = ina_mempool_dalloc(e->mp, sizeof(iarray_container_t));
	//c->dtshape = ina_mempool_dalloc(e->mp, sizeof(iarray_dtshape_t));
	//c->dtshape->ndim = 0;
	//c->dtshape->dims = NULL;
	//c->dtshape->dtype = IARRAY_DATA_TYPE_FLOAT;
	//c->scalar_value.f = val;
	return 0;
}

ina_rc_t iarray_container_shape_size(iarray_dtshape_t *dtshape, size_t *size)
{
	size_t type_size;
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
	return 0;
}

ina_rc_t iarray_temporary_shape_size(iarray_dtshape_t *temp_dtshape, size_t *temp_size)
{
	size_t type_size;
	switch (temp_dtshape->dtype) {
		case IARRAY_DATA_TYPE_DOUBLE:
			type_size = sizeof(double);
			break;
		case IARRAY_DATA_TYPE_FLOAT:
			type_size = sizeof(float);
			break;
	}
	for (int i = 0; i < temp_dtshape->ndim; ++i) {
		*temp_size += temp_dtshape->dims[i] * type_size;
	}
	return 0;
}

ina_rc_t iarray_temporary_new(iarray_expression_t *expr, iarray_container_t *c, iarray_dtshape_t *dtshape, iarray_temporary_t **temp)
{
	//*temp = ina_mempool_dalloc(expr->mp, sizeof(iarray_temporary_t));
	*temp = malloc(sizeof(iarray_temporary_t));
	size_t size = 0;
	iarray_temporary_shape_size(dtshape, &size);
	//(*temp)->dtshape = ina_mempool_dalloc(expr->mp, sizeof(iarray_temporary_dtshape_t));
	memcpy((*temp)->dtshape, dtshape, sizeof(iarray_dtshape_t));
	(*temp)->size = size;
	if (c != NULL) {
		memcpy(&(*temp)->scalar_value, &c->scalar_value, sizeof(double));
	}
	//(*temp)->data = ina_mempool_dalloc(expr->mp, (*temp)->size);

	return 0;
}

ina_rc_t _iarray_op_add(iarray_expression_t *expr, iarray_temporary_t *lhs, iarray_temporary_t *rhs, iarray_temporary_t **out)
{
	int scalar = 0;
	iarray_dtshape_t shape;
	memset(&shape, 0, sizeof(iarray_dtshape_t));
	iarray_operation_type_t op_type = IARRAY_OPERATION_TYPE_BLAS1;
	iarray_temporary_t *scalar_tmp = NULL;
	iarray_temporary_t *scalar_lhs = NULL;

	if (lhs->dtshape->ndim == 0 || rhs->dtshape->ndim == 0) { /* scalar test */
		if (lhs->dtshape->ndim == 0) {
			shape.dtype = rhs->dtshape->dtype;
			shape.ndim = rhs->dtshape->ndim;
			memcpy(shape.dims, rhs->dtshape->dims, sizeof(int)*rhs->dtshape->ndim);
			scalar_tmp = lhs;
			scalar_lhs = rhs;
		}
		else {
			shape.dtype = lhs->dtshape->dtype;
			shape.ndim = lhs->dtshape->ndim;
			memcpy(shape.dims, lhs->dtshape->dims, sizeof(int)*lhs->dtshape->ndim);
			scalar_tmp = rhs;
			scalar_lhs = lhs;
		}
		scalar = 1;
	}
	else if (lhs->dtshape->ndim == 1 && rhs->dtshape->ndim == 1) { /* vector vector test */
		shape.dtype = lhs->dtshape->dtype;
		shape.ndim = lhs->dtshape->ndim;
		memcpy(shape.dims, lhs->dtshape->dims, sizeof(int)*lhs->dtshape->ndim);
	}
	else {
		/* FIXME: matrix/vector and matrix/matrix addition */
	}

	iarray_temporary_new(expr, NULL, &shape, out);

	switch (shape.dtype) {
		case IARRAY_DATA_TYPE_DOUBLE: 
		{
			int len = (int)(*out)->size / sizeof(double);
			if (scalar) {
				for (int i = 0; i < len; ++i) {
					((double*)(*out)->data)[i] = ((double*)scalar_lhs->data)[i] + scalar_tmp->scalar_value.d;
				}
			}
			else {
				for (int i = 0; i < len; ++i) {
					((double*)(*out)->data)[i] = ((double*)lhs->data)[i] + ((double*)rhs->data)[i];
				}
			}
		}
		break;
		case IARRAY_DATA_TYPE_FLOAT:
		{
			int len = (int)(*out)->size / sizeof(float);
			if (scalar) {
				for (int i = 0; i < len; ++i) {
					((float*)(*out)->data)[i] = ((float*)scalar_lhs->data)[i] + scalar_tmp->scalar_value.f;
				}
			}
			else {
				for (int i = 0; i < len; ++i) {
					((float*)(*out)->data)[i] = ((float*)lhs->data)[i] + ((float*)rhs->data)[i];
				}
			}
		}
		break;
	}

	return 0;
}

/*
  Example program demonstrating how to execute an expression with super-chunks as operands.

  To compile this program:

  $ gcc -O3 iarray.c -o iarray -lblosc

  To run:

  $ ./iarray
  ...

*/





// Fill X values in regular array
int fill_x(double *x) {
  double incx = 10. / NELEM;

  /* Fill even values between 0 and 10 */
  for (int i = 0; i < NELEM; i++) {
    x[i] = incx * i;
  }
  return 0;
}

// Compute and fill X values in a buffer
void fill_buffer(double *x, int nchunk) {
  double incx = 10. / NELEM;

  for (int i = 0; i < CHUNKSIZE; i++) {
    x[i] = incx * (nchunk * CHUNKSIZE + i);
  }
}

void fill_sc_x(blosc2_schunk *sc_x, const size_t isize) {
  double buffer_x[CHUNKSIZE];

  /* Fill with even values between 0 and 10 */
  for (int nchunk = 0; nchunk < NCHUNKS; nchunk++) {
    fill_buffer(buffer_x, nchunk);
    blosc2_schunk_append_buffer(sc_x, buffer_x, isize);
  }
}

double poly(const double x) {
  return (x - 1.35) * (x - 4.45) * (x - 8.5);
}

// Compute and fill Y values in regular array
void compute_y(const double *x, double *y) {
  for (int i = 0; i < NELEM; i++) {
    y[i] = poly(x[i]);
  }
}

// Compute and fill Y values in a buffer
void fill_buffer_y(const double *x, double *y) {
  for (int i = 0; i < CHUNKSIZE; i++) {
    y[i] = poly(x[i]);
  }
}


int main() {
  printf("Blosc version info: %s (%s)\n",
         BLOSC_VERSION_STRING, BLOSC_VERSION_DATE);

  blosc_init();

  const size_t isize = CHUNKSIZE * sizeof(double);
  double buffer_x[CHUNKSIZE];
  double buffer_y[CHUNKSIZE];
  int dsize;
  blosc2_cparams cparams = BLOSC_CPARAMS_DEFAULTS;
  blosc2_dparams dparams = BLOSC_DPARAMS_DEFAULTS;
  blosc2_schunk *sc_x, *sc_y;
  int nchunk;
  blosc_timestamp_t last, current;
  double ttotal;
  double prev_value;

  /* Create a super-chunk container for input (X values) */
  cparams.typesize = sizeof(double);
  cparams.compcode = BLOSC_LZ4;
  cparams.clevel = 5;
  cparams.filters[0] = BLOSC_TRUNC_PREC;
  cparams.filters_meta[0] = 23;  // treat doubles as floats
  cparams.nthreads = NTHREADS;
  dparams.nthreads = NTHREADS;

  // Fill the plain x operand
  static double x[NELEM];
  blosc_set_timestamp(&last);
  fill_x(x);
  blosc_set_timestamp(&current);
  ttotal = blosc_elapsed_secs(last, current);
  printf("Time for filling X values: %.3g s, %.1f MB/s\n",
         ttotal, sizeof(x) / (ttotal * MB));

  // Create and fill a super-chunk for the x operand
  sc_x = blosc2_new_schunk(cparams, dparams, NULL);
  blosc_set_timestamp(&last);
  fill_sc_x(sc_x, isize);
  blosc_set_timestamp(&current);
  ttotal = blosc_elapsed_secs(last, current);
  printf("Time for filling X values (compressed): %.3g s, %.1f MB/s\n",
         ttotal, sc_x->nbytes / (ttotal * MB));
  printf("Compression for X values: %.1f MB -> %.1f MB (%.1fx)\n",
         sc_x->nbytes / MB, sc_x->cbytes / MB,
         (1. * sc_x->nbytes) / sc_x->cbytes);

  // Compute the plain y vector
  static double y[NELEM];
  blosc_set_timestamp(&last);
  compute_y(x, y);
  blosc_set_timestamp(&current);
  ttotal = blosc_elapsed_secs(last, current);
  printf("Time for computing and filling Y values: %.3g s, %.1f MB/s\n",
         ttotal, sizeof(y) / (ttotal * MB));
  // To prevent the optimizer to be too smart and remove 'dead' code
  int retcode = y[0] > y[1];

  // Create a super-chunk container and compute y values
  sc_y = blosc2_new_schunk(cparams, dparams, NULL);
  blosc_set_timestamp(&last);
  for (nchunk = 0; nchunk < sc_x->nchunks; nchunk++) {
    dsize = blosc2_schunk_decompress_chunk(sc_x, nchunk, buffer_x, isize);
    if (dsize < 0) {
      printf("Decompression error.  Error code: %d\n", dsize);
      return dsize;
    }
    fill_buffer_y(buffer_x, buffer_y);
    blosc2_schunk_append_buffer(sc_y, buffer_y, isize);
  }
  blosc_set_timestamp(&current);
  ttotal = blosc_elapsed_secs(last, current);
  printf("Time for computing and filling Y values (compressed): %.3g s, %.1f MB/s\n",
         ttotal, sc_y->nbytes / (ttotal * MB));
  printf("Compression for Y values: %.1f MB -> %.1f MB (%.1fx)\n",
         sc_y->nbytes / MB, sc_y->cbytes / MB,
         (1. * sc_y->nbytes) / sc_y->cbytes);


  // Free resources
  blosc2_free_schunk(sc_x);
  blosc2_free_schunk(sc_y);

  blosc_destroy();

  double x1, y1;
  /* Store variable names and pointers. */
  te_variable vars[] = {{"x", &x1}, {"y", &y1}};

  int err;
  /* Compile the expression with variables. */
  te_expr *expr = te_compile("x+y", vars, 2, &err);

  if (expr) {
    x1 = 3; y1 = 4;
    const iarray_temporary_t *h1 = te_eval(expr); /* Returns 5. */
    //printf("h1: %f\n", h1);

    x1 = 5; y1 = 12;
    const iarray_temporary_t *h2 = te_eval(expr); /* Returns 13. */
    //printf("h2: %f\n", h2);

    te_free(expr);
  } else {
    printf("Parse error at %d\n", err);
  }

  return retcode;
}
