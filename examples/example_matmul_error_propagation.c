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


int mult_c(const double *a, const double *b, double *c, const int I, const int J, const int K) {

    for (int i = 0; i < I; ++i) {
        for (int j = 0; j < J; ++j) {
            double sum = 0;
            for (int k = 0; k < K; ++k) {
                sum = sum + a[i * K + k] * b[k * J + j];
            }
            c[i * J + j] = sum;
        }
    }

    return 0;
}

int mult_mkl(const double *a, const double *b, double *c, const int I, const int J, const int K) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, I, J, K,
        1.0, a, (int) K, b, (int) J, 0.0, c, (int) J);
    return 0;
}


int mult_iarray(iarray_context_t *ctx, iarray_container_t *a, int64_t *bshape_a,
    iarray_container_t *b, int64_t *bshape_b, iarray_container_t *c) {
    INA_SUCCEED(iarray_linalg_matmul(ctx, a, b, c, bshape_a, bshape_b, IARRAY_OPERATOR_GENERAL));
    return 0;
}

double error_percent(const double *a, const double *b, uint64_t size) {
    int cont = 0;
    for (uint64_t i = 0; i < size; ++i) {
        double rel_error = fabs((a[i] - b[i]) / a[i]);
        if (rel_error > 1e-14) {
            cont++;
        }
    }
    return cont / (double) size;
}

int main()
{
    iarray_init();
    ina_rc_t rc;
    int n_threads = 1;
    int8_t ndim = 2;
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int64_t shape_a[] = {100, 100};
    int64_t shape_b[] = {100, 100};
    int64_t shape_z[] = {100, 100};

    int I = (int) shape_a[0];
    int J = (int) shape_b[1];
    int K = (int) shape_a[1];

    int64_t size_a = shape_a[0] * shape_a[1];
    int64_t size_b = shape_a[0] * shape_a[1];
    int64_t size_c = 100 * 100;


    int64_t pshape_a[] = {0, 0};
    int64_t pshape_b[] = {0, 0};
    int64_t pshape_c[] = {10, 10};

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.max_num_threads = n_threads;
    iarray_context_t *ctx;
    IARRAY_FAIL_IF_ERROR(iarray_context_new(&cfg, &ctx));

    iarray_dtshape_t dtshape_x;
    dtshape_x.ndim = ndim;
    dtshape_x.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape_x.shape[i] = shape_a[i];
        dtshape_x.pshape[i] = pshape_a[i];
    }
    iarray_container_t *cont_a;
    IARRAY_FAIL_IF_ERROR(iarray_linspace(ctx, &dtshape_x, size_a, -100, 100, NULL, 0, &cont_a));

    iarray_dtshape_t dtshape_y;
    dtshape_y.ndim = ndim;
    dtshape_y.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape_y.shape[i] = shape_b[i];
        dtshape_y.pshape[i] = pshape_b[i];
    }

    iarray_container_t *cont_b;
    IARRAY_FAIL_IF_ERROR(iarray_linspace(ctx, &dtshape_y, size_b, -100, 100, NULL, 0, &cont_b));

    iarray_dtshape_t dtshape_z;
    dtshape_z.ndim = ndim;
    dtshape_z.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape_z.shape[i] = shape_z[i];
        dtshape_z.pshape[i] = pshape_c[i];
    }

    iarray_container_t *cont_c;
    IARRAY_FAIL_IF_ERROR(iarray_container_new(ctx, &dtshape_z, NULL, 0, &cont_c));

    double *a = (double *) malloc(size_a * sizeof(double));
    double *b = (double *) malloc(size_b * sizeof(double));
    double *c_c = (double *) malloc(size_c * sizeof(double));
    double *c_mkl = (double *) malloc(size_c * sizeof(double));
    double *c_iarray = (double *) malloc(size_c * sizeof(double));

    IARRAY_FAIL_IF_ERROR(iarray_to_buffer(ctx, cont_a, a, size_a * sizeof(double)));
    IARRAY_FAIL_IF_ERROR(iarray_to_buffer(ctx, cont_b, b, size_b * sizeof(double)));

    mult_c(a, b, c_c, I, J, K);

    mult_mkl(a, b, c_mkl, I, J, K);

   int64_t bshape_a[] = {10, 10};
   int64_t bshape_b[] = {10, 10};

   mult_iarray(ctx, cont_a, bshape_a, cont_b, bshape_b, cont_c);

   IARRAY_FAIL_IF_ERROR(iarray_to_buffer(ctx, cont_c, c_iarray, size_c * sizeof(double)));

    printf("Error percentage (C - MKL): %.4f\n", error_percent(c_c, c_mkl, size_c));
    printf("Error percentage (C - iarray): %.4f\n", error_percent(c_c, c_iarray, size_c));
    printf("Error percentage (MKL - iarray): %.4f\n", error_percent(c_mkl, c_iarray, size_c));

    rc = INA_SUCCESS;
    goto cleanup;
    fail:
        rc = ina_err_get_rc();
    cleanup:
        iarray_container_free(ctx, &cont_a);
        iarray_container_free(ctx, &cont_b);
        iarray_container_free(ctx, &cont_c);
        free(a);
        free(b);
        free(c_c);
        free(c_mkl);
        free(c_iarray);
        iarray_context_free(&ctx);
        iarray_destroy();

    return rc;
}
