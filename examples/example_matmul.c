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

int main(int argc, char **argv)
{
    iarray_init();
    ina_stopwatch_t *w = NULL;
    double elapsed_sec = 0;
    INA_STOPWATCH_NEW(-1, -1, &w);
    if (argc != 2) {
        return -1;
    }
    int n_threads = atoi(argv[1]);
    int8_t ndim = 2;
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    
    int64_t shape[] = {2000, 2000};
    int64_t size = 2000 * 2000;
    
    int64_t pshape_x[] = {0, 0};
    int64_t pshape_y[] = {0, 0};
    int64_t pshape_z[] = {0, 0};
    
    int64_t bshape_x[] = {2000, 2000};
    int64_t bshape_y[] = {2000, 2000};

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.max_num_threads = n_threads;
    iarray_context_t *ctx;
    iarray_context_new(&cfg, &ctx);

    iarray_dtshape_t dtshape_x;
    dtshape_x.ndim = ndim;
    dtshape_x.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape_x.shape[i] = shape[i];
        dtshape_x.pshape[i] = pshape_x[i];
    }
    iarray_container_t *c_x;
    iarray_linspace(ctx, &dtshape_x, size, 0, 1, NULL, 0, &c_x);

    iarray_dtshape_t dtshape_y;
    dtshape_y.ndim = ndim;
    dtshape_y.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape_y.shape[i] = shape[i];
        dtshape_y.pshape[i] = pshape_y[i];
    }
    
    iarray_container_t *c_y;
    iarray_linspace(ctx, &dtshape_y, size, 0, 1, NULL, 0, &c_y);

    iarray_dtshape_t dtshape_z;
    dtshape_z.ndim = ndim;
    dtshape_z.dtype = dtype;
    for (int i = 0; i < ndim; ++i) {
        dtshape_z.shape[i] = shape[i];
        dtshape_z.pshape[i] = pshape_z[i];
    }
    
    iarray_container_t *c_z;
    iarray_container_new(ctx, &dtshape_z, NULL, 0, &c_z);
    mkl_set_num_threads(n_threads);


    double *b_x = (double *) malloc(size * sizeof(double));
    double *b_y = (double *) malloc(size * sizeof(double));
    double *b_z = (double *) malloc(size * sizeof(double));
    double *b_res = (double *) malloc(size * sizeof(double));

    iarray_to_buffer(ctx, c_x, b_x, size * sizeof(double));
    iarray_to_buffer(ctx, c_y, b_y, size * sizeof(double));


    INA_STOPWATCH_START(w);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int) shape[0], (int) shape[1], (int) shape[1],
                1.0, b_x, (int) shape[1], b_y, (int) shape[1], 0.0, b_z, (int) shape[1]);
    INA_STOPWATCH_STOP(w);

    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));

    printf("Time mkl (C): %.4f\n", elapsed_sec);

    INA_STOPWATCH_START(w);
    INA_MUST_SUCCEED(iarray_linalg_matmul(ctx, c_x, c_y ,c_z, bshape_x, bshape_y, IARRAY_OPERATOR_GENERAL));
    INA_STOPWATCH_STOP(w);
    INA_MUST_SUCCEED(ina_stopwatch_duration(w, &elapsed_sec));

    printf("Time iarray: %.4f\n", elapsed_sec);

    iarray_to_buffer(ctx, c_z, b_res, size * sizeof(double));

    for (int i = 0; i < size; ++i) {
        if (fabs((b_res[i] - b_z[i]) / b_res[i]) > 1e-8) {
            printf("%f - %f = %f\n", b_res[i], b_z[i], b_res[i] - b_z[i]);
            printf("Error in element %d\n", i);
            return INA_ERROR(INA_ERR_ERROR);
        }
    }

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_z);
    free(b_x);
    free(b_y);
    free(b_z);
    free(b_res);
    iarray_context_free(&ctx);

    INA_STOPWATCH_FREE(&w);
    iarray_destroy();
    return EXIT_SUCCESS;
}
