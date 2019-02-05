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

#include <tests/iarray_test.h>

static ina_rc_t test_gemv(iarray_context_t *ctx, iarray_data_type_t dtype, int typesize,
                          uint64_t *xshape, uint64_t *xpshape, uint64_t *xbshape, int xtrans,
                          uint64_t *yshape, uint64_t *ypshape, uint64_t *ybshape, int ytrans,
                          uint64_t *zshape, uint64_t *zpshape)
{
    int xflag = CblasNoTrans;

    //Define iarray container x
    iarray_dtshape_t xdtshape;
    xdtshape.ndim = 2;
    xdtshape.dtype = dtype;
    uint64_t xsize = 1;
    for (int i = 0; i < xdtshape.ndim; ++i) {
        xdtshape.shape[i] = xshape[i];
        xdtshape.pshape[i] = xpshape[i];
        xsize *= xshape[i];
    }
    iarray_container_t *c_x;
    INA_TEST_ASSERT_SUCCEED(iarray_linspace(ctx, &xdtshape, xsize, 0, 10, NULL, 0, &c_x));

    // iarray container x to buffer
    uint8_t *xbuffer = malloc(xsize * typesize);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, xbuffer, xsize * typesize));

    // transpose x
    if (xtrans == 1) {
        xflag = CblasTrans;
        INA_TEST_ASSERT_SUCCEED(iarray_linalg_transpose(ctx, c_x));
    }

    //Define iarray container y
    iarray_dtshape_t ydtshape;
    ydtshape.ndim = 1;
    ydtshape.dtype = dtype;
    uint64_t ysize = 1;
    for (int i = 0; i < ydtshape.ndim; ++i) {
        ydtshape.shape[i] = yshape[i];
        ydtshape.pshape[i] = ypshape[i];
        ysize *= yshape[i];
    }
    iarray_container_t *c_y;
    INA_TEST_ASSERT_SUCCEED(iarray_linspace(ctx, &ydtshape, ysize, 0, 10, NULL, 0, &c_y));

    // iarray container y to buffer
    uint8_t *ybuffer = malloc(ysize * typesize);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_y, ybuffer, ysize * typesize));


    // define o buffer
    uint64_t osize = c_x->dtshape->shape[0];

    uint8_t *obuffer = malloc(osize * typesize);

    // MKL matrix-matrix multiplication
    int M = (int) c_x->dtshape->shape[0];
    int K = (int) c_x->dtshape->shape[1];
    int ldx = K;

    if (xflag == CblasTrans) {
        ldx = M;
        M = (int) c_x->dtshape->shape[1];
        K = (int) c_x->dtshape->shape[0];
    }

    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            cblas_dgemv(CblasRowMajor, xflag, M, K, 1.0, (double *) xbuffer, ldx, (double *) ybuffer, 1, 0.0, (double *) obuffer, 1);
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            cblas_sgemv(CblasRowMajor, xflag, M, K, 1.0, (float *) xbuffer, ldx, (float *) ybuffer, 1, 0.0, (float *) obuffer, 1);
            break;
    }

    //Define iarray container z
    iarray_dtshape_t zdtshape;
    zdtshape.ndim = 1;
    zdtshape.dtype = dtype;
    uint64_t zsize = 1;
    for (int i = 0; i < zdtshape.ndim; ++i) {
        zdtshape.shape[i] = zshape[i];
        zdtshape.pshape[i] = zpshape[i];
        zsize *= zshape[i];
    }
    iarray_container_t *c_z;
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &zdtshape, NULL, 0, &c_z));

    // iarray multiplication
    INA_TEST_ASSERT_SUCCEED(iarray_linalg_matmul(ctx, c_x, c_y, c_z, xbshape, ybshape, IARRAY_OPERATOR_GENERAL));

    // define z buffer
    uint8_t *zbuffer = malloc(zsize * typesize);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_z, zbuffer, zsize * typesize));

    // assert
    double res;
    for (uint64_t i = 0; i < zsize; ++i) {
        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                res = (((double *) zbuffer)[i] - ((double *) obuffer)[i]) / ((double *) zbuffer)[i];
                if (res > 1e-14) {
                    return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
                }
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                res = (((float *) zbuffer)[i] - ((float *) obuffer)[i]) / ((float *) zbuffer)[i];
                if (res > 1e-5) {
                    return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
                }
                break;
        }
    }

    return INA_SUCCESS;
}

INA_TEST_DATA(linalg_gemv) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(linalg_gemv) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.flags = IARRAY_EXPR_EVAL_CHUNK;
    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(linalg_gemv) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(linalg_gemv, float_data_n) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    uint64_t xshape[] = {1000, 2000};
    uint64_t xpshape[] = {100, 300};

    uint64_t xbshape[] = {200, 200};
    int xtrans = 0;

    uint64_t yshape[] = {2000};
    uint64_t ypshape[] = {250};

    uint64_t ybshape[] = {200};
    int ytrans = 0;

    uint64_t zshape[] = {1000};
    uint64_t zpshape[] = {200};

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize, xshape, xpshape, xbshape, xtrans,
                                      yshape, ypshape, ybshape, ytrans, zshape, zpshape));
}



INA_TEST_FIXTURE(linalg_gemv, double_data_n) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    uint64_t xshape[] = {1300, 1670};
    uint64_t xpshape[] = {287, 300};

    uint64_t xbshape[] = {430, 200};
    int xtrans = 0;


    uint64_t yshape[] = {1670};
    uint64_t ypshape[] = {200};

    uint64_t ybshape[] = {200};
    int ytrans = 0;

    uint64_t zshape[] = {1300};
    uint64_t zpshape[] = {430};

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize, xshape, xpshape, xbshape, xtrans,
                                      yshape, ypshape, ybshape, ytrans, zshape, zpshape));
}


INA_TEST_FIXTURE(linalg_gemv, double_data_t) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    uint64_t xshape[] = {1670, 1300};
    uint64_t xpshape[] = {287, 300};

    uint64_t xbshape[] = {430, 200};
    int xtrans = 1;


    uint64_t yshape[] = {1670};
    uint64_t ypshape[] = {200};

    uint64_t ybshape[] = {200};
    int ytrans = 0;

    uint64_t zshape[] = {1300};
    uint64_t zpshape[] = {430};

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize, xshape, xpshape, xbshape, xtrans,
                                      yshape, ypshape, ybshape, ytrans, zshape, zpshape));
}


INA_TEST_FIXTURE(linalg_gemv, float_data_t) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    uint64_t xshape[] = {900, 650};
    uint64_t xpshape[] = {200, 140};

    uint64_t xbshape[] = {155, 300};
    int xtrans = 1;

    uint64_t yshape[] = {900};
    uint64_t ypshape[] = {421};

    uint64_t ybshape[] = {300};
    int ytrans = 1;

    uint64_t zshape[] = {650};
    uint64_t zpshape[] = {155};

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize, xshape, xpshape, xbshape, xtrans,
                                      yshape, ypshape, ybshape, ytrans, zshape, zpshape));
}
