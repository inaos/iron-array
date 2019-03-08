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


static ina_rc_t test_gemv(iarray_context_t *ctx, iarray_data_type_t dtype, int typesize,
                          const int64_t *xshape, const int64_t *xpshape, int64_t *xbshape,
                          int xtrans, const int64_t *yshape, const int64_t *ypshape,
                          int64_t *ybshape, const int64_t *zshape, const int64_t *zpshape)
{
    int xflag = CblasNoTrans;

    //Define iarray container x
    iarray_dtshape_t xdtshape;
    xdtshape.ndim = 2;
    xdtshape.dtype = dtype;
    size_t xsize = 1;
    for (int i = 0; i < xdtshape.ndim; ++i) {
        xdtshape.shape[i] = xshape[i];
        xdtshape.pshape[i] = xpshape[i];
        xsize *= xshape[i];
    }
    iarray_container_t *c_x;
    INA_TEST_ASSERT_SUCCEED(iarray_linspace(ctx, &xdtshape, (int64_t)xsize, 0, 10, NULL, 0, &c_x));

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
    size_t ysize = 1;
    for (int i = 0; i < ydtshape.ndim; ++i) {
        ydtshape.shape[i] = yshape[i];
        ydtshape.pshape[i] = ypshape[i];
        ysize *= yshape[i];
    }
    iarray_container_t *c_y;
    INA_TEST_ASSERT_SUCCEED(iarray_linspace(ctx, &ydtshape, (int64_t)ysize, 0, 10, NULL, 0, &c_y));

    // iarray container y to buffer
    uint8_t *ybuffer = malloc(ysize * typesize);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_y, ybuffer, ysize * typesize));


    // define o buffer
    int64_t osize = c_x->dtshape->shape[0];

    uint8_t *obuffer = malloc((size_t)osize * typesize);

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
        default:
            return INA_ERR_EXCEEDED;
    }

    //Define iarray container z
    iarray_dtshape_t zdtshape;
    zdtshape.ndim = 1;
    zdtshape.dtype = dtype;
    size_t zsize = 1;
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
    for (size_t i = 0; i < zsize; ++i) {
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
            default:
                return INA_ERR_EXCEEDED;
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
    cfg.eval_flags = IARRAY_EXPR_EVAL_CHUNK;
    iarray_context_new(&cfg, &data->ctx);
}

INA_TEST_TEARDOWN(linalg_gemv) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(linalg_gemv, float_data_n) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {1000, 2000};
    int64_t xpshape[] = {100, 300};

    int64_t xbshape[] = {200, 200};
    int xtrans = 0;

    int64_t yshape[] = {2000};
    int64_t ypshape[] = {250};

    int64_t ybshape[] = {200};

    int64_t zshape[] = {1000};
    int64_t zpshape[] = {200};

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize, xshape, xpshape, xbshape, xtrans,
                                      yshape, ypshape, ybshape, zshape, zpshape));
}



INA_TEST_FIXTURE(linalg_gemv, double_data_n) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {1300, 1670};
    int64_t xpshape[] = {287, 300};

    int64_t xbshape[] = {430, 200};
    int xtrans = 0;


    int64_t yshape[] = {1670};
    int64_t ypshape[] = {200};

    int64_t ybshape[] = {200};

    int64_t zshape[] = {1300};
    int64_t zpshape[] = {430};

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize, xshape, xpshape, xbshape, xtrans,
                                      yshape, ypshape, ybshape, zshape, zpshape));
}


INA_TEST_FIXTURE(linalg_gemv, double_data_t) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {1670, 1300};
    int64_t xpshape[] = {287, 300};

    int64_t xbshape[] = {430, 200};
    int xtrans = 1;

    int64_t yshape[] = {1670};
    int64_t ypshape[] = {200};

    int64_t ybshape[] = {200};

    int64_t zshape[] = {1300};
    int64_t zpshape[] = {430};

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize, xshape, xpshape, xbshape, xtrans,
                                      yshape, ypshape, ybshape, zshape, zpshape));
}


INA_TEST_FIXTURE(linalg_gemv, float_data_t) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {900, 650};
    int64_t xpshape[] = {200, 140};

    int64_t xbshape[] = {155, 300};
    int xtrans = 1;

    int64_t yshape[] = {900};
    int64_t ypshape[] = {421};

    int64_t ybshape[] = {300};

    int64_t zshape[] = {650};
    int64_t zpshape[] = {155};

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize, xshape, xpshape, xbshape, xtrans,
                                      yshape, ypshape, ybshape, zshape, zpshape));
}
