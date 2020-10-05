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

#include "src/iarray_private.h"
#include <libiarray/iarray.h>


static ina_rc_t test_gemv(iarray_context_t *ctx, iarray_data_type_t dtype, int typesize,
                          const int64_t *xshape, const int64_t *xcshape, const int64_t *xbshape, int64_t *xblockshape, int xtrans,
                          const int64_t *yshape, const int64_t *ycshape, const int64_t *ybshape, int64_t *yblockshape,
                          const int64_t *zshape, const int64_t *zcshape, const int64_t *zbshape)
{
    int xflag = CblasNoTrans;

    //Define iarray container x
    iarray_dtshape_t xdtshape;
    xdtshape.ndim = 2;
    xdtshape.dtype = dtype;
    int64_t xsize = 1;
    for (int i = 0; i < xdtshape.ndim; ++i) {
        xdtshape.shape[i] = xshape[i];
        xsize *= xshape[i];
    }

    iarray_storage_t xstore;
    xstore.backend = xcshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    xstore.filename = NULL;
    xstore.enforce_frame = false;
    if (xcshape != NULL) {
        for (int i = 0; i < xdtshape.ndim; ++i) {
            xstore.chunkshape[i] = xcshape[i];
            xstore.blockshape[i] = xbshape[i];
        }
    }
    iarray_container_t *c_x;
    INA_TEST_ASSERT_SUCCEED(iarray_linspace(ctx, &xdtshape, xsize, 0, 10, &xstore, 0, &c_x));

    // iarray container x to buffer
    uint8_t *xbuffer = ina_mem_alloc(xsize * typesize);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, xbuffer, xsize * typesize));

    //Define iarray container y
    iarray_dtshape_t ydtshape;
    ydtshape.ndim = 1;
    ydtshape.dtype = dtype;
    int64_t ysize = 1;
    for (int i = 0; i < ydtshape.ndim; ++i) {
        ydtshape.shape[i] = yshape[i];
        ysize *= yshape[i];
    }

    iarray_storage_t ystore;
    ystore.backend = ycshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    ystore.filename = NULL;
    ystore.enforce_frame = false;
    if (ycshape != NULL) {
        for (int i = 0; i < ydtshape.ndim; ++i) {
            ystore.chunkshape[i] = ycshape[i];
            ystore.blockshape[i] = ybshape[i];
        }
    }

    iarray_container_t *c_y;
    INA_TEST_ASSERT_SUCCEED(iarray_linspace(ctx, &ydtshape, ysize, 0, 10, &ystore, 0, &c_y));

    // iarray container y to buffer
    uint8_t *ybuffer = ina_mem_alloc(ysize * typesize);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_y, ybuffer, ysize * typesize));


    // define o buffer
    int64_t osize = c_x->dtshape->shape[0];

    uint8_t *obuffer = ina_mem_alloc((size_t)osize * typesize);

    // MKL matrix-matrix multiplication
    int M = (int) c_x->dtshape->shape[0];
    int K = (int) c_x->dtshape->shape[1];
    int ldx = K;

    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            cblas_dgemv(CblasRowMajor, xflag, M, K, 1.0, (double *) xbuffer, ldx, (double *) ybuffer, 1, 0.0, (double *) obuffer, 1);
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            cblas_sgemv(CblasRowMajor, xflag, M, K, 1.0f, (float *) xbuffer, ldx, (float *) ybuffer, 1, 0.0f, (float *) obuffer, 1);
            break;
        default:
            return INA_ERR_EXCEEDED;
    }

    //Define iarray container z
    iarray_dtshape_t zdtshape;
    zdtshape.ndim = 1;
    zdtshape.dtype = dtype;
    int64_t zsize = 1;
    for (int i = 0; i < zdtshape.ndim; ++i) {
        zdtshape.shape[i] = zshape[i];
        zsize *= zshape[i];
    }

    iarray_storage_t zstore;
    zstore.backend = zcshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    zstore.filename = NULL;
    zstore.enforce_frame = false;
    if (zcshape != NULL) {
        for (int i = 0; i < zdtshape.ndim; ++i) {
            zstore.chunkshape[i] = zcshape[i];
            zstore.blockshape[i] = zbshape[i];
        }
    }
    iarray_container_t *c_z;
    INA_TEST_ASSERT_SUCCEED(iarray_container_new(ctx, &zdtshape, &zstore, 0, &c_z));


    // iarray multiplication
    INA_TEST_ASSERT_SUCCEED(iarray_linalg_matmul(ctx, c_x, c_y, c_z, xblockshape, yblockshape, IARRAY_OPERATOR_GENERAL));

    // define z buffer
    uint8_t *zbuffer = ina_mem_alloc(zsize * typesize);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_z, zbuffer, zsize * typesize));

    // assert
    double res;
    for (int64_t i = 0; i < zsize; ++i) {
        switch (dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                res = (((double *) zbuffer)[i] - ((double *) obuffer)[i]) / ((double *) zbuffer)[i];
                if (fabs(res) > 1e-14) {
                    return INA_ERROR(INA_ERR_INVALID_PATTERN);
                }
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                res = (((float *) zbuffer)[i] - ((float *) obuffer)[i]) / ((float *) zbuffer)[i];
                if (fabs(res) > 1e-5) {
                    return INA_ERROR(INA_ERR_INVALID_PATTERN);
                }
                break;
            default:
                printf("Unhandled data type\n");
                return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
        }
    }

    iarray_container_free(ctx, &c_x);
    iarray_container_free(ctx, &c_y);
    iarray_container_free(ctx, &c_z);

    INA_MEM_FREE_SAFE(xbuffer);
    INA_MEM_FREE_SAFE(ybuffer);
    INA_MEM_FREE_SAFE(obuffer);
    INA_MEM_FREE_SAFE(zbuffer);
    
    return INA_SUCCESS;
}

INA_TEST_DATA(linalg_gemv) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(linalg_gemv) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(linalg_gemv) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(linalg_gemv, f_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {157, 200};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int64_t *xblockshape = NULL;
    int xtrans = 0;

    int64_t yshape[] = {200};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int64_t *yblockshape = NULL;

    int64_t zshape[] = {157};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize,
                                           xshape, xcshape, xbshape, xblockshape, xtrans,
                                           yshape, ycshape, ybshape, yblockshape,
                                           zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemv, d_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {10, 10};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int64_t *xblockshape = NULL;
    int xtrans = 0;

    int64_t yshape[] = {10};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int64_t *yblockshape = NULL;

    int64_t zshape[] = {10};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize,
                                           xshape, xcshape, xbshape, xblockshape, xtrans,
                                           yshape, ycshape, ybshape, yblockshape,
                                           zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemv, f_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {234, 200};
    int64_t xcshape[] = {11, 33};
    int64_t xbshape[] = {5, 10};

    int64_t xblockshape[] = {234, 20};
    int xtrans = 0;

    int64_t yshape[] = {200};
    int64_t ycshape[] = {100};
    int64_t ybshape[] = {40};

    int64_t yblockshape[] = {20};

    int64_t zshape[] = {234};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize,
                                           xshape, xcshape, xbshape, xblockshape, xtrans,
                                           yshape, ycshape, ybshape, yblockshape,
                                           zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemv, d_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {100, 200};
    int64_t xcshape[] = {30, 30};
    int64_t xbshape[] = {10, 15};

    int64_t xblockshape[] = {80, 20};
    int xtrans = 0;

    int64_t yshape[] = {200};
    int64_t ycshape[] = {125};
    int64_t ybshape[] = {45};

    int64_t yblockshape[] = {20};

    int64_t zshape[] = {100};
    int64_t zcshape[] = {80};
    int64_t zbshape[] = {37};

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize,
                                           xshape, xcshape, xbshape, xblockshape, xtrans,
                                           yshape, ycshape, ybshape, yblockshape,
                                           zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemv, f_plain_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {130, 160};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int64_t *xblockshape = NULL;
    int xtrans = 1;

    int64_t yshape[] = {160};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int64_t *yblockshape = NULL;

    int64_t zshape[] = {130};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize,
                                           xshape, xcshape, xbshape, xblockshape, xtrans,
                                           yshape, ycshape, ybshape, yblockshape,
                                           zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemv, d_plain_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {100, 167};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int64_t *xblockshape = NULL;
    int xtrans = 1;

    int64_t yshape[] = {167};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int64_t *yblockshape = NULL;

    int64_t zshape[] = {100};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize,
                                           xshape, xcshape, xbshape, xblockshape, xtrans,
                                           yshape, ycshape, ybshape, yblockshape,
                                           zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemv, f_schunk_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {650, 900};
    int64_t xcshape[] = {200, 140};
    int64_t xbshape[] = {20, 14};

    int64_t xblockshape[] = {155, 300};
    int xtrans = 1;

    int64_t yshape[] = {900};
    int64_t ycshape[] = {421};
    int64_t ybshape[] = {55};

    int64_t yblockshape[] = {300};

    int64_t zshape[] = {650};
    int64_t zcshape[] = {155};
    int64_t zbshape[] = {47};

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize,
                                           xshape, xcshape, xbshape, xblockshape, xtrans,
                                           yshape, ycshape, ybshape, yblockshape,
                                           zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemv, d_schunk_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {60, 300};
    int64_t xcshape[] = {14, 200};
    int64_t xbshape[] = {5, 14};

    int64_t xblockshape[] = {18, 30};
    int xtrans = 1;

    int64_t yshape[] = {300};
    int64_t ycshape[] = {41};
    int64_t ybshape[] = {17};

    int64_t yblockshape[] = {30};

    int64_t zshape[] = {60};
    int64_t zcshape[] = {18};
    int64_t zbshape[] = {18};

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize,
                                           xshape, xcshape, xbshape, xblockshape, xtrans,
                                           yshape, ycshape, ybshape, yblockshape,
                                           zshape, zcshape, zbshape));;
}

INA_TEST_FIXTURE(linalg_gemv, f_schunk_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {900, 650};
    int64_t xcshape[] = {200, 140};
    int64_t xbshape[] = {11, 3};

    int64_t xblockshape[] = {155, 650};
    int xtrans = 0;

    int64_t yshape[] = {650};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int64_t *yblockshape = NULL;

    int64_t zshape[] = {900};
    int64_t zcshape[] = {155};
    int64_t zbshape[] = {31};

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize,
                                           xshape, xcshape, xbshape, xblockshape, xtrans,
                                           yshape, ycshape, ybshape, yblockshape,
                                           zshape, zcshape, zbshape));;
}

INA_TEST_FIXTURE(linalg_gemv, d_plain_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {60, 300};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int64_t *xblockshape = NULL;
    int xtrans = 1;

    int64_t yshape[] = {300};
    int64_t ycshape[] = {41};
    int64_t ybshape[] = {21};

    int64_t yblockshape[] = {300};

    int64_t zshape[] = {60};
    int64_t zcshape[] = {60};
    int64_t zbshape[] = {60};

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize,
                                           xshape, xcshape, xbshape, xblockshape, xtrans,
                                           yshape, ycshape, ybshape, yblockshape,
                                           zshape, zcshape, zbshape));;
}

