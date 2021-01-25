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


static ina_rc_t
test_gemm(iarray_context_t *ctx, iarray_data_type_t dtype, int typesize, const int64_t *xshape, const int64_t *xcshape,
          const int64_t *xbshape, int64_t *xblockshape, const int64_t *yshape, const int64_t *ycshape,
          const int64_t *ybshape, int64_t *yblockbshape, const int64_t *zshape, const int64_t *zcshape,
          const int64_t *zbshape)
{
    int xflag = CblasNoTrans;
    int yflag = CblasNoTrans;

    //Define iarray container x
    iarray_dtshape_t xdtshape;
    xdtshape.ndim = 2;
    xdtshape.dtype = dtype;
    size_t xsize = 1;
    for (int i = 0; i < xdtshape.ndim; ++i) {
        xdtshape.shape[i] = xshape[i];
        xsize *= xshape[i];
    }

    iarray_storage_t xstore;
    xstore.backend = xcshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    xstore.urlpath = NULL;
    xstore.enforce_frame = false;
    if (xcshape != NULL) {
        for (int i = 0; i < xdtshape.ndim; ++i) {
            xstore.chunkshape[i] = xcshape[i];
            xstore.blockshape[i] = xbshape[i];
        }
    }
    
    iarray_container_t *c_x;
    INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &xdtshape, 0, (double) xsize, 1, &xstore, 0, &c_x));

    // iarray container x to buffer
    uint8_t *xbuffer = ina_mem_alloc(xsize * typesize);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, xbuffer, xsize * typesize));

    //Define iarray container y
    iarray_dtshape_t ydtshape;
    ydtshape.ndim = 2;
    ydtshape.dtype = dtype;
    size_t ysize = 1;
    for (int i = 0; i < ydtshape.ndim; ++i) {
        ydtshape.shape[i] = yshape[i];
        ysize *= yshape[i];
    }

    iarray_storage_t ystore;
    ystore.backend = ycshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    ystore.urlpath = NULL;
    ystore.enforce_frame = false;
    if (ycshape != NULL) {
        for (int i = 0; i < ydtshape.ndim; ++i) {
            ystore.chunkshape[i] = ycshape[i];
            ystore.blockshape[i] = ybshape[i];
        }
    }
    iarray_container_t *c_y;
    INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &ydtshape, 0, (double) ysize, 1, &ystore, 0, &c_y));

    // iarray container y to buffer
    uint8_t *ybuffer = ina_mem_alloc(ysize * typesize);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_y, ybuffer, ysize * typesize));

    // define o buffer
    int64_t osize = c_x->dtshape->shape[0] * c_y->dtshape->shape[1];
    uint8_t *obuffer = ina_mem_alloc((size_t)osize * typesize);

    // MKL matrix-matrix multiplication
    int M = (int) c_x->dtshape->shape[0];
    int K = (int) c_x->dtshape->shape[1];
    int N = (int) c_y->dtshape->shape[1];

    int ldx = K;
    int ldy = N;
    int ldo = N;

    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            cblas_dgemm(CblasRowMajor, xflag, yflag, M, N, K, 1.0, (double *) xbuffer, ldx,
                        (double *) ybuffer, ldy, 0.0, (double *) obuffer, ldo);
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            cblas_sgemm(CblasRowMajor, xflag, yflag, M, N, K, 1.0f, (float *) xbuffer, ldx,
                        (float *) ybuffer, ldy, 0.0f, (float *) obuffer, ldo);
            break;
        default:
            return INA_ERR_EXCEEDED;
    }

    //Define iarray container z
    iarray_dtshape_t zdtshape;
    zdtshape.ndim = 2;
    zdtshape.dtype = dtype;
    int64_t zsize = 1;
    for (int i = 0; i < zdtshape.ndim; ++i) {
        zdtshape.shape[i] = zshape[i];
        zsize *= zshape[i];
    }

    iarray_storage_t zstore;
    zstore.backend = zcshape ? IARRAY_STORAGE_BLOSC : IARRAY_STORAGE_PLAINBUFFER;
    zstore.urlpath = NULL;
    zstore.enforce_frame = false;
    if (zcshape != NULL) {
        for (int i = 0; i < zdtshape.ndim; ++i) {
            zstore.chunkshape[i] = zcshape[i];
            zstore.blockshape[i] = zbshape[i];
        }
    }

    iarray_container_t *c_z;

    // iarray multiplication
    INA_TEST_ASSERT_SUCCEED(iarray_linalg_matmul(ctx, c_x, c_y, &zstore, &c_z));

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
                    printf("%"PRId64" - %.15f ", i, fabs(res));
                    return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
                }
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                res = (((float *) zbuffer)[i] - ((float *) obuffer)[i]) / ((float *) zbuffer)[i];
                if (fabs(res) > 1e-5) {
                    printf("%"PRId64" - %.6f ", i, fabs(res));
                    return INA_ERROR(INA_ERR_INVALID_ARGUMENT);
                }
                break;
            default:
                return INA_ERR_EXCEEDED;
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

INA_TEST_DATA(linalg_gemm) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(linalg_gemm) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(linalg_gemm) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(linalg_gemm, f_plain_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {150, 250};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int64_t *xblockshape = NULL;

    int64_t yshape[] = {250, 100};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int64_t *yblockshape = NULL;

    int64_t zshape[] = {150, 100};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                    xshape, xcshape, xbshape, xblockshape,
                                    yshape, ycshape, ybshape, yblockshape,
                                    zshape, zcshape, zbshape));
}


INA_TEST_FIXTURE(linalg_gemm, d_plain_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {100, 200};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int64_t *xblockshape = NULL;

    int64_t yshape[] = {200, 150};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int64_t *yblockshape = NULL;

    int64_t zshape[] = {100, 150};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape,
                                      yshape, ycshape, ybshape, yblockshape,
                                      zshape, zcshape, zbshape));
}


INA_TEST_FIXTURE(linalg_gemm, f_schunk_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {100, 170};
    int64_t xcshape[] = {27, 30};
    int64_t xbshape[] = {12, 7};

    int64_t xblockshape[] = {40, 20};

    int64_t yshape[] = {170, 21};
    int64_t ycshape[] = {20, 11};
    int64_t ybshape[] = {10, 5};

    int64_t yblockshape[] = {20, 3};

    int64_t zshape[] = {100, 21};
    int64_t zcshape[] = {40, 3};
    int64_t zbshape[] = {20, 3};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape,
                                      yshape, ycshape, ybshape, yblockshape,
                                      zshape, zcshape, zbshape));
}


INA_TEST_FIXTURE(linalg_gemm, d_schunk_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {300, 670};
    int64_t xcshape[] = {28, 300};
    int64_t xbshape[] = {11, 31};

    int64_t xblockshape[] = {43, 20};

    int64_t yshape[] = {670, 210};
    int64_t ycshape[] = {20, 45};
    int64_t ybshape[] = {7, 12};

    int64_t yblockshape[] = {20, 11};

    int64_t zshape[] = {300, 210};
    int64_t zcshape[] = {43, 11};
    int64_t zbshape[] = {13, 5};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape,
                                      yshape, ycshape, ybshape, yblockshape,
                                      zshape, zcshape, zbshape));
}


INA_TEST_FIXTURE(linalg_gemm, f_plain_plain2) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {100, 200};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int64_t *xblockshape = NULL;

    int64_t yshape[] = {200, 150};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int64_t *yblockshape = NULL;

    int64_t zshape[] = {100, 150};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape,
                                      yshape, ycshape, ybshape, yblockshape,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, d_plain_plain2) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {100, 200};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int64_t *xblockshape = NULL;

    int64_t yshape[] = {200, 150};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int64_t *yblockshape = NULL;

    int64_t zshape[] = {100, 150};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape,
                                      yshape, ycshape, ybshape, yblockshape,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, f_schunk_schunk2) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {100, 200};
    int64_t xcshape[] = {10, 30};
    int64_t xbshape[] = {10, 10};

    int64_t xblockshape[] = {20, 20};

    int64_t yshape[] = {200, 150};
    int64_t ycshape[] = {25, 30};
    int64_t ybshape[] = {15, 12};

    int64_t yblockshape[] = {20, 30};

    int64_t zshape[] = {100, 150};
    int64_t zcshape[] = {20, 30};
    int64_t zbshape[] = {11, 6};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape,
                                      yshape, ycshape, ybshape, yblockshape,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, d_schunk_schunk2) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {100, 200};
    int64_t xcshape[] = {10, 30};
    int64_t xbshape[] = {7, 7};

    int64_t xblockshape[] = {20, 20};

    int64_t yshape[] = {200, 150};
    int64_t ycshape[] = {25, 30};
    int64_t ybshape[] = {5, 11};

    int64_t yblockshape[] = {20, 30};

    int64_t zshape[] = {100, 150};
    int64_t zcshape[] = {20, 30};
    int64_t zbshape[] = {17, 7};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape,
                                      yshape, ycshape, ybshape, yblockshape,
                                      zshape, zcshape, zbshape));
}


INA_TEST_FIXTURE(linalg_gemm, f_plain_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {1230, 456};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int64_t *xblockshape = NULL;

    int64_t yshape[] = {456, 534};
    int64_t ycshape[] = {200, 210};
    int64_t ybshape[] = {23, 11};

    int64_t yblockshape[] = {456, 124};

    int64_t zshape[] = {1230, 534};
    int64_t zcshape[] = {1230, 124};
    int64_t zbshape[] = {13, 21};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape,
                                      yshape, ycshape, ybshape, yblockshape,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, d_plain_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {456, 1230};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int64_t *xblockshape = NULL;

    int64_t yshape[] = {1230, 534};
    int64_t ycshape[] = {1230, 210};
    int64_t ybshape[] = {55, 11};

    int64_t yblockshape[] = {1230, 210};

    int64_t zshape[] = {456, 534};
    int64_t zcshape[] = {456, 210};
    int64_t zbshape[] = {45, 21};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape,
                                      yshape, ycshape, ybshape, yblockshape,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, f_schunk_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {555, 433};
    int64_t xcshape[] = {123, 234};
    int64_t xbshape[] = {13, 24};

    int64_t xblockshape[] = {236, 111};

    int64_t yshape[] = {433, 678};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int64_t yblockshape[] = {111, 88};

    int64_t zshape[] = {555, 678};
    int64_t zcshape[] = {236, 88};
    int64_t zbshape[] = {26, 88};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape,
                                      yshape, ycshape, ybshape, yblockshape,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, d_schunk_plain_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {456, 1230};
    int64_t xcshape[] = {231, 124};
    int64_t xbshape[] = {23, 14};

    int64_t xblockshape[] = {123, 1230};

    int64_t yshape[] = {1230, 534};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int64_t *yblockshape = NULL;

    int64_t zshape[] = {456, 534};
    int64_t zcshape[] = {123, 534};
    int64_t zbshape[] = {13, 5};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape,
                                      yshape, ycshape, ybshape, yblockshape,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, d_schunk_plain_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {456, 1230};
    int64_t xcshape[] = {231, 456};
    int64_t xbshape[] = {21, 6};

    int64_t xblockshape[] = {456, 123};

    int64_t yshape[] = {1230, 534};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int64_t yblockshape[] = {123, 534};

    int64_t zshape[] = {456, 534};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape,
                                      yshape, ycshape, ybshape, yblockshape,
                                      zshape, zcshape, zbshape));
}
