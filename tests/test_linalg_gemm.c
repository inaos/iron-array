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


static ina_rc_t test_gemm(iarray_context_t *ctx, iarray_data_type_t dtype, int typesize,
                          const int64_t *xshape, const int64_t *xcshape, const int64_t *xbshape, int xtrans,
                          const int64_t *yshape, const int64_t *ycshape, const int64_t *ybshape, int ytrans,
                          const int64_t *zshape, const int64_t *zcshape, const int64_t *zbshape)
{
    int xflag = CblasNoTrans;
    int yflag = CblasNoTrans;

    //Define iarray container x
    iarray_dtshape_t xdtshape = {0};
    xdtshape.ndim = 2;
    xdtshape.dtype = dtype;
    size_t xsize = 1;
    for (int i = 0; i < xdtshape.ndim; ++i) {
        xdtshape.shape[i] = xshape[i];
        xsize *= xshape[i];
    }

    iarray_storage_t xstore = {0};
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
    INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &xdtshape, 0, (double) xsize, 1, &xstore, 0, &c_x));

    // iarray container x to buffer
    uint8_t *xbuffer = ina_mem_alloc(xsize * typesize);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, xbuffer, xsize * typesize));

    // transpose x
    if (xtrans == 1) {
        xflag = CblasTrans;
        INA_TEST_ASSERT_SUCCEED(iarray_linalg_transpose(ctx, c_x));
    }

    //Define iarray container y
    iarray_dtshape_t ydtshape = {0};
    ydtshape.ndim = 2;
    ydtshape.dtype = dtype;
    size_t ysize = 1;
    for (int i = 0; i < ydtshape.ndim; ++i) {
        ydtshape.shape[i] = yshape[i];
        ysize *= yshape[i];
    }

    iarray_storage_t ystore = {0};
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
    INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &ydtshape, 0, (double) ysize, 1, &ystore, 0, &c_y));

    // iarray container y to buffer
    uint8_t *ybuffer = ina_mem_alloc(ysize * typesize);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_y, ybuffer, ysize * typesize));

    // transpose y
    if (ytrans == 1) {
        yflag = CblasTrans;
        INA_TEST_ASSERT_SUCCEED(iarray_linalg_transpose(ctx, c_y));
    }

    // define o buffer
    int64_t osize = c_x->dtshape->shape[0] * c_y->dtshape->shape[1];
    uint8_t *obuffer = ina_mem_alloc((size_t)osize * typesize);

    // MKL matrix-matrix multiplication
    int M = (int) c_x->dtshape->shape[0];
    int K = (int) c_x->dtshape->shape[1];
    int N = (int) c_y->dtshape->shape[1];

    int ldx = K;
    if (xflag == CblasTrans) {
        ldx = M;
    }
    int ldy = N;
    if (yflag == CblasTrans) {
        ldy = K;
    }
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
    iarray_dtshape_t zdtshape = {0};
    zdtshape.ndim = 2;
    zdtshape.dtype = dtype;
    int64_t zsize = 1;
    for (int i = 0; i < zdtshape.ndim; ++i) {
        zdtshape.shape[i] = zshape[i];
        zsize *= zshape[i];
    }

    iarray_storage_t zstore = {0};
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

    // iarray multiplication
    INA_TEST_ASSERT_SUCCEED(iarray_linalg_parallel_matmul(ctx, c_x, c_y, &zstore, &c_z));

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

INA_TEST_FIXTURE(linalg_gemm, f_notrans_notrans_plain_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {150, 250};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int xtrans = 0;

    int64_t yshape[] = {250, 100};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int ytrans = 0;

    int64_t zshape[] = {150, 100};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                           xshape, xcshape, xbshape, xtrans,
                                           yshape, ycshape, ybshape, ytrans,
                                           zshape, zcshape, zbshape));
}


INA_TEST_FIXTURE(linalg_gemm, d_notrans_notrans_plain_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {100, 200};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int xtrans = 0;

    int64_t yshape[] = {200, 150};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int ytrans = 0;

    int64_t zshape[] = {100, 150};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xtrans,
                                      yshape, ycshape, ybshape, ytrans,
                                      zshape, zcshape, zbshape));
}


INA_TEST_FIXTURE(linalg_gemm, f_notrans_notrans_schunk_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {100, 170};
    int64_t xcshape[] = {27, 30};
    int64_t xbshape[] = {12, 7};

    int xtrans = 0;


    int64_t yshape[] = {170, 21};
    int64_t ycshape[] = {20, 11};
    int64_t ybshape[] = {10, 5};

    int ytrans = 0;

    int64_t zshape[] = {100, 21};
    int64_t zcshape[] = {40, 3};
    int64_t zbshape[] = {20, 3};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xtrans,
                                      yshape, ycshape, ybshape, ytrans,
                                      zshape, zcshape, zbshape));
}


INA_TEST_FIXTURE(linalg_gemm, d_notrans_notrans_schunk_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {300, 670};
    int64_t xcshape[] = {28, 300};
    int64_t xbshape[] = {11, 31};

    int xtrans = 0;


    int64_t yshape[] = {670, 210};
    int64_t ycshape[] = {20, 45};
    int64_t ybshape[] = {7, 12};

    int ytrans = 0;

    int64_t zshape[] = {300, 210};
    int64_t zcshape[] = {43, 11};
    int64_t zbshape[] = {13, 5};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xtrans,
                                      yshape, ycshape, ybshape, ytrans,
                                      zshape, zcshape, zbshape));
}


INA_TEST_FIXTURE(linalg_gemm, f_notrans_trans_plain_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {100, 200};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int xtrans = 0;

    int64_t yshape[] = {150, 200};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int ytrans = 1;

    int64_t zshape[] = {100, 150};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xtrans,
                                      yshape, ycshape, ybshape, ytrans,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, d_notrans_trans_plain_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {100, 200};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int xtrans = 0;

    int64_t yshape[] = {150, 200};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int ytrans = 1;

    int64_t zshape[] = {100, 150};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xtrans,
                                      yshape, ycshape, ybshape, ytrans,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, f_notrans_trans_schunk_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {100, 200};
    int64_t xcshape[] = {10, 30};
    int64_t xbshape[] = {10, 10};

    int xtrans = 0;

    int64_t yshape[] = {150, 200};
    int64_t ycshape[] = {25, 30};
    int64_t ybshape[] = {15, 12};

    int ytrans = 1;

    int64_t zshape[] = {100, 150};
    int64_t zcshape[] = {20, 30};
    int64_t zbshape[] = {11, 6};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xtrans,
                                      yshape, ycshape, ybshape, ytrans,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, d_notrans_trans_schunk_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {100, 200};
    int64_t xcshape[] = {10, 30};
    int64_t xbshape[] = {7, 7};

    int xtrans = 0;

    int64_t yshape[] = {150, 200};
    int64_t ycshape[] = {25, 30};
    int64_t ybshape[] = {5, 11};

    int ytrans = 1;

    int64_t zshape[] = {100, 150};
    int64_t zcshape[] = {20, 30};
    int64_t zbshape[] = {17, 7};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xtrans,
                                      yshape, ycshape, ybshape, ytrans,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, f_trans_notrans_plain_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {170, 130};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int xtrans = 1;

    int64_t yshape[] = {170, 210};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int ytrans = 0;

    int64_t zshape[] = {130, 210};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xtrans,
                                      yshape, ycshape, ybshape, ytrans,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, d_trans_notrans_plain_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {167, 100};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int xtrans = 1;

    int64_t yshape[] = {167, 200};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int ytrans = 0;

    int64_t zshape[] = {100, 200};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xtrans,
                                      yshape, ycshape, ybshape, ytrans,
                                      zshape, zcshape, zbshape));
}


INA_TEST_FIXTURE(linalg_gemm, f_trans_trans_schunk_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {900, 650};
    int64_t xcshape[] = {200, 140};
    int64_t xbshape[] = {21, 14};

    int xtrans = 1;

    int64_t yshape[] = {874, 900};
    int64_t ycshape[] = {300, 421};
    int64_t ybshape[] = {33, 7};

    int ytrans = 1;

    int64_t zshape[] = {650, 874};
    int64_t zcshape[] = {155, 234};
    int64_t zbshape[] = {15, 23};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xtrans,
                                      yshape, ycshape, ybshape, ytrans,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, d_trans_trans_schunk_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {200, 350};
    int64_t xcshape[] = {78, 140};
    int64_t xbshape[] = {7, 17};

    int xtrans = 1;

    int64_t yshape[] = {150, 200};
    int64_t ycshape[] = {34, 42};
    int64_t ybshape[] = {11, 5};

    int ytrans = 1;

    int64_t zshape[] = {350, 150};
    int64_t zcshape[] = {125, 43};
    int64_t zbshape[] = {12, 4};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xtrans,
                                      yshape, ycshape, ybshape, ytrans,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, f_trans_trans_plain_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {345, 212};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int xtrans = 1;

    int64_t yshape[] = {432, 345};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int ytrans = 1;

    int64_t zshape[] = {212, 432};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xtrans,
                                      yshape, ycshape, ybshape, ytrans,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, d_trans_trans_plain_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {1230, 456};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int xtrans = 1;

    int64_t yshape[] = {874, 1230};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int ytrans = 1;

    int64_t zshape[] = {456, 874};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xtrans,
                                      yshape, ycshape, ybshape, ytrans,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, f_notrans_notrans_plain_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {1230, 456};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int xtrans = 0;

    int64_t yshape[] = {456, 534};
    int64_t ycshape[] = {200, 210};
    int64_t ybshape[] = {23, 11};

    int ytrans = 0;

    int64_t zshape[] = {1230, 534};
    int64_t zcshape[] = {1230, 124};
    int64_t zbshape[] = {13, 21};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xtrans,
                                      yshape, ycshape, ybshape, ytrans,
                                      zshape, zcshape, zbshape));
}

/*
INA_TEST_FIXTURE(linalg_gemm, d_trans_notrans_plain_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {1230, 456};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int64_t *xblockshape = NULL;
    int xtrans = 1;

    int64_t yshape[] = {1230, 534};
    int64_t ycshape[] = {1230, 210};
    int64_t ybshape[] = {55, 11};

    int64_t yblockshape[] = {1230, 210};
    int ytrans = 0;

    int64_t zshape[] = {456, 534};
    int64_t zcshape[] = {456, 210};
    int64_t zbshape[] = {45, 21};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape, xtrans,
                                      yshape, ycshape, ybshape, yblockshape, ytrans,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, f_trans_trans_schunk_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {433, 555};
    int64_t xcshape[] = {123, 234};
    int64_t xbshape[] = {13, 24};

    int64_t xblockshape[] = {236, 111};
    int xtrans = 1;

    int64_t yshape[] = {678, 433};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int64_t yblockshape[] = {111, 88};
    int ytrans = 1;

    int64_t zshape[] = {555, 678};
    int64_t zcshape[] = {236, 88};
    int64_t zbshape[] = {26, 88};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape, xtrans,
                                      yshape, ycshape, ybshape, yblockshape, ytrans,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, d_trans_notrans_schunk_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {1230, 456};
    int64_t xcshape[] = {231, 124};
    int64_t xbshape[] = {23, 14};

    int64_t xblockshape[] = {123, 1230};
    int xtrans = 1;

    int64_t yshape[] = {1230, 534};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int64_t *yblockshape = NULL;
    int ytrans = 0;

    int64_t zshape[] = {456, 534};
    int64_t zcshape[] = {123, 534};
    int64_t zbshape[] = {13, 5};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape, xtrans,
                                      yshape, ycshape, ybshape, yblockshape, ytrans,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, d_notrans_notrans_schunk_plain_plain) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {456, 1230};
    int64_t xcshape[] = {231, 456};
    int64_t xbshape[] = {21, 6};

    int64_t xblockshape[] = {456, 123};
    int xtrans = 0;

    int64_t yshape[] = {1230, 534};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int64_t yblockshape[] = {123, 534};
    int ytrans = 0;

    int64_t zshape[] = {456, 534};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape, xtrans,
                                      yshape, ycshape, ybshape, yblockshape, ytrans,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, f_notrans_notrans_plain_plain_nc_nc) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {150, 250};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int64_t xblockshape[] = {150, 30};
    int xtrans = 0;

    int64_t yshape[] = {250, 100};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int64_t yblockshape[] = {30, 100};
    int ytrans = 0;

    int64_t zshape[] = {150, 100};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape, xtrans,
                                      yshape, ycshape, ybshape, yblockshape, ytrans,
                                      zshape, zcshape, zbshape));
}

INA_TEST_FIXTURE(linalg_gemm, f_trans_trans_plain_plain_nc_nc) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {100, 250};
    int64_t *xcshape = NULL;
    int64_t *xbshape = NULL;

    int64_t xblockshape[] = {250, 30};
    int xtrans = 1;

    int64_t yshape[] = {250, 100};
    int64_t *ycshape = NULL;
    int64_t *ybshape = NULL;

    int64_t yblockshape[] = {30, 250};
    int ytrans = 1;

    int64_t zshape[] = {250, 250};
    int64_t *zcshape = NULL;
    int64_t *zbshape = NULL;

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape, xblockshape, xtrans,
                                      yshape, ycshape, ybshape, yblockshape, ytrans,
                                      zshape, zcshape, zbshape));
}

*/
