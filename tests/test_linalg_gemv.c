/*
 * Copyright ironArray SL 2021.
 *
 * All rights reserved.
 *
 * This software is the confidential and proprietary information of ironArray SL
 * ("Confidential Information"). You shall not disclose such Confidential
 * Information and shall use it only in accordance with the terms of the license agreement.
 *
 */

#include "src/iarray_private.h"
#include <libiarray/iarray.h>


static ina_rc_t test_gemv(iarray_context_t *ctx, iarray_data_type_t dtype, int typesize,
                          const int64_t *xshape, const int64_t *xcshape, const int64_t *xbshape,
                          const int64_t *yshape, const int64_t *ycshape, const int64_t *ybshape,
                          const int64_t *zshape, const int64_t *zcshape, const int64_t *zbshape,
                          bool xcontiguous, char *xurlpath, bool ycontiguous, char *yurlpath, bool zcontiguous, char *zurlpath)
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
    xstore.urlpath = xurlpath;
    xstore.contiguous = xcontiguous;
    for (int i = 0; i < xdtshape.ndim; ++i) {
        xstore.chunkshape[i] = xcshape[i];
        xstore.blockshape[i] = xbshape[i];
    }
    blosc2_remove_urlpath(xstore.urlpath);
    iarray_container_t *c_x;
    INA_TEST_ASSERT_SUCCEED(iarray_linspace(ctx, &xdtshape, 0, 10, &xstore, &c_x));

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
    ystore.urlpath = yurlpath;
    ystore.contiguous = ycontiguous;
    for (int i = 0; i < ydtshape.ndim; ++i) {
        ystore.chunkshape[i] = ycshape[i];
        ystore.blockshape[i] = ybshape[i];
    }
    blosc2_remove_urlpath(ystore.urlpath);
    iarray_container_t *c_y;
    INA_TEST_ASSERT_SUCCEED(iarray_linspace(ctx, &ydtshape, 0, 10, &ystore, &c_y));

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
    zstore.urlpath = zurlpath;
    zstore.contiguous = zcontiguous;
    for (int i = 0; i < zdtshape.ndim; ++i) {
        zstore.chunkshape[i] = zcshape[i];
        zstore.blockshape[i] = zbshape[i];
    }
    blosc2_remove_urlpath(zstore.urlpath);
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
    blosc2_remove_urlpath(xstore.urlpath);
    blosc2_remove_urlpath(ystore.urlpath);
    blosc2_remove_urlpath(zstore.urlpath);

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

INA_TEST_FIXTURE(linalg_gemv, d_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {100, 200};
    int64_t xcshape[] = {30, 30};
    int64_t xbshape[] = {10, 15};

    int64_t yshape[] = {200};
    int64_t ycshape[] = {125};
    int64_t ybshape[] = {45};

    int64_t zshape[] = {100};
    int64_t zcshape[] = {80};
    int64_t zbshape[] = {37};

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape,
                                      yshape, ycshape, ybshape,
                                      zshape, zcshape, zbshape,
                                      false, "xarr.iarr", false, NULL, true, NULL));
}

INA_TEST_FIXTURE(linalg_gemv, f_schunk_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {650, 900};
    int64_t xcshape[] = {200, 140};
    int64_t xbshape[] = {20, 14};

    int64_t yshape[] = {900};
    int64_t ycshape[] = {421};
    int64_t ybshape[] = {55};

    int64_t zshape[] = {650};
    int64_t zcshape[] = {155};
    int64_t zbshape[] = {47};

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape,
                                      yshape, ycshape, ybshape,
                                      zshape, zcshape, zbshape,
                                      true, NULL, true, "yarr.iarr", false, "zarr.iarr"));
}

INA_TEST_FIXTURE(linalg_gemv, d_schunk_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int typesize = sizeof(double);

    int64_t xshape[] = {60, 300};
    int64_t xcshape[] = {14, 200};
    int64_t xbshape[] = {5, 14};

    int64_t yshape[] = {300};
    int64_t ycshape[] = {41};
    int64_t ybshape[] = {17};

    int64_t zshape[] = {60};
    int64_t zcshape[] = {18};
    int64_t zbshape[] = {18};

    INA_TEST_ASSERT_SUCCEED(test_gemv(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape,
                                      yshape, ycshape, ybshape,
                                      zshape, zcshape, zbshape,
                                      true, NULL, false, NULL, true, "zarr.iarr"));
}
