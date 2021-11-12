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
test_gemm(iarray_context_t *ctx, iarray_data_type_t dtype, int typesize, const int64_t *xshape,
          const int64_t *xcshape, const int64_t *xbshape, const int64_t *yshape,
          const int64_t *ycshape, const int64_t *ybshape, const int64_t *zshape,
          const int64_t *zcshape, const int64_t *zbshape, bool xcontiguous, char *xurlpath, bool ycontiguous, char *yurlpath,
          bool zcontiguous, char *zurlpath)
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
    xstore.urlpath = xurlpath;
    xstore.contiguous = xcontiguous;
    if (xcshape != NULL) {
        for (int i = 0; i < xdtshape.ndim; ++i) {
            xstore.chunkshape[i] = xcshape[i];
            xstore.blockshape[i] = xbshape[i];
        }
    }
    blosc2_remove_urlpath(xstore.urlpath);
    
    iarray_container_t *c_x;
    INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &xdtshape, 0, (double) xsize, 1, &xstore, 0, &c_x));
    iarray_container_t *c_xtrans;
    INA_TEST_ASSERT_SUCCEED(iarray_linalg_transpose(ctx, c_x, &c_xtrans));
    
    // iarray container x to buffer
    uint8_t *xbuffer = ina_mem_alloc(xsize * typesize);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_xtrans, xbuffer, xsize * typesize));

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
    ystore.urlpath = yurlpath;
    ystore.contiguous = ycontiguous;
    for (int i = 0; i < ydtshape.ndim; ++i) {
        ystore.chunkshape[i] = ycshape[i];
        ystore.blockshape[i] = ybshape[i];
    }
    blosc2_remove_urlpath(ystore.urlpath);
    
    iarray_container_t *c_y;
    INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &ydtshape, 0, (double) ysize, 1, &ystore, 0, &c_y));
    iarray_container_t *c_ytrans;
    INA_TEST_ASSERT_SUCCEED(iarray_linalg_transpose(ctx, c_y, &c_ytrans));

    // iarray container y to buffer
    uint8_t *ybuffer = ina_mem_alloc(ysize * typesize);
    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_ytrans, ybuffer, ysize * typesize));

    // define o buffer
    int64_t osize = c_x->dtshape->shape[0] * c_y->dtshape->shape[1];
    uint8_t *obuffer = ina_mem_alloc((size_t)osize * typesize);

    // MKL matrix-matrix multiplication
    int M = (int) c_xtrans->dtshape->shape[0];
    int K = (int) c_xtrans->dtshape->shape[1];
    int N = (int) c_ytrans->dtshape->shape[1];

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
    zstore.urlpath = zurlpath;
    zstore.contiguous = zcontiguous;
    for (int i = 0; i < zdtshape.ndim; ++i) {
        zstore.chunkshape[i] = zcshape[i];
        zstore.blockshape[i] = zbshape[i];
    }
    blosc2_remove_urlpath(zstore.urlpath);

    iarray_container_t *c_z;

    // iarray multiplication
    INA_TEST_ASSERT_SUCCEED(iarray_linalg_matmul(ctx, c_xtrans, c_ytrans, &zstore, &c_z));

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
    iarray_container_free(ctx, &c_xtrans);
    iarray_container_free(ctx, &c_ytrans);
    blosc2_remove_urlpath(xstore.urlpath);
    blosc2_remove_urlpath(ystore.urlpath);
    blosc2_remove_urlpath(zstore.urlpath);
    INA_MEM_FREE_SAFE(xbuffer);
    INA_MEM_FREE_SAFE(ybuffer);
    INA_MEM_FREE_SAFE(obuffer);
    INA_MEM_FREE_SAFE(zbuffer);

    return INA_SUCCESS;
}

INA_TEST_DATA(linalg_gemm_transpose) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(linalg_gemm_transpose) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(linalg_gemm_transpose) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(linalg_gemm_transpose, f_schunk_schunk) {

    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int typesize = sizeof(float);

    int64_t xshape[] = {170, 100};
    int64_t xcshape[] = {30, 27};
    int64_t xbshape[] = {7, 12};

    int64_t yshape[] = {21, 170};
    int64_t ycshape[] = {11, 20};
    int64_t ybshape[] = {5, 10};

    int64_t zshape[] = {100, 21};
    int64_t zcshape[] = {40, 3};
    int64_t zbshape[] = {20, 3};

    INA_TEST_ASSERT_SUCCEED(test_gemm(data->ctx, dtype, typesize,
                                      xshape, xcshape, xbshape,
                                      yshape, ycshape, ybshape,
                                      zshape, zcshape, zbshape,
                                      true, "xarr.iarr", false, NULL, false, "zarr.iarr"));
}
