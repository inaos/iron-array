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

static ina_rc_t
test_btune_favor(iarray_config_t *cfg, iarray_data_type_t dtype, size_t type_size, int8_t ndim,
                 const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                 int32_t *prev_cbytes)
{

    iarray_context_t *ctx;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(cfg, &ctx));

    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
    }

    int64_t buf_size = 1;
    for (int j = 0; j < ndim; ++j) {
        buf_size *= shape[j];
    }
    uint8_t *buf_src = malloc((size_t)buf_size * type_size);

    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        double *buff = (double *) buf_src;
        for (int64_t i = 0; i < buf_size; ++i) {
            buff[i] = (double) i;
        }
    } else {
        float *buff = (float *) buf_src;
        for (int64_t i = 0; i < buf_size; ++i) {
            buff[i] = (float) i;
        }
    }

    iarray_storage_t xstore = {.urlpath=NULL, .enforce_frame=false};
    if (cshape == NULL) {
        xstore.backend = IARRAY_STORAGE_PLAINBUFFER;
    } else {
        xstore.backend = IARRAY_STORAGE_BLOSC;
        for (int i = 0; i < ndim; ++i) {
            xstore.chunkshape[i] = cshape[i];
            xstore.blockshape[i] = bshape[i];
        }
    }

    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buf_src, (size_t) buf_size * type_size, &xstore, 0, &c_x));

    INA_TEST_ASSERT(*prev_cbytes < c_x->catarr->sc->cbytes);

    uint8_t *buf_dest = malloc((size_t)buf_size * type_size);

    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, buf_dest, (size_t)buf_size * type_size));

    if (dtype == IARRAY_DATA_TYPE_DOUBLE) {
        double *buff = (double *) buf_dest;
        for (int64_t i = 0; i < buf_size; ++i) {
            INA_TEST_ASSERT_EQUAL_FLOATING(buff[i], (double) i);
        }
    } else {
        float *buff = (float *) buf_dest;
        for (int64_t i = 0; i < buf_size; ++i) {
            INA_TEST_ASSERT_EQUAL_FLOATING(buff[i], (float) i);
        }
    }

    free(buf_dest);
    free(buf_src);
    iarray_container_free(ctx, &c_x);

    iarray_context_free(&ctx);
    
    return INA_SUCCESS;
}

INA_TEST_DATA(btune_favor) {
    iarray_config_t cfg;
    int32_t cbytes;
};

INA_TEST_SETUP(btune_favor)
{
    iarray_init();
    data->cbytes = 0;
    data->cfg = IARRAY_CONFIG_DEFAULTS;
}

INA_TEST_TEARDOWN(btune_favor)
{
    iarray_destroy();
}

INA_TEST_FIXTURE(btune_favor, cratio)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {367, 333};
    int64_t cshape[] = {70, 91};
    int64_t bshape[] = {12, 25};
    
    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_CRATIO;
    
    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes));
}

INA_TEST_FIXTURE(btune_favor, balance)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {367, 333};
    int64_t cshape[] = {70, 91};
    int64_t bshape[] = {12, 25};

    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_BALANCE;

    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes));
}

INA_TEST_FIXTURE(btune_favor, speed)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {367, 333};
    int64_t cshape[] = {70, 91};
    int64_t bshape[] = {12, 25};

    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_SPEED;

    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes));
}
