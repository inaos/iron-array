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
                 int32_t *prev_cbytes, bool contiguous, char *urlpath)
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

    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE: {
            double *buff = (double *) buf_src;
            for (int64_t i = 0; i < buf_size; ++i) {
                buff[i] = (double) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_FLOAT: {
            float *buff = (float *) buf_src;
            for (int64_t i = 0; i < buf_size; ++i) {
                buff[i] = (float) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_INT64: {
            int64_t *buff = (int64_t *) buf_src;
            for (int64_t i = 0; i < buf_size; ++i) {
                buff[i] = (int64_t) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_INT32: {
            int32_t *buff = (int32_t *) buf_src;
            for (int64_t i = 0; i < buf_size; ++i) {
                buff[i] = (int32_t) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_INT16: {
            int16_t *buff = (int16_t *) buf_src;
            for (int64_t i = 0; i < buf_size; ++i) {
                buff[i] = (int16_t) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_INT8: {
            int8_t *buff = (int8_t *) buf_src;
            for (int64_t i = 0; i < buf_size; ++i) {
                buff[i] = (int8_t) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_UINT64: {
            uint64_t *buff = (uint64_t *) buf_src;
            for (int64_t i = 0; i < buf_size; ++i) {
                buff[i] = (uint64_t) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_UINT32: {
            uint32_t *buff = (uint32_t *) buf_src;
            for (int64_t i = 0; i < buf_size; ++i) {
                buff[i] = (uint32_t) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_UINT16: {
            uint16_t *buff = (uint16_t *) buf_src;
            for (int64_t i = 0; i < buf_size; ++i) {
                buff[i] = (uint16_t) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_UINT8: {
            uint8_t *buff = (uint8_t *) buf_src;
            for (int64_t i = 0; i < buf_size; ++i) {
                buff[i] = (uint8_t) i;
            }
            break;
        }
        case IARRAY_DATA_TYPE_BOOL: {
            bool *buff = (bool *) buf_src;
            for (int64_t i = 0; i < buf_size; ++i) {
                buff[i] = (bool) i;
            }
            break;
        }
    }

    iarray_storage_t xstore = {.urlpath=urlpath, .contiguous=contiguous};

    for (int i = 0; i < ndim; ++i) {
        xstore.chunkshape[i] = cshape[i];
        xstore.blockshape[i] = bshape[i];
    }

    iarray_container_t *c_x;
    blosc2_remove_urlpath(xstore.urlpath);

    INA_TEST_ASSERT_SUCCEED(iarray_from_buffer(ctx, &xdtshape, buf_src,
                                               (size_t) buf_size * type_size, &xstore, &c_x));

    INA_TEST_ASSERT(*prev_cbytes < c_x->catarr->sc->cbytes);

    uint8_t *buf_dest = malloc((size_t)buf_size * type_size);

    INA_TEST_ASSERT_SUCCEED(iarray_to_buffer(ctx, c_x, buf_dest, (size_t)buf_size * type_size));

    switch (dtype) {
        case IARRAY_DATA_TYPE_DOUBLE: {
            double *buff = (double *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_FLOATING(buff[i], (double) i);
            }
            break;
        }
        case IARRAY_DATA_TYPE_FLOAT: {
            float *buff = (float *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_FLOATING(buff[i], (float) i);
            }
            break;
        }
        case IARRAY_DATA_TYPE_INT64: {
            int64_t *buff = (int64_t *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_INT64(buff[i], (int64_t) i);
            }
            break;
        }
        case IARRAY_DATA_TYPE_INT32: {
            int32_t *buff = (int32_t *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_INT(buff[i], (int32_t) i);
            }
            break;
        }
        case IARRAY_DATA_TYPE_INT16: {
            int16_t *buff = (int16_t *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_INT(buff[i], (int16_t) i);
            }
            break;
        }
        case IARRAY_DATA_TYPE_INT8: {
            int8_t *buff = (int8_t *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_INT(buff[i], (int8_t) i);
            }
            break;
        }
        case IARRAY_DATA_TYPE_UINT64: {
            uint64_t *buff = (uint64_t *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_UINT64(buff[i], (uint64_t) i);
            }
            break;
        }
        case IARRAY_DATA_TYPE_UINT32: {
            uint32_t *buff = (uint32_t *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_UINT(buff[i], (uint32_t) i);
            }
            break;
        }
        case IARRAY_DATA_TYPE_UINT16: {
            uint16_t *buff = (uint16_t *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_UINT(buff[i], (uint16_t) i);
            }
            break;
        }
        case IARRAY_DATA_TYPE_UINT8: {
            uint8_t *buff = (uint8_t *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT_EQUAL_UINT(buff[i], (uint8_t) i);
            }
            break;
        }
        case IARRAY_DATA_TYPE_BOOL: {
            bool *buff = (bool *) buf_dest;
            for (int64_t i = 0; i < buf_size; ++i) {
                INA_TEST_ASSERT(buff[i] == (bool) i);
            }
            break;
        }
    }

    free(buf_dest);
    free(buf_src);
    iarray_container_free(ctx, &c_x);
    blosc2_remove_urlpath(xstore.urlpath);

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

INA_TEST_FIXTURE(btune_favor, d_cratio)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    size_t type_size = sizeof(double);

    int8_t ndim = 2;
    int64_t shape[] = {367, 333};
    int64_t cshape[] = {70, 91};
    int64_t bshape[] = {12, 25};
    
    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_CRATIO;
    
    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes, false, NULL));
}

INA_TEST_FIXTURE(btune_favor, f_balance)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    size_t type_size = sizeof(float);

    int8_t ndim = 2;
    int64_t shape[] = {367, 333};
    int64_t cshape[] = {70, 91};
    int64_t bshape[] = {12, 25};

    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_BALANCE;

    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes, true, NULL));
}

INA_TEST_FIXTURE(btune_favor, ll_speed)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;
    size_t type_size = sizeof(int64_t);

    int8_t ndim = 2;
    int64_t shape[] = {367, 333};
    int64_t cshape[] = {70, 91};
    int64_t bshape[] = {12, 25};

    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_SPEED;

    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes, true, "arr.iarr"));

    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes, false, "arr.iarr"));
}


INA_TEST_FIXTURE(btune_favor, i_speed)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;
    size_t type_size = sizeof(int32_t);

    int8_t ndim = 2;
    int64_t shape[] = {367, 333};
    int64_t cshape[] = {70, 91};
    int64_t bshape[] = {12, 25};

    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_SPEED;

    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes, true, "arr.iarr"));

    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes, false, "arr.iarr"));
}

INA_TEST_FIXTURE(btune_favor, s_cratio)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    size_t type_size = sizeof(int16_t);

    int8_t ndim = 2;
    int64_t shape[] = {367, 333};
    int64_t cshape[] = {70, 91};
    int64_t bshape[] = {12, 25};

    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_CRATIO;

    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes, false, NULL));
}

INA_TEST_FIXTURE(btune_favor, sc_balance)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;
    size_t type_size = sizeof(int8_t);

    int8_t ndim = 2;
    int64_t shape[] = {367, 333};
    int64_t cshape[] = {70, 91};
    int64_t bshape[] = {12, 25};

    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_BALANCE;

    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes, true, NULL));
}

INA_TEST_FIXTURE(btune_favor, ull_speed)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;
    size_t type_size = sizeof(uint64_t);

    int8_t ndim = 2;
    int64_t shape[] = {367, 333};
    int64_t cshape[] = {70, 91};
    int64_t bshape[] = {12, 25};

    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_SPEED;

    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes, true, "arr.iarr"));

    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes, false, "arr.iarr"));
}

INA_TEST_FIXTURE(btune_favor, ui_cratio)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;
    size_t type_size = sizeof(uint32_t);

    int8_t ndim = 2;
    int64_t shape[] = {367, 333};
    int64_t cshape[] = {70, 91};
    int64_t bshape[] = {12, 25};

    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_CRATIO;

    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes, false, NULL));
}

INA_TEST_FIXTURE(btune_favor, us_balance)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;
    size_t type_size = sizeof(uint16_t);

    int8_t ndim = 2;
    int64_t shape[] = {367, 333};
    int64_t cshape[] = {70, 91};
    int64_t bshape[] = {12, 25};

    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_BALANCE;

    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes, true, NULL));
}

INA_TEST_FIXTURE(btune_favor, uc_speed)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;
    size_t type_size = sizeof(uint8_t);

    int8_t ndim = 2;
    int64_t shape[] = {367, 333};
    int64_t cshape[] = {70, 91};
    int64_t bshape[] = {12, 25};

    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_SPEED;

    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes, true, "arr.iarr"));

    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes, false, "arr.iarr"));
}

INA_TEST_FIXTURE(btune_favor, b_cratio)
{
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_BOOL;
    size_t type_size = sizeof(bool);

    int8_t ndim = 2;
    int64_t shape[] = {367, 333};
    int64_t cshape[] = {70, 91};
    int64_t bshape[] = {12, 25};

    data->cfg.compression_favor = IARRAY_COMPRESSION_FAVOR_CRATIO;

    INA_TEST_ASSERT_SUCCEED(test_btune_favor(&data->cfg, dtype, type_size, ndim, shape, cshape,
                                             bshape, &data->cbytes, false, NULL));
}
