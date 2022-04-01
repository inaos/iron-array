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

#include <src/iarray_private.h>
#include <libiarray/iarray.h>


static ina_rc_t test_constructor_frame(iarray_context_t *ctx, iarray_data_type_t dtype, int8_t ndim,
                                const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                                double start, double stop, bool contiguous, char *urlpath)
{

    // Create dtshape
    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    int64_t size = 1;
    for (int i = 0; i < ndim; ++i) {
        xdtshape.shape[i] = shape[i];
        size *= shape[i];
    }

    double step = (stop - start) / size;

    iarray_storage_t xstore = {.urlpath=urlpath, .contiguous=contiguous};

    for (int i = 0; i < ndim; ++i) {
        xstore.chunkshape[i] = cshape[i];
        xstore.blockshape[i] = bshape[i];
    }

    iarray_container_t *c_x;
    blosc2_remove_urlpath(xstore.urlpath);

    INA_TEST_ASSERT_SUCCEED(iarray_arange(ctx, &xdtshape, start, step, &xstore, &c_x));

    // Assert iterator reading it

    iarray_iter_read_t *I2;
    iarray_iter_read_value_t val;
    INA_TEST_ASSERT_SUCCEED(iarray_iter_read_new(ctx, &I2, c_x, &val));

    while (INA_SUCCEED(iarray_iter_read_has_next(I2))) {
        INA_TEST_ASSERT_SUCCEED(iarray_iter_read_next(I2));

        switch(dtype) {
            case IARRAY_DATA_TYPE_DOUBLE:
                INA_TEST_ASSERT_EQUAL_FLOATING(val.elem_flat_index * step + start, ((double *) val.elem_pointer)[0]);
                break;
            case IARRAY_DATA_TYPE_FLOAT:
                INA_TEST_ASSERT_EQUAL_FLOATING( (float) (val.elem_flat_index * step + start), ((float *) val.elem_pointer)[0]);
                break;
            case IARRAY_DATA_TYPE_INT64:
                INA_TEST_ASSERT_EQUAL_INT64( (int64_t) (val.elem_flat_index * step + start), ((int64_t *) val.elem_pointer)[0]);
                break;
            case IARRAY_DATA_TYPE_INT32:
                INA_TEST_ASSERT_EQUAL_INT( (int32_t) (val.elem_flat_index * step + start), ((int32_t *) val.elem_pointer)[0]);
                break;
            case IARRAY_DATA_TYPE_INT16:
                INA_TEST_ASSERT_EQUAL_INT( (int16_t) (val.elem_flat_index * step + start), ((int16_t *) val.elem_pointer)[0]);
                break;
            case IARRAY_DATA_TYPE_INT8:
                INA_TEST_ASSERT_EQUAL_INT( (int8_t) (val.elem_flat_index * step + start), ((int8_t *) val.elem_pointer)[0]);
                break;
            case IARRAY_DATA_TYPE_UINT64:
                INA_TEST_ASSERT_EQUAL_UINT64( (uint64_t) (val.elem_flat_index * step + start), ((uint64_t *) val.elem_pointer)[0]);
                break;
            case IARRAY_DATA_TYPE_UINT32:
                INA_TEST_ASSERT_EQUAL_UINT( (uint32_t) (val.elem_flat_index * step + start), ((uint32_t *) val.elem_pointer)[0]);
                break;
            case IARRAY_DATA_TYPE_UINT16:
                INA_TEST_ASSERT_EQUAL_UINT( (uint16_t) (val.elem_flat_index * step + start), ((uint16_t *) val.elem_pointer)[0]);
                break;
            case IARRAY_DATA_TYPE_UINT8:
                INA_TEST_ASSERT_EQUAL_UINT( (uint8_t) (val.elem_flat_index * step + start), ((uint8_t *) val.elem_pointer)[0]);
                break;
            default:
                return INA_ERR_EXCEEDED;
        }
    }

    iarray_iter_read_free(&I2);
    INA_TEST_ASSERT(ina_err_get_rc() == INA_RC_PACK(IARRAY_ERR_END_ITER, 0));
    blosc2_remove_urlpath(xstore.urlpath);

    iarray_container_free(ctx, &c_x);
    return INA_SUCCESS;
}

INA_TEST_DATA(constructor_frame) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(constructor_frame) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(constructor_frame) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}

INA_TEST_FIXTURE(constructor_frame, 2_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 2;
    int64_t shape[] = {100, 100};
    int64_t cshape[] = {44, 6};
    int64_t bshape[] = {23, 3};
    double start = - 0.1;
    double stop = - 0.25;

    INA_TEST_ASSERT_SUCCEED(test_constructor_frame(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, false, NULL));
}

INA_TEST_FIXTURE(constructor_frame, 2_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 2;
    int64_t shape[] = {445, 321};
    int64_t cshape[] = {21, 221};
    int64_t bshape[] = {15, 13};
    double start = 3123;
    double stop = 45654;

    INA_TEST_ASSERT_SUCCEED(test_constructor_frame(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, false, "arr.iarr"));
}

INA_TEST_FIXTURE(constructor_frame, 3_ll) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;

    int8_t ndim = 3;
    int64_t shape[] = {17, 13, 21};
    int64_t cshape[] = {14, 3, 20};
    int64_t bshape[] = {3, 2, 3};
    double start = 1;
    double stop = 20 * 18 * 17 * 13 * 21 + 1;

    INA_TEST_ASSERT_SUCCEED(test_constructor_frame(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, true, "arr.iarr"));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(constructor_frame, 5_ll) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;

    int8_t ndim = 5;
    int64_t shape[] = {20, 18, 17, 13, 21};
    int64_t cshape[] = {3, 12, 14, 3, 20};
    int64_t bshape[] = {3, 5, 3, 2, 3};
    double start = 1;
    double stop = 20 * 18 * 17 * 13 * 21 + 1;

    INA_TEST_ASSERT_SUCCEED(test_constructor_frame(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, true, "arr.iarr"));
}
*/

INA_TEST_FIXTURE(constructor_frame, 1_i) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;

    int8_t ndim = 1;
    int64_t shape[] = {5};
    int64_t cshape[] = {2};
    int64_t bshape[] = {2};
    double start = 10;
    double stop = 5 * 7 * 8 * 9 * 6 * 5 * 7 + 10 + 1;

    INA_TEST_ASSERT_SUCCEED(test_constructor_frame(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, true, NULL));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(constructor_frame, 7_i) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;

    int8_t ndim = 7;
    int64_t shape[] = {5, 7, 8, 9, 6, 5, 7};
    int64_t cshape[] = {2, 2, 2, 2, 2, 2, 2};
    int64_t bshape[] = {2, 2, 1, 2, 1, 2, 2};
    double start = 10;
    double stop = 5 * 7 * 8 * 9 * 6 * 5 * 7 + 10 + 1;

    INA_TEST_ASSERT_SUCCEED(test_constructor_frame(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, true, NULL));
}
*/

INA_TEST_FIXTURE(constructor_frame, 2_s) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;

    int8_t ndim = 2;
    int64_t shape[] = {100, 100};
    int64_t cshape[] = {44, 6};
    int64_t bshape[] = {23, 3};
    double start = -1;
    double stop = -100 * 100 - 1;

    INA_TEST_ASSERT_SUCCEED(test_constructor_frame(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, false, NULL));
}

INA_TEST_FIXTURE(constructor_frame, 1_sc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;

    int8_t ndim = 1;
    int64_t shape[] = {100};
    int64_t cshape[] = {20};
    int64_t bshape[] = {10};
    double start = 3;
    double stop = 100 + 3 + 1;

    INA_TEST_ASSERT_SUCCEED(test_constructor_frame(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, false, "arr.iarr"));
}

INA_TEST_FIXTURE(constructor_frame, 2_ull) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;

    int8_t ndim = 2;
    int64_t shape[] = {20, 18};
    int64_t cshape[] = {3, 12};
    int64_t bshape[] = {3, 5};
    double start = 1;
    double stop = 20 + 18 * 17 * 13 * 21 * 4 + 1;

    INA_TEST_ASSERT_SUCCEED(test_constructor_frame(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, true, "arr.iarr"));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(constructor_frame, 5_ull) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;

    int8_t ndim = 5;
    int64_t shape[] = {20, 18, 17, 13, 21};
    int64_t cshape[] = {3, 12, 14, 3, 20};
    int64_t bshape[] = {3, 5, 3, 2, 3};
    double start = 1;
    double stop = 20 + 18 * 17 * 13 * 21 * 4 + 1;

    INA_TEST_ASSERT_SUCCEED(test_constructor_frame(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, true, "arr.iarr"));
}
*/

INA_TEST_FIXTURE(constructor_frame, 2_ui) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;

    int8_t ndim = 2;
    int64_t shape[] = {5, 7};
    int64_t cshape[] = {2, 2};
    int64_t bshape[] = {2, 2};
    double start = 5 * 7 * 8 * 9 * 6 * 5 * 7;
    double stop = 0;

    INA_TEST_ASSERT_SUCCEED(test_constructor_frame(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, true, NULL));
}

/* Avoid heavy tests
INA_TEST_FIXTURE(constructor_frame, 7_ui) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;

    int8_t ndim = 7;
    int64_t shape[] = {5, 7, 8, 9, 6, 5, 7};
    int64_t cshape[] = {2, 2, 2, 2, 2, 2, 2};
    int64_t bshape[] = {2, 2, 1, 2, 1, 2, 2};
    double start = 5 * 7 * 8 * 9 * 6 * 5 * 7;
    double stop = 0;

    INA_TEST_ASSERT_SUCCEED(test_constructor_frame(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, true, NULL));
}
*/

INA_TEST_FIXTURE(constructor_frame, 2_us) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;

    int8_t ndim = 2;
    int64_t shape[] = {445, 321};
    int64_t cshape[] = {21, 221};
    int64_t bshape[] = {15, 13};
    double start = 3123;
    double stop = 445 * 321 + 3123 + 1;

    INA_TEST_ASSERT_SUCCEED(test_constructor_frame(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, false, NULL));
}

INA_TEST_FIXTURE(constructor_frame, 1uc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;

    int8_t ndim = 1;
    int64_t shape[] = {100};
    int64_t cshape[] = {44};
    int64_t bshape[] = {23};
    double start = 1;
    double stop = 100 + 1;

    INA_TEST_ASSERT_SUCCEED(test_constructor_frame(data->ctx, dtype, ndim, shape, cshape, bshape, start, stop, false, "arr.iarr"));
}
