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
#include <tests/iarray_test.h>


static ina_rc_t test_orthogonal_selection(iarray_context_t *ctx,
                                          iarray_data_type_t dtype,
                                          int64_t type_size,
                                          int8_t ndim,
                                          const int64_t *shape,
                                          const int64_t *cshape,
                                          const int64_t *bshape,
                                          int64_t **selection,
                                          int64_t *selection_size,
                                          bool contiguous, char *urlpath) {

    iarray_dtshape_t xdtshape;

    xdtshape.dtype = dtype;
    xdtshape.ndim = ndim;
    for (int j = 0; j < xdtshape.ndim; ++j) {
        xdtshape.shape[j] = shape[j];
    }

    iarray_storage_t store;
    store.contiguous = contiguous;
    store.urlpath = urlpath;
    for (int j = 0; j < xdtshape.ndim; ++j) {
        store.chunkshape[j] = cshape[j];
        store.blockshape[j] = bshape[j];
    }

    blosc2_remove_urlpath(store.urlpath);

    iarray_container_t *c_x;

    INA_TEST_ASSERT_SUCCEED(iarray_linspace(ctx, &xdtshape,
                                            0., 1.,
                                               &store, &c_x));

    int64_t buffer_size = 1;
    for (int i = 0; i < ndim; ++i) {
        buffer_size *= selection_size[i];
    }
    uint8_t *buffer = calloc(buffer_size, type_size);

    INA_TEST_ASSERT_SUCCEED(iarray_set_orthogonal_selection(ctx, c_x, selection, selection_size, buffer,
                                                             selection_size, buffer_size * type_size));
    INA_TEST_ASSERT_SUCCEED(iarray_get_orthogonal_selection(ctx, c_x, selection, selection_size, buffer,
                                                            selection_size, buffer_size * type_size));
    for (int i = 0; i < buffer_size * type_size; ++i) {
        INA_ASSERT_EQUAL(0, buffer[i]);
    }
    free(buffer);
    iarray_container_free(ctx, &c_x);
    blosc2_remove_urlpath(store.urlpath);

    return INA_SUCCESS;
}

INA_TEST_DATA(orthogonal_selection) {
    iarray_context_t *ctx;
    int64_t **selection;
    int64_t *selection_size;
};

INA_TEST_SETUP(orthogonal_selection) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    cfg.compression_codec = IARRAY_COMPRESSION_LZ4;
    cfg.eval_method = IARRAY_EVAL_METHOD_ITERCHUNK;

    iarray_context_new(&cfg, &data->ctx);

    data->selection = malloc(IARRAY_DIMENSION_MAX * sizeof(int64_t *));
    data->selection_size = malloc(IARRAY_DIMENSION_MAX * sizeof(int64_t));
    for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
        data->selection_size[i] = rand() % 10;
        data->selection[i] = malloc(data->selection_size[i] * sizeof(int64_t));
        for (int j = 0; j < data->selection_size[i]; ++j) {
            data->selection[i][j] = rand() % 20;
        }
    }
}

INA_TEST_TEARDOWN(orthogonal_selection) {
    free(data->selection_size);
    for (int i = 0; i < IARRAY_DIMENSION_MAX; ++i) {
        free(data->selection[i]);
    }
    free(data->selection);
    iarray_context_free(&data->ctx);
    iarray_destroy();
}
INA_TEST_FIXTURE(orthogonal_selection, 2_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int32_t type_size = sizeof(double);

    const int8_t ndim = 2;
    int64_t shape[] = {100, 100};
    int64_t cshape[] = {25, 25};
    int64_t bshape[] = {10, 5};

    INA_TEST_ASSERT_SUCCEED(test_orthogonal_selection(data->ctx, dtype, type_size, ndim,
                                                      shape, cshape, bshape,
                                                      data->selection,
                                                      data->selection_size,
                                                      false, NULL));
}


INA_TEST_FIXTURE(orthogonal_selection, 4_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;
    int32_t type_size = sizeof(float);

    const int8_t ndim = 4;
    int64_t shape[] = {40, 100, 30, 30};
    int64_t cshape[] = {7, 10, 5, 5};
    int64_t bshape[] = {2, 2, 2, 2};

    INA_TEST_ASSERT_SUCCEED(test_orthogonal_selection(data->ctx, dtype, type_size, ndim,
                                                      shape, cshape, bshape,
                                                      data->selection,
                                                      data->selection_size,
                                                      false, NULL));
}


INA_TEST_FIXTURE(orthogonal_selection, 3_i16) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;
    int32_t type_size = sizeof(int16_t);

    const int8_t ndim = 3;
    int64_t shape[] = {40, 30, 30};
    int64_t cshape[] = {40, 5, 5};
    int64_t bshape[] = {2, 2, 5};

    INA_TEST_ASSERT_SUCCEED(test_orthogonal_selection(data->ctx, dtype, type_size, ndim,
                                                      shape, cshape, bshape,
                                                      data->selection,
                                                      data->selection_size,
                                                      true, NULL));
}