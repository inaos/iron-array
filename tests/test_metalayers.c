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

static ina_rc_t test_metalayers(iarray_context_t *ctx, iarray_data_type_t dtype, int8_t ndim,
                                const int64_t *shape, const int64_t *cshape, const int64_t *bshape,
                                bool contiguous, char *urlpath)
{
    // Create dtshape
    iarray_dtshape_t dtshape;

    dtshape.dtype = dtype;
    dtshape.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
    }

    iarray_storage_t store = {.urlpath=urlpath, .contiguous=contiguous};

    for (int i = 0; i < ndim; ++i) {
        store.chunkshape[i] = cshape[i];
        store.blockshape[i] = bshape[i];
    }

    iarray_container_t *c;
    blosc2_remove_urlpath(store.urlpath);

    INA_TEST_ASSERT_SUCCEED(iarray_ones(ctx, &dtshape, &store, &c));

    iarray_metalayer_t vlmeta1;
    uint64_t sdata1 = 56;
    vlmeta1.name = "vlmeta1";
    vlmeta1.sdata = (uint8_t *) &sdata1;
    vlmeta1.size = sizeof(sdata1);

    int16_t nvlmetalayers;
    INA_TEST_ASSERT_SUCCEED(iarray_vlmeta_nitems(ctx, c, &nvlmetalayers));
    INA_TEST_ASSERT(nvlmetalayers == 0);

    INA_TEST_ASSERT_SUCCEED(iarray_vlmeta_add(ctx, c, &vlmeta1));
    INA_TEST_ASSERT_SUCCEED(iarray_vlmeta_nitems(ctx, c, &nvlmetalayers));
    INA_TEST_ASSERT(nvlmetalayers == 1);

    char names_[2][8] = {"vlmeta1", "vlmeta2"};
    char **names = malloc(sizeof(char*) * nvlmetalayers);
    INA_TEST_ASSERT_SUCCEED(iarray_vlmeta_get_names(ctx, c, names));
    for (int i = 0; i < nvlmetalayers; ++i) {
        INA_TEST_ASSERT(strcmp(names[i], names_[i]) == 0);
    }

    bool exists;
    INA_TEST_ASSERT_SUCCEED(iarray_vlmeta_exists(ctx, c, "vlmeta2", &exists));
    INA_TEST_ASSERT(exists == false);
    INA_TEST_ASSERT_SUCCEED(iarray_vlmeta_exists(ctx, c, vlmeta1.name, &exists));
    INA_TEST_ASSERT(exists == true);

    iarray_metalayer_t vlmeta2;
    INA_TEST_ASSERT_SUCCEED(iarray_vlmeta_get(ctx, c, vlmeta1.name, &vlmeta2));
    INA_TEST_ASSERT(*((uint64_t *) vlmeta1.sdata) == *((uint64_t *) vlmeta2.sdata));
    INA_TEST_ASSERT(vlmeta1.size == vlmeta2.size);
    free(vlmeta2.name);
    free(vlmeta2.sdata);

    float sdata11 = 4.5f;
    vlmeta1.sdata = (uint8_t *) &sdata11;
    vlmeta1.size = sizeof(sdata11);

    INA_TEST_ASSERT_SUCCEED(iarray_vlmeta_update(ctx, c, &vlmeta1));

    iarray_metalayer_t vlmeta3;
    INA_TEST_ASSERT_SUCCEED(iarray_vlmeta_get(ctx, c, vlmeta1.name, &vlmeta3));
    INA_TEST_ASSERT(*((float *) vlmeta1.sdata) == *((float *) vlmeta3.sdata));
    INA_TEST_ASSERT(vlmeta1.size == vlmeta3.size);
    free(vlmeta3.name);
    free(vlmeta3.sdata);

    vlmeta2.name = "vlmeta2";
    vlmeta2.sdata = (uint8_t *) &sdata1;
    vlmeta2.size = sizeof(sdata1);
    INA_TEST_ASSERT_SUCCEED(iarray_vlmeta_add(ctx, c, &vlmeta2));
    INA_TEST_ASSERT_SUCCEED(iarray_vlmeta_nitems(ctx, c, &nvlmetalayers));

    INA_TEST_ASSERT(nvlmetalayers == 2);
    free(names);
    names = malloc(sizeof(char*) * nvlmetalayers);
    INA_TEST_ASSERT_SUCCEED(iarray_vlmeta_get_names(ctx, c, names));
    for (int i = 0; i < nvlmetalayers; ++i) {
        INA_TEST_ASSERT(strcmp(names[i], names_[i]) == 0);
    }
    INA_TEST_ASSERT_SUCCEED(iarray_vlmeta_delete(ctx, c, vlmeta2.name));
    INA_TEST_ASSERT_SUCCEED(iarray_vlmeta_exists(ctx, c, "vlmeta2", &exists));
    INA_TEST_ASSERT(exists == false);

    INA_TEST_ASSERT_SUCCEED(iarray_vlmeta_add(ctx, c, &vlmeta2));

    iarray_container_free(ctx, &c);

    if (urlpath != NULL) {
        iarray_container_t *c2;
        INA_TEST_ASSERT_SUCCEED(iarray_container_open(ctx, urlpath, &c2));

        INA_TEST_ASSERT_SUCCEED(iarray_vlmeta_get(ctx, c2, vlmeta2.name, &vlmeta3));
        INA_TEST_ASSERT(*((uint64_t *) vlmeta2.sdata) == *((uint64_t *) vlmeta3.sdata));
        INA_TEST_ASSERT(vlmeta2.size == vlmeta3.size);
        free(vlmeta3.name);
        free(vlmeta3.sdata);

        /* Free mallocs */
        iarray_container_free(ctx, &c2);
        blosc2_remove_urlpath(urlpath);
    }

    return 0;
}


INA_TEST_DATA(metalayers) {
    iarray_context_t *ctx;
};

INA_TEST_SETUP(metalayers) {
    iarray_init();

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    INA_TEST_ASSERT_SUCCEED(iarray_context_new(&cfg, &data->ctx));
}

INA_TEST_TEARDOWN(metalayers) {
    iarray_context_free(&data->ctx);
    iarray_destroy();
}


INA_TEST_FIXTURE(metalayers, 0_f) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_FLOAT;

    int8_t ndim = 0;
    int64_t shape[] = {0};
    int64_t cshape[] = {0};
    int64_t bshape[] = {0};

    INA_TEST_ASSERT_SUCCEED(test_metalayers(data->ctx, dtype, ndim, shape, cshape, bshape, true, "arr.iarr"));
}


INA_TEST_FIXTURE(metalayers, 1_d) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;

    int8_t ndim = 1;
    int64_t shape[] = {10};
    int64_t cshape[] = {7};
    int64_t bshape[] = {2};

    INA_TEST_ASSERT_SUCCEED(test_metalayers(data->ctx, dtype, ndim, shape, cshape, bshape, false, "arr.iarr"));
}

INA_TEST_FIXTURE(metalayers, 2_ll) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT64;

    int8_t ndim = 2;
    int64_t shape[] = {100, 100};
    int64_t cshape[] = {20, 20};
    int64_t bshape[] = {10, 10};

    INA_TEST_ASSERT_SUCCEED(test_metalayers(data->ctx, dtype, ndim, shape, cshape, bshape, true, NULL));
}


INA_TEST_FIXTURE(metalayers, 0_i) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT32;

    int8_t ndim = 0;
    int64_t shape[] = {0};
    int64_t cshape[] = {0};
    int64_t bshape[] = {0};

    INA_TEST_ASSERT_SUCCEED(test_metalayers(data->ctx, dtype, ndim, shape, cshape, bshape, false, NULL));
}


INA_TEST_FIXTURE(metalayers, 1_s) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT16;

    int8_t ndim = 1;
    int64_t shape[] = {10};
    int64_t cshape[] = {7};
    int64_t bshape[] = {2};

    INA_TEST_ASSERT_SUCCEED(test_metalayers(data->ctx, dtype, ndim, shape, cshape, bshape, true, "arr.iarr"));
}


INA_TEST_FIXTURE(metalayers, 1_sc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_INT8;

    int8_t ndim = 1;
    int64_t shape[] = {20};
    int64_t cshape[] = {3};
    int64_t bshape[] = {2};

    INA_TEST_ASSERT_SUCCEED(test_metalayers(data->ctx, dtype, ndim, shape, cshape, bshape, false, "arr.iarr"));
}


INA_TEST_FIXTURE(metalayers, 2_ull) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT64;

    int8_t ndim = 2;
    int64_t shape[] = {100, 100};
    int64_t cshape[] = {60, 50};
    int64_t bshape[] = {17, 13};

    INA_TEST_ASSERT_SUCCEED(test_metalayers(data->ctx, dtype, ndim, shape, cshape, bshape, true, NULL));
}


INA_TEST_FIXTURE(metalayers, 2_ui) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT32;

    int8_t ndim = 2;
    int64_t shape[] = {100, 100};
    int64_t cshape[] = {20, 20};
    int64_t bshape[] = {10, 10};

    INA_TEST_ASSERT_SUCCEED(test_metalayers(data->ctx, dtype, ndim, shape, cshape, bshape, false, "arr.iarr"));
}


INA_TEST_FIXTURE(metalayers, 2_us) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT16;

    int8_t ndim = 2;
    int64_t shape[] = {100, 100};
    int64_t cshape[] = {60, 50};
    int64_t bshape[] = {17, 13};

    INA_TEST_ASSERT_SUCCEED(test_metalayers(data->ctx, dtype, ndim, shape, cshape, bshape, true, NULL));
}


INA_TEST_FIXTURE(metalayers, 1_uc) {
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_UINT8;

    int8_t ndim = 1;
    int64_t shape[] = {18};
    int64_t cshape[] = {12};
    int64_t bshape[] = {5};

    INA_TEST_ASSERT_SUCCEED(test_metalayers(data->ctx, dtype, ndim, shape, cshape, bshape, false, "arr.iarr"));
}
