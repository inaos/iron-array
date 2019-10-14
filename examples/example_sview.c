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


int main()
{
    int8_t ndim = 2;
    iarray_data_type_t dtype = IARRAY_DATA_TYPE_DOUBLE;
    int64_t shape[] = {10, 10};
    int64_t pshape[] = {2, 3};
    int64_t bshape[] = {2, 10};

    iarray_config_t cfg = IARRAY_CONFIG_DEFAULTS;
    iarray_context_t *ctx;
    INA_FAIL_IF_ERROR(iarray_context_new(&cfg, &ctx));

    iarray_dtshape_t dtshape;
    dtshape.ndim = ndim;
    dtshape.dtype = dtype;
    int64_t size = 1;

    for (int i = 0; i < ndim; ++i) {
        dtshape.shape[i] = shape[i];
        dtshape.pshape[i] = pshape[i];
        size *= shape[i];
    }
    iarray_container_t *cont;
    INA_FAIL_IF_ERROR(iarray_arange(ctx, &dtshape, 0, size, 1, NULL, 0, &cont));

    int64_t start[] = {2, 3};
    int64_t stop[] = {9, 7};

    iarray_container_t *cout;
    iarray_get_slice(ctx, cont, start, stop, pshape, NULL, 0, true, &cout);
    iarray_linalg_transpose(ctx, cout);

    uint8_t *sview;
    int64_t sview_len;

    INA_FAIL_IF_ERROR(iarray_to_sview(ctx, cout, &sview, &sview_len));

    iarray_container_t *cview;
    INA_FAIL_IF_ERROR(iarray_from_sview(ctx, sview, sview_len, &cview));

    return INA_SUCCESS;

    fail:
    iarray_container_free(ctx, &cout);
    iarray_container_free(ctx, &cont);
    iarray_context_free(&ctx);
    return ina_err_get_rc();

}
