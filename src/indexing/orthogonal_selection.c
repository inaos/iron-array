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

#include <libiarray/iarray.h>
#include <iarray_private.h>

// Indexing section
int caterva_get_orthogonal_selection(caterva_ctx_t *ctx, caterva_array_t *array,
                                     int64_t **selection, int64_t *selection_size,
                                     void *buffer, int64_t *buffershape,
                                     int64_t buffersize);

int caterva_set_orthogonal_selection(caterva_ctx_t *ctx, caterva_array_t *array,
                                     int64_t **selection, int64_t *selection_size,
                                     void *buffer, int64_t *buffershape,
                                     int64_t buffersize);


INA_API(ina_rc_t) iarray_set_orthogonal_selection(iarray_context_t *ctx,
                                                  iarray_container_t *c,
                                                  int64_t **selection, int64_t *selection_size,
                                                  void *buffer,
                                                  int64_t *buffer_shape,
                                                  int64_t buffer_size) {

    INA_ASSERT_NOT_NULL(ctx);
    INA_ASSERT_NOT_NULL(c);
    INA_ASSERT_NOT_NULL(selection);
    for (int i = 0; i < c->dtshape->ndim; ++i) {
        INA_ASSERT_NOT_NULL(selection[i]);
    }
    INA_ASSERT_NOT_NULL(selection_size);
    INA_ASSERT_NOT_NULL(buffer);
    INA_ASSERT_NOT_NULL(buffer_shape);

    if (c->container_viewed != NULL) {
        IARRAY_TRACE1(iarray.trace, "Views are not supported yet");
        IARRAY_RETURN_IF_FAILED(IARRAY_ERR_INVALID_STORAGE);
    }
    caterva_config_t cfg = {0};
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
    caterva_ctx_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_ctx_new(&cfg, &cat_ctx));
    IARRAY_ERR_CATERVA(caterva_set_orthogonal_selection(cat_ctx, c->catarr, selection, selection_size, buffer, buffer_shape, buffer_size));
    IARRAY_ERR_CATERVA(caterva_ctx_free(&cat_ctx));

    return INA_SUCCESS;
}

INA_API(ina_rc_t) iarray_get_orthogonal_selection(iarray_context_t *ctx,
                                                  iarray_container_t *c,
                                                  int64_t **selection, int64_t *selection_size,
                                                  void *buffer,
                                                  int64_t *buffer_shape,
                                                  int64_t buffer_size) {

    INA_ASSERT_NOT_NULL(ctx);
    INA_ASSERT_NOT_NULL(c);
    INA_ASSERT_NOT_NULL(selection);
    for (int i = 0; i < c->dtshape->ndim; ++i) {
        INA_ASSERT_NOT_NULL(selection[i]);
    }
    INA_ASSERT_NOT_NULL(selection_size);
    INA_ASSERT_NOT_NULL(buffer);
    INA_ASSERT_NOT_NULL(buffer_shape);

    if (c->container_viewed != NULL) {
        IARRAY_TRACE1(iarray.trace, "Views are not supported yet");
        IARRAY_RETURN_IF_FAILED(IARRAY_ERR_INVALID_STORAGE);
    }

    caterva_config_t cfg = {0};
    iarray_create_caterva_cfg(ctx->cfg, ina_mem_alloc, ina_mem_free, &cfg);
    caterva_ctx_t *cat_ctx;
    IARRAY_ERR_CATERVA(caterva_ctx_new(&cfg, &cat_ctx));
    IARRAY_ERR_CATERVA(caterva_get_orthogonal_selection(cat_ctx, c->catarr, selection, selection_size, buffer, buffer_shape, buffer_size));
    IARRAY_ERR_CATERVA(caterva_ctx_free(&cat_ctx));

    return INA_SUCCESS;
}
