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

#include "../iarray_private.h"
#include <libiarray/iarray.h>

int iarray_constructor_element_fn(uint8_t *dest,
                                  iarray_constructor_array_info_t *array_params,
                                  iarray_constructor_chunk_info_t *chunk_params,
                                  iarray_constructor_block_info_t *block_params,
                                  void *custom_params,
                                  void *custom_chunk_params,
                                  void *custom_block_params,
                                  iarray_constructor_item_fn item_fn) {

    iarray_container_t *a = array_params->a;
    uint8_t itemsize = a->catarr->itemsize;

    // Compute the location of each element

    iarray_constructor_item_info_t item_params;
    uint8_t *item = malloc(itemsize);

    for (int64_t nitem = 0; nitem < block_params->size; ++nitem) {
        // Compute the location of each element
        uint8_t ndim = array_params->ndim;

        item_params.index_in_block_flat = nitem;
        iarray_index_unidim_to_multidim(ndim, block_params->block_strides,
                                        item_params.index_in_block_flat, item_params.index_in_block);

        iarray_index_multidim_to_unidim(ndim, array_params->block_strides,
                                        item_params.index_in_block,
                                        &item_params.index_in_block2_flat);

        for (int i = 0; i < ndim; ++i) {
            item_params.index[i] = item_params.index_in_block[i] + block_params->start[i];
        }

        iarray_index_multidim_to_unidim(ndim, array_params->strides, item_params.index,
                                        &item_params.index_flat);

        IARRAY_RETURN_IF_FAILED(
                item_fn(array_params,
                        chunk_params,
                        block_params,
                        &item_params,
                        custom_params,
                        custom_chunk_params,
                        custom_block_params,
                        item)
        );
        memcpy(&dest[item_params.index_in_block2_flat * itemsize], item, a->dtshape->dtype_size);
    }
    free(item);
    return 0;
}


ina_rc_t iarray_constructor_element(iarray_context_t *ctx,
                                    iarray_dtshape_t *dtshape,
                                    iarray_constructor_element_params_t *element_params,
                                    iarray_storage_t *storage,
                                    iarray_container_t **container)
{
    iarray_constructor_block_params_t block_fns;
    block_fns.constructor_info = element_params->constructor_info;
    block_fns.block_init_fn = element_params->block_init_fn;
    block_fns.block_destroy_fn = element_params->block_destroy_fn;
    block_fns.chunk_init_fn = element_params->chunk_init_fn;
    block_fns.chunk_destroy_fn = element_params->chunk_destroy_fn;
    block_fns.array_init_fn = element_params->array_init_fn;
    block_fns.array_destroy_fn = element_params->array_destroy_fn;
    block_fns.item_fn = element_params->item_fn;
    block_fns.generator_fn = iarray_constructor_element_fn;

    return iarray_constructor_block(ctx, dtshape, &block_fns, storage, container);

}
