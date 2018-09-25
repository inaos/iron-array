//
// Created by Francesc Alted on 25/09/2018.
//

#include "libinac.h"

void* ina_mempool_dalloc(ina_mempool_t *mp, size_t size)
{
  return malloc(size);
}
