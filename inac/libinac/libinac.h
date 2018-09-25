/* temporary mock of INAC */

#ifndef __INAC__
#define __INAC__

#include <stdlib.h>

#define INA_API(x) x  // FIXME: make the function public?
#define INA_SUCCESS 0
typedef int ina_rc_t;

typedef void* ina_mempool_t;

#define ina_mem_alloc malloc
#define ina_mem_free free
#define ina_mem_set memset
#define ina_mem_cpy memcpy

#define INA_VERIFY_NOT_NULL(ptrptr)
#define INA_RETURN_IF_NULL(ptrptr)
#define INA_FREE_CHECK(ptrptr)
#define INA_MEM_FREE_SAFE(ptrptr)

void* ina_mempool_dalloc(ina_mempool_t *mp, size_t size);

#endif
