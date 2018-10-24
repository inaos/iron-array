
#include <stdlib.h>
#include <libinac/libinac.h>

/*
 * Idea of this benchmark is to compare different implementations of the cblosc2 frames
 *
 * 1. Traditional handling of the frames by using malloc/realloc (memory) and the posix file functions to read/write the file
 * 2. Creating a branch of cblosc2 which would handle frames by using:
 *    - mmap
 *    - hugepages
 * 
 * 
 *
 *
 */

INA_BENCH_DATA(chunk_store){
    double *A;
    double *B;
    double *C;
    size_t elements;
};

INA_BENCH_SETUP(chunk_store)
{
    ina_bench_set_precision(0);
    data->elements = 1024;
    data->A = (double*)ina_mem_alloc(sizeof(double)*data->elements);
    data->B = (double*)ina_mem_alloc(sizeof(double)*data->elements);
    data->C = (double*)ina_mem_alloc(sizeof(double)*data->elements);
}

INA_BENCH_TEARDOWN(chunk_store)
{
    ina_mem_free(data->A);
    ina_mem_free(data->C);
    ina_mem_free(data->B);
}

INA_BENCH_SCALE(chunk_store)
{
    ina_bench_set_scale(1);
}

INA_BENCH_BEGIN(chunk_store, realloc) {}
INA_BENCH_END(chunk_store, realloc) {}

INA_BENCH(chunk_store, realloc, 1)
{
    size_t i_a = 0, i_b = 0;
    size_t counter = 0;

    ina_bench_stopwatch_start();
    
    ina_bench_set_int64(ina_bench_stopwatch_stop());
}
