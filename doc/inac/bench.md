### Performance testing

#### Features
 * Easy adding benchmarks test with minimal effort. Non header files required.
 * CSV output
 * Integrated stopwatch
 * Supports CPU scheduling
 * Supports skipping series
 * Minimal memory footprint
 * Automatic result aggregation
 * Working the same way on Linux, Windows and OS X 
 
#### Adding benchmarks
To add a benchmark use the INA_BENCH_* macros. Every benchmark is composed at least 
of 1 series which is executed at least 1 time. A benchmark has the following phases:
- __Setup__: called once for every series in the benchmark
- __Scale__: called for the current series at every repetition 
- __Begin__: called once for a series at each repetition
- __Benchmark__: called for the current series at every iteration
- __End__: called once for a series at every repetition
- __Teardown__: called for every series in the benchmark

Every phase must be declared by the corresponding macro.

##### The setup phase

Use _INA_BENCH_SETUP_ to define the setup phase. This phase is destined to run 
common setup code for series, setting scale label and precision. 

`INA_BENCH_SETUP([benchmark name])`

You may call _ina_bench_set_scale_label()_, _ina_bench_get_iterations()_, 
_ina_bench_get_repetitions()_, _ina_bench_get_name()_, 
_ina_bench_get_series_name()_ during this phase.

```C
INA_BENCH_SETUP(sort) {
    ina_bench_set_scale_label("ns");
    ina_bench_set_precision(2);
    data->nr_of_elements = 1000000;
    data->elements = ina_mem_alloc(sizeof(element)*data->nr_of_elements);
}
```

If the setup phase is not need just declare it with an empty body. 

```C
INA_BENCH_SETUP(sort) {}
```
##### The scale phase
The scale phase is intended to capture the scale value. You need to
call _ina_bench_set_scale()_ before leaving the phase. This phase is called 
for each repetition just before running the benchmark code.

`INA_BENCH_SCALE([benchmark name])`

You may also call _ina_bench_get_iterations()_, _ina_bench_get_repetition()_, 
_ina_bench_get_repetitions()_, _ina_bench_get_name()_, 
_ina_bench_get_series_name()_
during this phase.

```C
INA_BENCH_SCALE(sort) {
    ina_bench_set_scale(data->nr_of_elements*ina_bench_get_repetition());
}
```

##### The begin phase
The begin phase is designated to run setup code for a single series for
a repetition.  

`INA_BENCH_BEGIN([benchmark name], [series name])`

You may call _ina_bench_get_iterations()_, _ina_bench_get_repetitions()_,
_ina_bench_get_name()_, _ina_bench_get_series_name()_ during this phase.

```C
INA_BENCH_BEGIN(sort, quick_sort) {
    data->sort_fn = __ina_quicksort;
}

```
If the begin phase is not need just declare it with an empty body. 
```
INA_BENCH_BEGIN(sort, quick_sort) {}
```


##### The benchmark phase
This phase is intended to run the effective benchmark code and capture the
measurement. The benchmark phase is called for each series, iteration and
repetition. 
_ina_bench_set_value()_ must called  before leaving the benchmark phase.

```INA_BENCH(benchmark name], [series name], [nr of iterations] [nr of repetition])```

Most of the time the measurements consists of time measurements 
. The benchmark framework provide _ina_bench_stopwatch_start()_ and 
_ina_bench_stopwatch_stop()_ to this end. One have to call
_ina_bench_set_value()_before leaving the phase in order to store
the benchmark value for the current iteration and repetition.

You may also call _ina_bench_get_iterations()_, _ina_bench_get_iteration()_,
_ina_bench_get_repetitions()_, _ina_bench_get_repetition()_
_ina_bench_get_name()_, _ina_bench_get_series_name()_ during this phase.

```C
INA_BENCH(sort, quick_sort, 100, 10) {
    ina_bench_stopwatch_start();
    data->sort_fn(data->elements, data->nr_of_elements);
    ina_bench_set_value((double)ina_bench_stopwatch_stop());
}
```

The number of iterations can be overridden with `--x-iter` command line 
argument. The number of repetition can be overridden with command line option
`--x-repeat`.


##### The end phase
The begin phase is designated to run cleanup code for a single series.

`INA_BENCH_END([benchmark name], [series name])`

You may also call _ina_bench_get_iterations()_,  _ina_bench_get_repetitions()_,
_ina_bench_get_name()_, _ina_bench_get_series_name()_,  during this phase.

```C
INA_BENCHEND(sort, quick_sort) {
   __ina_do_something_after_bench();
}
```

If the end phase is not need just declare it with an empty body. 
```
INA_BENCH_END(sort, quick_sort) {}
```

##### The teardown phase
Use _INA_BENCH_TEARDOWN_ to define the teardown phase. This phase is 
destined to run common teardown code for series.

`INA_BENCH_TEARDOWN([benchmark name])`

You may call _ina_bench_get_iterations()_, _ina_bench_get_name()_, 
_ina_bench_get_series_name()_ during this phase.

```C

INA_BENCH_TEARDOWN(sort) {
    ina_mem_free(data->elements);
}
```

If the teardown phase is not need just declare it with an empty body. 

```C
INA_BENCH_TEARDOWN(sort) {}
```

#### How to run benchmarks

To run the benchmarks simply call _ina_bench_run()_. The application need
to be initialized as regular INAC application by calling `ina_app_init()`.
Therefore a minimal benchmark executable must like looks like this.


    #include <libinac/lib.h>

    int main(int argc,  char** argv)
    {
        if (INA_FAILED(ina_app_init(argc, argv, NULL))) {
            return EXIT_FAILURE;
        }
        return ina_bench_run();
    }
    
This will run all benchmarks with the defined repetitions and iterations 
without any warm.up iterations.  The reports will be generated in the 
current working directory. 

The benchmark runner looks for command line arguments:
 
 - `r`: specify the report location
 - `n`: to restrict benchmark execution by a name  filter
 - `x-repeat`: to override the number of repetitions
 - `x-iter`: to override the number of iterations
 - `x-warmp-up`: define the number of warm-up iterations (default 0)
 - `cache-size`: to specify L1/L2/L3 cache size
 - `disable-aggregation`:  disable result aggregation.

A more advanced benchmark runner could take in account of these command line
options.

    #include <libinac/lib.h>

    int main(int argc,  char** argv)
    {
        INA_OPTS(opt,
                 INA_OPT_STRING("r", "report-path", "."INA_PATH_SEPARATOR_STR, "Directory for report output"),
                 INA_OPT_INT(NULL, "x-repeat", INA_NUM2STR(0), "Override number of repetitions"),
                 INA_OPT_INT(NULL, "x-iter", INA_NUM2STR(0), "Override number of iteration"),
                 INA_OPT_INT(NULL, "cache-size", INA_NUM2STR(0), "L1/L2/L3 cache size"),
                 INA_OPT_INT(NULL, "x-warm-up", INA_NUM2STR(0), "Warm-up iterations"),     
                 INA_OPT_FLAG(NULL, "disable-aggregation", "Disable result aggregation"),
                 INA_OPT_STRING("n", "name", "", "Benchmark name"));
    
        if (INA_FAILED(ina_app_init(argc, argv, opt))) {
            return EXIT_FAILURE;
        }
        return ina_bench_run();
    }

Implemented in this way one can run from the command line prompt
all benchmarks, a single benchmark or group of benchmarks...

    ./bench
    ./bench -n io_file
    ./bench -n io_

... define the location where reports should be stored...

    ./bench -r /home/reports

... override iterations and repetitions use `--x-iter` and `--x-repeat`
command line options.

    ./bench --x-iter=100000 --x-repeat=1000

... add extra warm-up iterations for instruction cache.

    ./bench --x-warm-up=2
    
... or disable result aggregation to see result for each single iteration.

    ./bench --x-warm-up=2
