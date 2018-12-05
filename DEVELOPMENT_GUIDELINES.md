# Development Guidelines

## Style and code conventions

### Indentation

* Use 4 spaces

### Structs

Following our guideline:

     typedef struct iarray_example_s {
         int test;
     } iarray_example_t;

### Function and brackets

* Open and closing brackets of functions are alwayls on the beginning of the line
* The backet open or close is alwayls the only character on the line

#### API Functions

     INA_API(ina_rc_t) iarray_[module]_[function]_[op](...)
     {
     ...
     }
     
#### Private Functions

     static ina_rc_t _iarray_[module]_[function]_[op](...)
     {
     ...
     }
     
### Large number of arguments in functions

    INA_API(ina_rc_t)function(int arg1,
                              int arg2,
                              int arg3,
                              ...
                              )
     {
     ...
     }

### Adhere to INAC conventions wherever possible

* Alwalys use ina_rc_t as return type of functions
* Only for functions that end in suffix '_free' we should use the 'void'

### File names

Source files should prefer '_' (underscores) to '-' (dashes).  Example:

    test_this.c
    bench_that.c

### Good practices for developing C libraries

http://lucumr.pocoo.org/2013/8/18/beautiful-native-libraries/
