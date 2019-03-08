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

### Conditionals for data types

Although right now we only support floats and doubles for IronArray, it is recommended to use
a `switch` for dealing with the different data types rather than an `if ... else`.  Example:

    switch (dtshape->dtype) {
        case IARRAY_DATA_TYPE_DOUBLE:
            type_size = sizeof(double);
            break;
        case IARRAY_DATA_TYPE_FLOAT:
            type_size = sizeof(float);
            break;
        default:
            return INA_ERR_EXCEEDED;
    }

Note that the 'default' statement is there mainly for avoiding compiler warnings.

### Types for lengths in containers

IronArray containers lengths are expressed just as `int64_t`, so all the shapes, lengths and
other related variables should be exactly of this type.  Also, the `ndim` type is `int8_t` so this
is the type that must be used to express the number of dimensions.

### Be proactive and avoid compiler warnings

Since 2018-03-08 IronArray can compile mostly warning free (except in `Release` mode, where `assert`
checks and related macros like `INA_ASSERT_NOT_NULL` are removed by the optimizer).

Being able to compile warning free is a good practice that we must observe carefully *before*
doing a commit.  A nice way to be proactive is to look at the interactive CLion code analysis
and keep the number of warnings to a minimum (beware, some are false positives or not too important,
so try to get a balance by keeping CLion warnings under a minimum).

Also, it is useful to have a look at the building logs of the CI because MSVC and GCC can throw
different warnings in different places of the code.  Please regularly visit these logs and
try to fix the ones that you introduced.

### File names

Source files should prefer '_' (underscores) to '-' (dashes).  Example:

    test_this.c
    bench_that.c

### Good practices for developing C libraries

http://lucumr.pocoo.org/2013/8/18/beautiful-native-libraries/
