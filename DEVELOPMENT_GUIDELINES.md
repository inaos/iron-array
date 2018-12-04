# Development Guidelines

## Style and code conventions

### Structs

Following our guideline:

     typedef struct iarray_example_s {
         int test;
     } iarray_example_t;

### Function and brackets

* Open and closing brackets of functions are alwayls on the beginning of the line
* The backet open or close is alwayls the only character on the line

### Adhere to INAC conventions wherever possible

* Alwalys use ina_rc_t as return type of functions
* Only for functions that end in suffix '_free' we should use the 'void'

### Good practices for developing C libraries

http://lucumr.pocoo.org/2013/8/18/beautiful-native-libraries/
