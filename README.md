# iron-array

### Setup

#### Git commit-hooks

Execute the following commands:

      cp conf/pre-commit .git/hooks/


### Build

We use inac cmake build-system.

#### Windows

* INAC build setup
    * Make sure that you have a configured repository.txt file in ~\.inaos
    * Also you'll need a directory under ~\INAOS (can be empty)

* Create a build folder

         mkdir build
         cd build

* Invoke CMAKE, we have to define the generator as well as the build-type 

         cmake -G"Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Debug ..

#### Linux

* INAC build setup
    * Make sure that you have a configured repository.txt file in ~/.inaos
    * Also you'll need a directory under ~/INAOS (can be empty)

* Create a build folder

         mkdir build
         cd build

* Invoke CMAKE, we have to define the generator as well as the build-type 

         cmake -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug ..


### Limitations

#### Expressions

* For now only element-wise operations are supported in expression.

