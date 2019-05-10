[![Appveyor CI](https://ci.appveyor.com/api/projects/status/bfntjr38rymsm18w/branch/master?svg=true)](https://ci.appveyor.com/project/stoni/iron-array/branch/master) [![codecov](https://codecov.io/gh/inaos/iron-array/branch/master/graph/badge.svg?token=HFqpNSEpsN)](https://codecov.io/gh/inaos/iron-array)

# iron-array

### Setup

#### Git commit-hooks

Execute the following commands:

      cp conf/pre-commit .git/hooks/


### Build

We use inac cmake build-system.

#### Windows

* INAC build setup
    * Make sure that you have a configured repository.txt file in ~\.inaos\cmake
    * Also you'll need a directory under ~\INAOS (can be empty)

* Create a build folder

         mkdir build
         cd build

* Invoke CMAKE, we have to define the generator as well as the build-type

         cmake -G"Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Debug ..
         cmake -G"Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=RelWithDebInfo ..

#### Mac

* INAC build setup
    * Make sure that you have a configured repository.txt file in ~/.inaos/cmake
    * Also you'll need a directory under ~/INAOS (can be empty)

* Create a build folder

         mkdir build
         cd build

* Invoke CMAKE, we have to define the build-type

         cmake -DCMAKE_BUILD_TYPE=Debug ..
         cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..

#### Linux

* INAC build setup
    * Make sure that you have a configured repository.txt file in ~/.inaos
    * Also you'll need a directory under ~/INAOS (can be empty)

* Create a build folder

         mkdir build
         cd build

* Invoke CMAKE, we have to define the build-type, but only two types are supported

         cmake -DCMAKE_BUILD_TYPE=Debug ..
         cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..

### Limitations

#### Expressions

* For now only element-wise operations are supported in expression.

