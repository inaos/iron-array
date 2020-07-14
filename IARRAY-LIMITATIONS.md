# Limitations in ironArray

* The `iterchunk` evaluation method only have support for the `interpreted` method, not for the `compiled` engine.  The reason is that `iterchunk` does not have support for multithreading, and we decided that it is not worth to give support for compiled code to a method that cannot do multithreading.
