# bmc
`bmc` is a work-in-progress toy compiler for the B-Minor Programming Language specified in [Introduction to Compilers and Language Design](https://www3.nd.edu/~dthain/compilerbook/) by Douglas Thain.

I'm following along with this textbook and a few other resources in order to learn more about compilers. The end goal for this project is to write a compiler program which can accept a B-Minor source file and emit a file of arm32 assembly which can be linked into a larger program. A stretch goal would be to emit a WebAssembly text file in order to be able to run the program as part of a WebAssembly binary.