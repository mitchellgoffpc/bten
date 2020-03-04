# BTen

BTen is a lightweight tensor library written in C++11. It's named after PyTorch's ATen library, but has a few
significant differences. It's much smaller than ATen (currently only 4 cpp files), and uses C++ templates for
the underlying tensor types (int, float, etc) instead of a dynamic dispatch like ATen. I mostly chose to build
it this way because I wanted to play around more with C++'s template system and see how flexible it is compared
to Java/Scala generics. This design choice would make developing python bindings for BTen quite difficult, but
I plan on keeping it C++ only (besides potentially a Swift port to experiment with Metal's compute shaders), and
having statically-dispatched methods does slightly improve performance. I'm currently working on support for
automatic differentiation and backpropagation; unlike ATen, BTen's tensor objects will create computational graphs
by default during the forward pass of all differentiable tensor operations.


To compile and execute the included test code:
`g++ test.cpp -std=c++11 -o test && ./test`
