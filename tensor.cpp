#ifndef __TENSOR__
#define __TENSOR__

#include "core.cpp"
#include "shape.cpp"
#include "buffer.cpp"


// Tensor class

template <typename T> class Tensor {
public:
    Reference<Buffer<T>> buffer;
    Shape shape;
    Shape stride;

    // Constructors

    Tensor (T data) :
        buffer(Reference<Buffer<T>>(new Buffer<T>(1, { data }))),
        shape(Shape()),
        stride(Shape()) { }

    // Vector constructors
    Tensor (List<T> data) :
        Tensor (data, Shape((int) data.size())) { }
    Tensor (List<T> data, Shape shape) :
        Tensor (data, shape, getStrideForShape(shape)) { }
    Tensor (List<T> data, Shape shape, Shape stride) :
        buffer (Reference<Buffer<T>>(new Buffer<T>(data))),
        shape  (shape),
        stride (stride) { }

    // Array construtors
    Tensor (int length, T* data) :
        Tensor (length, data, Shape(length)) { }
    Tensor (int length, T* data, Shape shape) :
        Tensor (length, data, shape, getStrideForShape(shape)) { }
    Tensor (int length, T* data, Shape shape, Shape stride) :
        buffer (Reference<Buffer<T>>(new Buffer<T>(length, data))),
        shape  (shape),
        stride (stride) { }

    // Buffer constructor
    Tensor (Reference<Buffer<T>> buffer, Shape shape, Shape stride) :
        buffer(buffer), shape(shape), stride(stride) { }

    // Copy constructor
    Tensor<T> (const Tensor<T>& other) :
        Tensor (other.buffer, other.shape, other.stride) { }

    // Destructor

    ~Tensor () {
        this->buffer.reset(); }


    // We have a bunch of tensor operations to define, and since they all share
    // a significant percentage of their structure, we'll define them as macros
    // and then expand them into the correct methods. These operations basically
    // fall into two categories: ops that act only along a single dimension, and
    // ops that act along all dimensions at once. We'll define macros for both
    // types of methods.

    // Macro for all-dimensional reduction operations
    #define fullReduction(methodName, returnType, initialValue, reduction, resultValue) \
    returnType methodName () {                                                  \
        size_t startIndex = 0;                                                  \
        returnType result = initialValue;                                       \
        for (int i = 0; i < this->buffer->length; i++)                           \
            reduction;                                                          \
        return resultValue; }

    // Macro for single-dimensional reduction operations
    #define partialReduction(methodName, returnType, initialValue, reduction, resultValue) \
    Tensor<returnType> methodName (int dim) {                                   \
        if (dim < 0)                                                            \
            dim = this->shape.length + dim;                                     \
        if (dim < 0 || dim >= this->shape.length)                               \
            throw "Tensor.methodName - Dimension index out of range";           \
                                                                                \
        Shape outputShape = this->shape.flattenDimension(dim);                  \
        Shape outputStride = getStrideForShape(outputShape);                    \
        size_t outputSize = outputShape.volume();                               \
        returnType* data = new returnType[outputSize];                          \
                                                                                \
        for (int outputIndex = 0; outputIndex < outputSize; outputIndex++) {    \
            size_t startIndex = 0;                                              \
            for (int d = 0; d < outputShape.length; d++) {                      \
                int position = outputIndex / outputStride[d] % outputShape[d];  \
                startIndex += this->stride[d] * position; }                     \
                                                                                \
            returnType result = initialValue;                                   \
            for (int d = 0; d < this->shape[dim]; d++) {                        \
                int i = startIndex + d * this->stride[dim];                     \
                reduction; }                                                    \
                                                                                \
            data[outputIndex] = resultValue; }                                  \
                                                                                \
        return Tensor<returnType>(outputSize, data, outputShape, outputStride); }

    // Macro for creating both types of methods at once
    #define reduction(methodName, returnType, initialValue, reduction, resultValue) \
        fullReduction(methodName, returnType, initialValue, reduction, resultValue); \
        partialReduction(methodName, returnType, initialValue, reduction, resultValue);

    // Sum / Mean macro expansions
    reduction(sum, T, 0, result += this->at(i), result);
    fullReduction(mean, float, 0, result += this->at(i), result / this->buffer->length);
    partialReduction(mean, float, 0, result += this->at(i), result / this->shape[dim]);

    // Min / Max macro expansions
    #define maxReduction if (this->at(i) > result) { result = this->at(i); }
    #define minReduction if (this->at(i) < result) { result = this->at(i); }
    reduction(max, T, this->at(startIndex), maxReduction, result);
    reduction(min, T, this->at(startIndex), minReduction, result);

    // Argmin / Argmax macro expansions
    #define argmaxReduction if (this->at(i) > this->at(result)) { result = i; }
    #define argminReduction if (this->at(i) < this->at(result)) { result = i; }
    partialReduction(argmax, size_t, startIndex, argmaxReduction, (result - startIndex) / this->stride[dim]);
    partialReduction(argmin, size_t, startIndex, argminReduction, (result - startIndex) / this->stride[dim]);


    // Reshaping operations

    template <typename... Args> Tensor<T> permute (int n, Args... rest) {
        return permute(List<int>({ n }), rest...); }
    template <typename... Args> Tensor<T> permute (List<int> ordering, int n, Args... rest) {
        return permute(push(ordering, n), rest...); }
    Tensor<T> permute (List<int> ordering) {
        return Tensor<T>(this->buffer, permuteShape(this->shape, ordering), permuteShape(this->stride, ordering)); }

    Tensor<T> transpose () {
        return this->permute(range(this->shape.length - 1, -1, -1)); }


    // Helper methods

    String toString () const {
        return "Tensor { data=[" + this->buffer->toString() + "], shape=" + this->shape.toString() + " }"; }

    T& at (size_t i) const {
        return (*this->buffer)[i]; }


    // Operator overloads

    // Same as above, we'll define some methods inside a macro and then expand
    // the macro for each of the operators we want to overload. This macro is a bit
    // simpler than the last one, because the overloads are all exactly the same
    // except for the operator itself.

    #define tensorOp(op)                                                            \
    Tensor<T> operator op (T other) {                                               \
        size_t outputSize = this->buffer->length;                                   \
        T* data = new T[outputSize];                                                \
        for (int i = 0; i < outputSize; i++)                                        \
            data[i] = this->at(i) + other;                                          \
        return Tensor<T>(outputSize, data, this->shape, this->stride); }            \
                                                                                    \
    Tensor<T> operator op (const Tensor<T>& other) {                                \
        Shape outputShape = getBroadcastedShape(this->shape, other.shape);          \
        Shape outputStride = getStrideForShape(outputShape);                        \
        size_t outputSize = outputShape.volume();                                   \
        T* data = new T[outputSize];                                                \
                                                                                    \
        for (int i = 0; i < outputSize; i++) {                                      \
            size_t indexA = 0, indexB = 0;                                          \
                                                                                    \
            for (int dim = 0; dim < outputShape.length; dim++) {                    \
                size_t position = i / outputStride[dim] % outputShape[dim];         \
                if (dim < this->shape.length && this->shape[dim] > 1)               \
                    indexA += this->stride[dim] * position;                         \
                if (dim < other.shape.length && other.shape[dim] > 1)               \
                    indexB += other.stride[dim] * position; }                       \
                                                                                    \
            data[i] = this->at(indexA) op other.at(indexB); }                       \
                                                                                    \
        return Tensor<T>(outputSize, data, outputShape, outputStride); }

    // Now we expand the macro with each the operators we want to overload
    tensorOp(+);
    tensorOp(-);
    tensorOp(*);
    tensorOp(/);

    // Accessor operator
    T& operator() (int a) const {
        if (a < 0 || a >= this->buffer->length)
             throw "Tensor() - Invalid index";
        else return this->at(a); }};


// Print operator

template <typename T> std::ostream& operator<< (std::ostream& os, const Tensor<T>& tensor) {
    os << tensor.toString();
    return os; }


#endif
