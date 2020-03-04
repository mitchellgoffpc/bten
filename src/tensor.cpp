#ifndef __TENSOR__
#define __TENSOR__

#include "core.cpp"
#include "shape.cpp"
#include "buffer.cpp"

// Seed the RNG
std::random_device rd;
std::mt19937 gen(rd());


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


    // Static methods

    // Tensor::random
    static T random (T low, T high) {
        return std::uniform_int_distribution<T>(low, high)(gen); }
    static T random () {
        return random(std::numeric_limits<T>::min(), std::numeric_limits<T>::max()); }

    static Tensor<T> random (T low, T high, Shape shape) {
        let dist = std::uniform_int_distribution<T>(low, high);
        let outputSize = shape.volume();
        T* data = new T[outputSize];
        for (int i = 0; i < outputSize; i++) data[i] = dist(gen);
        return Tensor(outputSize, data, shape); }
    static Tensor<T> random (Shape shape) {
        return random(std::numeric_limits<T>::min(), std::numeric_limits<T>::max(), shape); }

    // Tensor::normal
    static T normal (T mean, T std) {
        return std::normal_distribution<T>(mean, std)(gen); }
    static T normal () { return normal(0, 1); }

    static Tensor<T> normal (T mean, T std, Shape shape) {
        let dist = std::normal_distribution<T>(mean, std);
        let outputSize = shape.volume();
        T* data = new T[outputSize];
        for (int i = 0; i < outputSize; i++) data[i] = dist(gen);
        return Tensor(outputSize, data, shape); }
    static Tensor<T> normal (Shape shape) { return normal(0, 1, shape); }

    // Tensor::constant
    static Tensor<T> constant (T value, Shape shape) {
        let outputSize = shape.volume();
        T* data = new T[outputSize];
        std::fill_n(data, outputSize, value);
        return Tensor<T>(outputSize, data, shape); }

    static Tensor<T> zeros (Shape shape) { return constant(0, shape); }
    static Tensor<T> ones  (Shape shape) { return constant(1, shape); }


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
        for (int i = 0; i < this->buffer->length; i++)                          \
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
        if (!this->shape.length)
            return "Tensor { " + std::to_string(this->at(0)) + " }";

        size_t offset = 0, length = this->shape.length - 1;
        Shape position = Shape(length, new int[length] { 0 });
        String result = "Tensor {";

        // This method is a bit of a doozy. We need to stringify one column at a time,
        // and we probably want to avoid recursing over the dimensions for performance
        // reasons, so we'll perform the iteration in a single pass, tracking both the
        // current position and current buffer offset independently. (Note: we might be
        // able to get away with tracking only the offset, and just computing the relevant
        // dimensional positions at each step, but for simplicitiy it's easiest to just
        // track them both.) The inner for-loops are just for handling indentations and
        // opening/closing brackets.

        while (position[0] < this->shape[0]) {
            result += "\n  ";

            // Render the opening brackets
            int overflowIndex = length - 1;
            for (; overflowIndex >= 0 && position[overflowIndex] == 0; overflowIndex--);
            for (int i = 0; i < length; i++)
                result += i <= overflowIndex ? " " : "[";

            // Render the current column, and then update the offset and position
            result += this->columnToString(offset);
            offset += this->stride[length - 1];
            position[-1] += 1;

            // Render the closing brackets, and carry any overflows in the position
            // over to the next dimension. For example, if `this->shape` is [2, 3, 4] and
            // `position` is [0, 2, 4], we should handle the overflow by updating `position`
            // to [1, 0, 0].
            int i = length - 1;
            for (; i > 0 && position[i] >= this->shape[i]; i--) {
                result += "]";
                offset -= this->stride[i] * this->shape[i];
                position[i] = 0;
                if (i > 0) {
                    offset += this->stride[i];
                    position[i - 1] += 1; }}

            // Render the comma after each line, plus any required newlines for
            // separation between elements in higher dimensions
            if (position[0] < this->shape[0]) {
                result += ',';
                for (; i < length - 1; i++)
                    result += '\n'; }}

        result += "]\n}";
        return result; }

    String columnToString (size_t offset) const {
        int length = this->shape[-1];
        int stride = this->stride[-1];
        T* values = new T[length];
        for (int i = 0; i < length; i++)
            values[i] = this->at(offset + i * stride);
        String result = join(length, values);
        delete[] values;
        return "[" + result + "]"; }

    // Helper method to get element `i` from this tensor's buffer. Useful
    // because `shared_ptr` doesn't support the [] operator.
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

// Template specializations for the Tensor::random method. Putting this in a
// macro probably violates Fowler's rule of three, but I didn't really want
// to write out all the definitions twice to handle both doubles and floats.

#define randomSpecialization(T)                                                 \
template<> T Tensor<T>::random (T low, T high) {                                \
    return std::uniform_real_distribution<T>(low, high)(gen); }                 \
template<> T Tensor<T>::random () {                                             \
    return Tensor<T>::random(0, 1); }                                           \
                                                                                \
template<> Tensor<T> Tensor<T>::random (T low, T high, Shape shape) {           \
    let dist = std::uniform_real_distribution<T>(low, high);                    \
    let outputSize = shape.volume();                                            \
    T* data = new T[outputSize];                                                \
                                                                                \
    for (int i = 0; i < outputSize; i++)                                        \
        data[i] = dist(gen);                                                    \
                                                                                \
    return Tensor<T>(outputSize, data, shape); }                                \
                                                                                \
template<> Tensor<T> Tensor<T>::random (Shape shape) {                          \
    return Tensor<T>::random(0, 1, shape); }

// Expand the macro for both floats and doubles
randomSpecialization(float);
randomSpecialization(double);


#endif
