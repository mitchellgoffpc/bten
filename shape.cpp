#ifndef __SHAPE__
#define __SHAPE__

#include "core.cpp"


struct Shape {
    size_t length;
    int* dimensions;

    // Constructors

    // Basic constructors
    Shape () : length(0), dimensions(NULL) { }
    Shape (int n) : length(1), dimensions(new int[1]) {
        this->dimensions[0] = n; }

    // Vector constructors
    template <typename... Args> Shape (int n, Args... rest) :
        Shape(List<int>({ n }), rest...) { }
    template <typename... Args> Shape (List<int> dimensions, int n, Args... rest) :
        Shape(push(dimensions, n), rest...) { }

    Shape (List<int> dimensions) : length(dimensions.size()), dimensions(new int[dimensions.size()]) {
        std::copy(dimensions.begin(), dimensions.end(), this->dimensions); }

    // Array constructor
    Shape (size_t length, int* dimensions) : length(length), dimensions(dimensions) { }

    // Copy constructor
    Shape (const Shape& other) : length(other.length), dimensions(new int[other.length]) {
        std::copy(other.dimensions, &other.dimensions[other.length], this->dimensions); }


    // Destructor

    ~Shape () {
        if (this->dimensions)
            delete[] this->dimensions; }


    // Helper methods

    String toString () const {
        return "Shape(" + join(map(this->length, this->dimensions, stringFromType(int))) + ")"; }

    size_t volume () const {
        int result = 1;
        for (int i = 0; i < this->length; i++)
            result *= this->dimensions[i];
        return result; }

    Shape flattenDimension (int dim) {
        if (dim < 0)
            dim = this->length + dim;
        if (dim < 0 || dim >= this->length)
            throw "Shape.flattenDimension - Index out of range";

        let newShape = Shape(*this);
        newShape[dim] = 1;
        return newShape;
    }


    // Operator overloads for Shape

    Shape& operator= (const Shape& other) {
    	if (this == &other) return *this;

	    if (this->dimensions) delete[] this->dimensions;
        this->length = other.length;
        this->dimensions = new int[other.length];
        std::copy(other.dimensions, &other.dimensions[other.length], this->dimensions);
        return *this; }

    int& operator[] (int i) const {
        if (i < 0 || i >= this->length)
             throw "Shape[] - Index out of range";
        else return this->dimensions[i]; }

    bool operator== (const Shape& other) const {
        if (this->length != other.length)
            return false;

        for (int i = 0; i < this->length; i++) {
            if (this->dimensions[i] != other.dimensions[i])
                return false; }

        return true; }

    bool operator!= (const Shape& other) const {
        return !(*this == other); }};



// Helper functions

bool shapesAreCompatible (const Shape& a, const Shape& b) {
    size_t num_dimensions = std::min(a.length, b.length);

    for (int i = 0; i < num_dimensions; i++) {
        if (a[i] != 1 && b[i] != 1 && a[i] != b[i])
             return false; }

    return true; }

Shape getBroadcastedShape (const Shape& a, const Shape& b) {
    size_t num_dimensions = std::max(a.length, b.length);
    int* dimensions = new int[num_dimensions];

    for (int i = 0; i < num_dimensions; i++) {
        if (i >= a.length)
            dimensions[i] = b[i];
        else if (i >= b.length)
            dimensions[i] = a[i];
        else if (a[i] == 1 || b[i] == 1 || a[i] == b[i])
            dimensions[i] = std::max(a[i], b[i]);
        else
            throw "These tensor shapes aren't compatible with each other!"; }

    return Shape(num_dimensions, dimensions); }

Shape getStrideForShape (const Shape& shape) {
    size_t volume = shape.volume();
    int* strides = new int[shape.length];

    for (int i = shape.length - 1; i >= 0; i--) {
        if (i == shape.length - 1)
             strides[i] = 1;
        else strides[i] = strides[i + 1] * shape[i + 1]; }

    return Shape(shape.length, strides); }

Shape permuteShape (const Shape& shape, const List<int>& ordering) {
    if (ordering.size() != shape.length)
        throw "permuteShape - The given ordering doesn't have the same number of elements as the shape being permuted";

    let shapeData = List<int>();
    for (int dim : ordering)
        shapeData.push_back(shape[dim]);
    return Shape(shapeData); }



// Print operator

std::ostream& operator<< (std::ostream& os, const Shape& shape) {
    os << shape.toString();
    return os; }


#endif
