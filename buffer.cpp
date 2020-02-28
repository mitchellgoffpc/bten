#ifndef __BUFFER__
#define __BUFFER__

#include "core.cpp"


template <typename T> class Buffer {
public:
    size_t length;
    T* data;

    // Constructors

    // Vector constructor
    Buffer (List<T> data) : length(data.size()), data(new T[data.size()]) {
        std::copy(data.begin(), data.end(), this->data); }

    // Array constructor
    Buffer (size_t length, T* data) : length(length), data(data) { }

    // Copy constructor
    Buffer (const Buffer<T>& other) : length(other.length), data(new T[other.length]) {
        std::copy(other.data, &other.data[other.length], this->data); }


    // Destructor

    ~Buffer () {
        if (this->data)
            delete[] this->data; }


    // Unimplemented methods

    Buffer<int> toInt () {
        throw "This method hasn't been implemented for this type"; }

    Buffer<float> toFloat () {
        throw "This method hasn't been implemented for this type"; }


    // Helper methods

    String toString () const {
        return join(map(this->length, this->data, stringFromType(T))); }


    // Operator overloads

    Buffer<T>& operator= (const Buffer<T>& other) {
    	if (this == &other) return *this;

	    if (this->data) delete[] this->data;
        this->length = other.length;
        this->data = new T[other.length];
        std::copy(other.data, &other.data[other.length], this->data);
        return *this; }

    T& operator[] (int i) const {
        if (i < 0 || i >= this->length)
             throw "Invalid index";
        else return this->data[i]; }};


// Overloads for Buffer<float> methods

template<> Buffer<float> Buffer<float>::toFloat () { return *this; }
template<> Buffer<int> Buffer<float>::toInt () {
    int* data = new int[this->length];
    for (int i = 0; i < this->length; i++) {
        data[i] = (int)this->data[i]; }
    return Buffer<int>(this->length, data); }


// Overloads for Buffer<int> methods

template<> Buffer<int> Buffer<int>::toInt () { return *this; }
template<> Buffer<float> Buffer<int>::toFloat () {
    float* data = new float[this->length];
    for (int i = 0; i < this->length; i++) {
        data[i] = (float)this->data[i]; }
    return Buffer<float>(this->length, data); }


#endif
