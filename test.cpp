#include "core.cpp"
#include "tensor.cpp"


int main() {
    try {
        print("Creating tensor<int> a and b");
        let a = Tensor<int>({ 1, 2, 3, 4, 5, 6 }, Shape(1, 2, 3));
        let b = Tensor<int>({ 4, 8, 12 }, Shape(3, 1));
        print(a.transpose() + b + 1);
    }
    catch (const char* error) {
        print(error); }}
