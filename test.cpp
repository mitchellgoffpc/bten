#include "src/tensor.cpp"


int main() {
    try {
        let a = Tensor<int>({ 1, 2, 3, 4, 5, 6, 7, 8 }, Shape(2, 2, 2));
        let b = Tensor<int>({ 1, 1, 2, 3 }, Shape(2, 2));
        print("a =", a);
        print("b =", b);
        print();
        print("a * b =", a * b);
        print("a * b.transpose() =", a * b.transpose());
        print("a.mean() =", a.mean());
        print("b.max(1) =", b.max(1));
    }
    catch (const char* error) {
        print(error);
    }
}
