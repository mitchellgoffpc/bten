#ifndef __CORE__
#define __CORE__

#include <limits>
#include <random>
#include <string>
#include <vector>
#include <iostream>


// Some type definitions
typedef std::string String;
template <typename T> using List = std::vector<T>;
template <typename T> using Reference = std::shared_ptr<T>;


// Macros
#define let auto
#define stringFromType(T) ((String(*)(T)) std::to_string)


// Print function
void print () { std::cout << std::endl; }

template <typename T, typename... Args> void print (T input, Args... rest) {
    std::cout << input << " ";
    print(rest...); }


// Range function
List<int> range (int start, int end, int stride) {
    let result = List<int>();
    for (int i = start; stride > 0 ? i < end : i > end; i += stride)
        result.push_back(i);
    return result; }

List<int> range (int start, int end) { return range(start, end, 1); }
List<int> range (int end) { return range(0, end); }


// Vector functions

template <typename A, typename B> List<B> map (List<A> input, B (*f)(A)) {
    List<B> result;
    transform(input.begin(), input.end(), std::back_inserter(result), f);
    return result; }
template <typename A, typename B> List<B> map (int length, A* input, B (*f)(A)) {
    List<B> result;
    transform(input, &input[length], std::back_inserter(result), f);
    return result; }

template <typename A> List<A> filter (List<A> input, bool (*f)(A)) {
    List<A> result;
    copy_if(input.begin(), input.end(), std::back_inserter(result), f);
    return result; }
template <typename A> List<A> filter (int length, A* input, bool (*f)(A)) {
    List<A> result;
    copy_if(input, &input[length], std::back_inserter(result), f);
    return result; }

template <typename A> List<A> push (List<A> input, A x) {
    input.push_back(x);
    return input; }

template <typename A> List<A> concat (List<A> a, List<A> b) {
    List<A> result;
    result.reserve(a.size() + b.size());
    copy(a.begin(), a.end(), std::back_inserter(result));
    copy(b.begin(), b.end(), std::back_inserter(result));
    return result; }


// String functions

template <typename T> String join(List<T> input, String delimiter) {
    return join(map(input, std::to_string), delimiter); }
template <typename T> String join(List<T> input) {
    return join(map(input, std::to_string)); }

template <> String join(List<String> input, String delimiter) {
    String result;
    for (String x : input) {
        if (!result.empty()) { result += delimiter; }
        result += x; }
    return result; }
template <> String join(List<String> input) {
    return join(input, ","); }

template <typename T> String join(int length, T* input, String delimiter) {
    return join(map(length, input, (String (*)(T)) std::to_string), delimiter); }
template <typename T> String join(int length, T* input) {
    return join(map(length, input, (String (*)(T)) std::to_string)); }


#endif
