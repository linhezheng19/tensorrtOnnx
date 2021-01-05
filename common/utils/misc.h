/**
 * Created by linhezheng.
 * Some general functions.
 * 2020/09/01
 */

#ifndef MISC_H
#define MISC_H

#include <iostream>

using namespace std;

/* ===========Template Misc Operation ========== */
template <typename T>
inline T sigmoid(const T &n) {
    return 1 / (1+exp(-n));
}

template <typename T>
inline T clip(const T &n, const T &lower, const T &upper) {
	return std::max(lower, std::min(n, upper));
}

template<class ForwardIterator>
inline size_t argmin(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::min_element(first, last));
}

template<class ForwardIterator>
inline size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

#endif // MISC_H
