#ifndef RADIX_SELECT_H
#define RADIX_SELECT_H

#include <cuda.h>
#include <assert.h>
#include <stdio.h>
#include <algorithm>

#define MAX_TENSORINFO_DIMS 4

#define CUDA_KERNEL_ASSERT(cron)

#define CHECK(res) {                                                        \
    if(res != cudaSuccess) {                                                \
        printf("Error ：%s:%d , ", __FILE__,__LINE__);                      \
        printf("code : %d , reason : %s \n", res, cudaGetErrorString(res)); \
        exit(-1);                                                           \
    }                                                                       \
}

constexpr int RADIX_BITS = 2; // digits are base-(2 ^ RADIX_BITS)
constexpr int RADIX_SIZE = 4; // 2 ^ RADIX_BITS
constexpr int RADIX_MASK = (RADIX_SIZE - 1);

template <typename T>
__host__ __device__ __forceinline__ T CeilDiv(T a, T b) {
    return (a + b - 1) / b;
}

template <typename T>
__host__ __device__ __forceinline__ T RoundUp(T a, T b) {
    return CeilDiv<T>(a, b) * b;
}

__device__ __forceinline__ uint32_t getLinearBlockId() {
    // return blockIdx.z * gridDim.y * gridDim.x +
    //        blockIdx.y * gridDim.x +
    //        blockIdx.x;
    return blockIdx.x;
}

struct AddOp {
    __device__ __forceinline__ int operator()(
            int const &lhs,
            int const &rhs) {
        return lhs + rhs;
    }
};

template <typename In, typename Out>
struct ScalarConvert {
    static __host__ __device__ Out to(const In v) {
        return (Out) v;
    }
};

struct TensorInfo {
    TensorInfo() {
        data = nullptr;
        dims = 0;
    }

    TensorInfo(
            float* p,
            int dim,
            uint32_t sz[MAX_TENSORINFO_DIMS],
            uint32_t st[MAX_TENSORINFO_DIMS]) {
        data = p;
        dims = dim;
        assert(dims <= MAX_TENSORINFO_DIMS);

        for (int i = 0; i < dim; ++i) {
            sizes[i] = sz[i];
            strides[i] = st[i];
        }
    }

    void reduceDim(int dim) {
        assert(dim < dims && dim >= 0);
        sizes[dim] = 1;
    }

    float* data;
    uint32_t sizes[MAX_TENSORINFO_DIMS];
    uint32_t strides[MAX_TENSORINFO_DIMS];
    int dims;
};

struct IndexToOffset {
    static __host__ __device__ uint32_t get(
            uint32_t linearId,
            const TensorInfo& info,
            int dims) {
        uint32_t offset = 0;

        if (dims < 0) {
            dims = info.dims - dims;
        }

        assert(dims >= 0 && dims < info.dims);

        // Uses static dims
        for (int i = dims; i > 0; --i) {
            uint32_t curDimIndex = linearId % info.sizes[i];
            uint32_t curDimOffset = curDimIndex * info.strides[i];
            offset += curDimOffset;
            linearId /= info.sizes[i];
        }

        return offset + linearId * info.strides[0];
    }
};

struct TopKTypeConfig {
    static inline __device__ uint32_t convert(float v) {
        uint32_t x = __float_as_int(v);
        uint32_t mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
        return (v == v) ? (x ^ mask) : 0xffffffff;
    }

    static inline __device__ float deconvert(uint32_t v) {
        uint32_t mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
        return __int_as_float(v ^ mask);
    }
};

struct Bitfield {
    static __device__ __forceinline__
    uint32_t getBitfield(uint32_t val, int pos, int len) {
        uint32_t ret;
        asm("bfe.u32 %0, %1, %2, %3;" :
        "=r"(ret) : "r"(val), "r"(pos), "r"(len));
        return ret;
    }

    static __device__ __forceinline__
    uint32_t setBitfield(uint32_t val, uint32_t toInsert, int pos, int len) {
        uint32_t ret;
        asm("bfi.b32 %0, %1, %2, %3, %4;" :
        "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
        return ret;
    }
};

__device__ __forceinline__ int32_t getLaneId() {
    int32_t laneId;
    asm("mov.s32 %0, %%laneid;" : "=r"(laneId) );
    return laneId;
}

__device__ __forceinline__ uint32_t getLaneMaskLe() {
    uint32_t mask;
    asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
    return mask;
}

template <typename T, bool KillWARDependency, class BinaryFunction>
__device__ void inclusiveBinaryPrefixScan(
        T* smem,
        bool in,
        T* out,
        BinaryFunction binop) {
    // Within-warp, we use warp voting.
    T vote = __ballot_sync(__activemask(), in);
    T index = __popc(getLaneMaskLe() & vote);
    T carry = __popc(vote);

    int warp = threadIdx.x / 32;

    // Per each warp, write out a value
    if (getLaneId() == 0) {
        smem[warp] = carry;
    }

    __syncthreads();

    // Sum across warps in one thread. This appears to be faster than a
    // warp shuffle scan for CC 3.0+
    if (threadIdx.x == 0) {
        int current = 0;
        for (int i = 0; i < blockDim.x / 32; ++i) {
            T v = smem[i];
            smem[i] = binop(smem[i], current);
            current = binop(current, v);
        }
    }

    __syncthreads();

    // load the carry from the preceding warp
    if (warp >= 1) {
        index = binop(index, smem[warp - 1]);
    }

    *out = index;

    if (KillWARDependency) {
        __syncthreads();
    }
}

template <typename T, bool KillWARDependency, class BinaryFunction>
__device__ void exclusiveBinaryPrefixScan(
        T* smem,
        bool in,
        T* out,
        T* carry,
        BinaryFunction binop) {
    inclusiveBinaryPrefixScan<T, false, BinaryFunction>(smem, in, out, binop);

    // Inclusive to exclusive
    *out -= (T) in;

    // The outgoing carry for all threads is the last warp's sum
    *carry = smem[CeilDiv<int>(blockDim.x, 32) - 1];

    if (KillWARDependency) {
        __syncthreads();
    }
}

__device__ void countRadixUsingMask(
        int counts[4],
        int* smem,
        uint32_t desired,
        uint32_t desiredMask,
        int radixDigitPos,
        uint32_t sliceSize,
        uint32_t withinSliceStride,
        float* data);

__device__ float findPattern(
        float* smem,
        float* data,
        uint32_t sliceSize,
        uint32_t withinSliceStride,
        uint32_t desired,
        uint32_t desiredMask);

__device__ void radixSelect(
        float* data,
        uint32_t k,
        uint32_t sliceSize,
        uint32_t withinSliceStride,
        int* smem,
        float* topK,
        bool order);

__launch_bounds__ (1024)
__global__ void gatherTopK(
        TensorInfo input,
        uint32_t inputSliceSize,
        uint32_t outputSliceSize, // aka `k`

        uint32_t numInputSlices,
        uint32_t inputWithinSliceStride,

        TensorInfo topK,
        uint32_t numTopKSlices,
        uint32_t topKWithinSliceStride,

        uint32_t extra_data,
        int dim,
        bool order);

//extern "C" void test();

extern "C" void det_topk(
        float* input,
        float* output,
        const int resCount,
        const int topk,
        const int extra_dim,
        const bool order);

#endif //RADIX_SELECT_H
