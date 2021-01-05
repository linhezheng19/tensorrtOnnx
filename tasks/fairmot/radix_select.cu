#include "radix_select.h"

__device__ void countRadixUsingMask(
        int counts[RADIX_SIZE],
        int* smem,
        uint32_t desired,
        uint32_t desiredMask,
        int radixDigitPos,
        uint32_t sliceSize,
        uint32_t withinSliceStride,
        float* data) {
    // Clear out per-thread counts from a previous round
#pragma unroll
    for (int i = 0; i < RADIX_SIZE; ++i) {
        counts[i] = 0;
    }

    if (threadIdx.x < RADIX_SIZE) {
        smem[threadIdx.x] = 0;
    }
    __syncthreads();

    // Scan over all the data. Upon a read, the warp will accumulate
    // counts per each digit in the radix using warp voting.
    for (uint32_t i = threadIdx.x; i < sliceSize; i += blockDim.x) {
        uint32_t val =
                TopKTypeConfig::convert(*(&data[i * withinSliceStride]));

        bool hasVal = ((val & desiredMask) == desired);
        uint32_t digitInRadix = Bitfield::getBitfield(val, radixDigitPos, RADIX_BITS);

#pragma unroll
        for (uint32_t j = 0; j < RADIX_SIZE; ++j) {
            bool vote = hasVal && (digitInRadix == j);
            counts[j] += __popc(__ballot_sync(__activemask(), vote));
        }
    }

    // Now, for each warp, sum values
    if (getLaneId() == 0) {
#pragma unroll
        for (uint32_t i = 0; i < RADIX_SIZE; ++i) {
            atomicAdd(&smem[i], counts[i]);
        }
    }

    __syncthreads();

    // For each thread, read in the total counts
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
        counts[i] = smem[i];
    }

    __syncthreads();
}

__device__ float findPattern(
        float* smem,
        float* data,
        uint32_t sliceSize,
        uint32_t withinSliceStride,
        uint32_t desired,
        uint32_t desiredMask) {
    if (threadIdx.x < 2) {
        smem[threadIdx.x] = static_cast<float>(0);
    }
    __syncthreads();

    // All threads participate in the loop, in order to sync on the flag
    uint32_t numIterations = RoundUp<uint32_t>(sliceSize, blockDim.x);
    for (uint32_t i = threadIdx.x; i < numIterations; i += blockDim.x) {
        bool inRange = (i < sliceSize);
        float v = inRange ? *(&data[i * withinSliceStride])
                            : static_cast<float>(0);

        if (inRange &&
            ((TopKTypeConfig::convert(v) & desiredMask) == desired)) {
            // There should not be conflicts if we are using findPattern,
            // since the result is unique
            smem[0] = static_cast<float>(1);
            smem[1] = v; // can't use val as the flag, since it could be 0
        }

        __syncthreads();

        float found = smem[0];
        float val = smem[1];

        __syncthreads();

        // Check to see if a thread found the value
        if (found != static_cast<float>(0)) {
            // all threads return this value
            return val;
        }
    }

    // should not get here
    assert(false);
    return static_cast<float>(0);
}

__device__ void radixSelect(
        float* data,
        uint32_t k,
        uint32_t sliceSize,
        uint32_t withinSliceStride,
        int* smem,
        float* topK,
        bool order) {
    // Per-thread buckets into which we accumulate digit counts in our
    // radix
    int counts[RADIX_SIZE];

    // We only consider elements x such that (x & desiredMask) == desired
    // Initially, we consider all elements of the array, so the above
    // statement is true regardless of input.
    uint32_t desired = 0;
    uint32_t desiredMask = 0;

    // We are looking for the top kToFind-th element when iterating over
    // digits; this count gets reduced by elimination when counting
    // successive digits
    int kToFind = k;

    // We start at the most significant digit in our radix, scanning
    // through to the least significant digit
#pragma unroll
    for (int digitPos = sizeof(float) * 8 - RADIX_BITS; digitPos >= 0;
        digitPos -= RADIX_BITS) {
        // Count radix distribution for the current position and reduce
        // across all threads
        countRadixUsingMask(
                counts,
                smem,
                desired,
                desiredMask,
                digitPos,
                sliceSize,
                withinSliceStride,
                data);

        auto found_unique = [&](int i, int count) -> bool {
            /* All threads have the same value in counts here, so all */
            /* threads will return from the function. */
            if (count == 1 && kToFind == 1) {
                /* There is a unique answer. */
                desired =
                        Bitfield::setBitfield(desired, i, digitPos, RADIX_BITS);
                desiredMask = Bitfield::setBitfield(
                        desiredMask, RADIX_MASK, digitPos, RADIX_BITS);

                /* The answer is now the unique element v such that: */
                /* (v & desiredMask) == desired */
                /* However, we do not yet know what the actual element is. We */
                /* need to perform a search through the data to find the */
                /* element that matches this pattern. */
                *topK = findPattern(
                        (float*)smem,
                        data,
                        sliceSize,
                        withinSliceStride,
                        desired,
                        desiredMask);
                return true;
            }
            return false;
        };
        auto found_non_unique = [&](int i, int count) -> bool {
            if (count >= kToFind) {
                desired =
                        Bitfield::setBitfield(desired, i, digitPos, RADIX_BITS);
                desiredMask = Bitfield::setBitfield(
                        desiredMask, RADIX_MASK, digitPos, RADIX_BITS);

                /* The top-Kth element v must now be one such that: */
                /* (v & desiredMask == desired) */
                /* but we haven't narrowed it down; we must check the next */
                /* least-significant digit */
                return true;
            }
            kToFind -= count;
            return false; // continue the loop
        };

        // All threads participate in the comparisons below to know the
        // final result
        if (order) {
            // Process in descending order
#pragma unroll
            for (int i = RADIX_SIZE - 1; i >= 0; --i) {
                int count = counts[i];
                if (found_unique(i, count)) {
                    return;
                }
                if (found_non_unique(i, count)) {
                    break;
                }
            }
        } else {
            // Process in ascending order
#pragma unroll
            for (int i = 0; i < RADIX_SIZE; ++i) {
                int count = counts[i];
                if (found_unique(i, count)) {
                    return;
                }
                if (found_non_unique(i, count)) {
                    break;
                }
            }
        }
    } // end digitPos for

    // There is no unique result, but there is a non-unique result
    // matching `desired` exactly
    *topK = TopKTypeConfig::deconvert(desired);
}

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
        bool order) {
    // Indices are limited to integer fp precision, so counts can fit in
    // int32, regardless of uint32_t
    __shared__ int smem[32];

    uint32_t slice = getLinearBlockId();

    if (slice >= numInputSlices) {
        return;
    }

    // Find the start offset for our slice
    uint32_t sliceStartIndex = IndexToOffset::get(slice, input, dim);
    uint32_t topKSliceStartIndex = IndexToOffset::get(slice, topK, dim);

    float* inputSliceStart = &input.data[sliceStartIndex];
    float* topKSliceStart = &topK.data[topKSliceStartIndex];

    // Find the k-th highest element in our input
    float topKValue = ScalarConvert<int, float>::to(0);
    radixSelect(
            inputSliceStart,
            outputSliceSize,
            inputSliceSize,
            inputWithinSliceStride,
            smem,
            &topKValue,
            order);
    const auto topKConverted = TopKTypeConfig::convert(topKValue);

    // Every value that is strictly less/greater than `pattern`
    // (depending on sort dir) in sorted int format is in the top-K.
    // The top-K value itself might not be unique.
    //
    // Since there are a variable number of elements that we see that
    // are within the top-k, we don't know at what index to write out
    // the resulting values.
    // In order to get this, we perform an exclusive prefix sum of
    // `hasTopK`. This will return the resulting index into which we
    // need to write the result, if a thread has a result.

    // All threads need to participate in the loop and the prefix sum,
    // but not necessarily in the load; hence loop bounds being rounded
    // up to a multiple of the block dim.
    uint32_t numIterations = RoundUp<uint32_t>(inputSliceSize, blockDim.x);
    uint32_t writeIndexStart = 0;

    for (uint32_t i = threadIdx.x; i < numIterations; i += blockDim.x) {
        bool inRange = (i < inputSliceSize);
        uint32_t inputOffset = i * inputWithinSliceStride;
        float v = inRange ? *(&inputSliceStart[inputOffset]) :
                    ScalarConvert<int, float>::to(0);
        const auto convertedV = TopKTypeConfig::convert(v);
        bool hasTopK;
        if (order) {
            hasTopK = inRange && (convertedV > topKConverted);
        } else {
            hasTopK = inRange && (convertedV < topKConverted);
        }

        int index;
        int carry;
        exclusiveBinaryPrefixScan<int, true>(
                smem, hasTopK, &index, &carry, AddOp());

        if (hasTopK) {
            int writeIndex = writeIndexStart + index;
            CUDA_KERNEL_ASSERT(writeIndex < outputSliceSize);

            uint32_t topKOffset = writeIndex * topKWithinSliceStride;

            topKSliceStart[topKOffset] = v;
            for (uint32_t idx = 1; idx <= extra_data; ++idx) {
                topKSliceStart[topKOffset + idx] =
                        *(&inputSliceStart[inputOffset + idx]);
            }
        }

        writeIndexStart += carry;
    }

    // We need to fill in the rest with actual == top-K values.
    // The number that we need is outputSliceSize -
    // writeIndexStart. There might be more than that number available,
    // in which case we have to choose the first seen set. We do this
    // via a prefix sum to calculate indices for writing results.
    CUDA_KERNEL_ASSERT(outputSliceSize >= writeIndexStart);
    uint32_t topKRemaining = (outputSliceSize - writeIndexStart);

    for (uint32_t i = threadIdx.x; i < numIterations; i += blockDim.x) {
        bool inRange = (i < inputSliceSize);
        uint32_t inputOffset = i * inputWithinSliceStride;
        float v = inRange ? *(&inputSliceStart[inputOffset]) :
                    ScalarConvert<int, float>::to(0);
        const auto convertedV = TopKTypeConfig::convert(v);
        bool hasTopK = inRange && (convertedV == topKConverted);

        int index;
        int carry;
        exclusiveBinaryPrefixScan<int, true>(
                smem, hasTopK, &index, &carry, AddOp());

        if (hasTopK && index < topKRemaining) {
            int writeIndex = writeIndexStart + index;
            CUDA_KERNEL_ASSERT(writeIndex < outputSliceSize);

            uint32_t topKOffset = writeIndex * topKWithinSliceStride;

            topKSliceStart[topKOffset] = v;
            for (uint32_t idx = 1; idx <= extra_data; ++idx) {
                topKSliceStart[topKOffset + idx] =
                        *(&inputSliceStart[inputOffset + idx]);
            }
        }

        if (carry >= topKRemaining) {
            break;
        }

        topKRemaining -= carry;
        writeIndexStart += carry;
    }
}

//void test()
//{
//    uint32_t N = 2, C = 1, H = 10, W = 10;
//    uint32_t HW = H * W;
//    uint32_t K = 20;
//    int dims = 3;
//    int dim = 2;
//    bool order = true;
//
//    int extra_data = 1;
//
//    float *data_on_cpu, *data_on_gpu;
//    float *topk_on_cpu, *topk_on_gpu;
//
//    data_on_cpu = (float*)malloc(N * C * (extra_data + 1) *HW * sizeof(float));
//    topk_on_cpu = (float*)malloc(N * C * (extra_data + 1) * K * sizeof(float));
//    cudaMalloc((void**)&data_on_gpu, N * C * (extra_data + 1) *HW * sizeof(float));
//    cudaMalloc((void**)&topk_on_gpu, N * C * (extra_data + 1) * K * sizeof(float));
//
//    uint32_t isz[3] = {N, C, (extra_data + 1) * HW};
//    uint32_t ist[3] = {C * (extra_data + 1) * HW, (extra_data + 1) * HW, 1};
//    TensorInfo input(data_on_gpu, dims, isz, ist);
//
//    uint32_t osz[3] = {N, C, (extra_data + 1) * K};
//    uint32_t ost[3] = {C * ((extra_data + 1) * K), (extra_data + 1) * K, 1};
//    TensorInfo output(topk_on_gpu, dims, osz, ost);
//
//    // fill in the host memory with data
//    printf("\ndata: \n");
//    int idx;
//    float val;
//    for (int n = 0; n < N; ++n) {
//        printf("batch %d: \n", n);
//        for (int c = 0; c < C; ++c) {
//            printf("channel %d: \n", c);
//            for (int h = 0; h < H; ++h) {
//                for (int w = 0; w < W; ++w) {
//                    idx = (n * C + c) * HW + w * H + h;
//                    val = idx / 8 + idx % 7;
//                    idx *= (extra_data + 1);
//                    data_on_cpu[idx] = val;
//                    printf("%f \t", val);
//
//                    for (int m = 1; m <= extra_data; ++m) {
//                        data_on_cpu[idx + m] = val + 0.314;
//                    }
//                }
//                printf("\n");
//            }
//        }
//    }
//
//    cudaMemcpy(data_on_gpu, data_on_cpu,
//               N * C * (extra_data + 1) * HW * sizeof(float), cudaMemcpyHostToDevice);
//
//    input.reduceDim(dim);
//    output.reduceDim(dim);
//
//    uint32_t inputSlices = 1;
//    for (int i = 0; i < input.dims; ++i)
//        inputSlices *= input.sizes[i];
//
//    uint32_t outputSlices = 1;
//    for (int i = 0; i < output.dims; ++i)
//        outputSlices *= output.sizes[i];
//
//    dim3 grid(N * C);
//    dim3 block(std::min(RoundUp<uint32_t>(HW, 32), (uint32_t) 1024));
//
//    gatherTopK<<<grid, block>>>(
//            input,
//            HW,
//            K,
//            inputSlices,
//            input.strides[dim] * (extra_data + 1),
//            output,
//            outputSlices,
//            output.strides[dim] * (extra_data + 1),
//            extra_data,
//            dim,
//            order
//    );
//
//    cudaMemcpy(topk_on_cpu, topk_on_gpu,
//               N * C * (extra_data + 1) * K * sizeof(float), cudaMemcpyDeviceToHost);
//
//    printf("\ntopk: \n");
//    for (int n = 0; n < N; ++n) {
//        printf("batch %d: \n", n);
//        for (int c = 0; c < C; ++c) {
//            printf("channel %d: \n", c);
//            for (int k = 0; k < K; ++k) {
//                idx = ((n * C + c) * K + k) * (extra_data + 1);
//                printf("%f \t", topk_on_cpu[idx]);
//
//                for (int m = 1; m <= extra_data; ++m) {
//                    printf("%f \t", topk_on_cpu[idx + m]);
//                }
//            }
//            printf("\n");
//        }
//    }
//
//    CHECK(cudaDeviceSynchronize());
//}

void det_topk(
        float* input,
        float* output,
        const int resCount,
        const int topk,
        const int extra_dim,
        const bool order) {
    uint32_t isz[1] = { (extra_dim + 1) * resCount };
    uint32_t ist[1] = { 1 };
    TensorInfo topk_input_info(input, 1, isz, ist);

    uint32_t osz[1] = { (extra_dim + 1) * topk };
    uint32_t ost[1] = { 1 };
    TensorInfo topk_output_info(output, 1, osz, ost);

    dim3 block(std::min(RoundUp<uint32_t>(resCount, 32), (uint32_t) 1024));
    gatherTopK<<<1, block>>>(
            topk_input_info,
            resCount,
            topk,
            isz[0],
            extra_dim + 1,
            topk_output_info,
            osz[0],
            extra_dim + 1,
            extra_dim,
            0,
            order);
}
