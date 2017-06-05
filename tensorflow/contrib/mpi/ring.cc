#define EIGEN_USE_THREADS

#include "tensorflow/contrib/mpi/ring.h"

namespace tensorflow {
namespace contrib {
namespace mpi {

using CPUDevice = Eigen::ThreadPoolDevice;

extern template MPI_Datatype MPIType<float>();
extern template MPI_Datatype MPIType<int>();
extern template MPI_Datatype MPIType<long long>();
extern template DataType TensorFlowDataType<float>();
extern template DataType TensorFlowDataType<int>();
extern template DataType TensorFlowDataType<long long>();


// Generate all necessary specializations for RingAllreduce.
template Status RingAllreduce<CPUDevice, int>(OpKernelContext*, Tensor&, Tensor*);
template Status RingAllreduce<CPUDevice, long long>(OpKernelContext*, Tensor&, Tensor*);
template Status RingAllreduce<CPUDevice, float>(OpKernelContext*, Tensor&, Tensor*);

// Generate all necessary specializations for RingAllgather.
template Status RingAllgather<CPUDevice, int>(
    OpKernelContext*, Tensor&, Tensor*, std::vector<size_t>&);
template Status RingAllgather<CPUDevice, long long>(
    OpKernelContext*, Tensor&, Tensor*, std::vector<size_t>&);
template Status RingAllgather<CPUDevice, float>(
    OpKernelContext*, Tensor&, Tensor*, std::vector<size_t>&);

// Copy data on a CPU using a straight-forward memcpy.
template<> void CopyTensorData<CPUDevice>(void* dst, void* src, size_t size) {
    std::memcpy(dst, src, size);
};

// Accumulate values on a CPU.
template<> void AccumulateTensorData<CPUDevice, float>(
        float* dst, float* src, size_t size) {
    for (unsigned int i = 0; i < size; i++) {
        dst[i] += src[i];
    }
};
template<> void AccumulateTensorData<CPUDevice, int>(
        int* dst, int* src, size_t size) {
    for (unsigned int i = 0; i < size; i++) {
        dst[i] += src[i];
    }
}
template<> void AccumulateTensorData<CPUDevice, long long>(
        long long* dst, long long* src, size_t size) {
    for (unsigned int i = 0; i < size; i++) {
        dst[i] += src[i];
    }
}

}
}
}
