#include <ATen/hipdnn/cudnn_compat.h>

#if AT_HIPDNN_ENABLED()

#include <ATen/hip/Exceptions.h>
#include <ATen/hip/detail/DeviceThreadHandles.h>
#include <c10/hip/HIPStream.h>

namespace at::native {
namespace {

void checkHipdnnStatus(cudnnStatus_t status, const char* what) {
  TORCH_CHECK(
      status == CUDNN_STATUS_SUCCESS,
      "hipDNN ",
      what,
      " failed: ",
      cudnnGetErrorString(status));
}

void createHipdnnHandle(cudnnHandle_t* handle) {
  checkHipdnnStatus(cudnnCreate(handle), "cudnnCreate");
}

void destroyHipdnnHandle(cudnnHandle_t /* handle */) {
  // Left deliberately empty, matching ATen/miopen/Handle.cpp and
  // ATen/cudnn/Handle.cpp: at process teardown the GPU context can already be
  // gone by the time these pooled handles are reclaimed.
}

using HipdnnPoolType = at::cuda::DeviceThreadHandlePool<
    cudnnHandle_t,
    createHipdnnHandle,
    destroyHipdnnHandle>;

} // namespace

cudnnHandle_t getCudnnHandle() {
  c10::DeviceIndex device = 0;
  AT_CUDA_CHECK(at::cuda::GetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<HipdnnPoolType>();
  thread_local std::unique_ptr<HipdnnPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  checkHipdnnStatus(
      cudnnSetStream(handle, at::cuda::getCurrentCUDAStream()),
      "cudnnSetStream");
  return handle;
}

} // namespace at::native

#endif // AT_HIPDNN_ENABLED()
