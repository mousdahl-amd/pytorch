#include <ATen/hipdnn/Handle.h>

#if AT_HIPDNN_ENABLED()

#include <ATen/hip/Exceptions.h>
#include <ATen/hip/detail/DeviceThreadHandles.h>
#include <c10/hip/HIPStream.h>

namespace at::native {
namespace {

void checkHipdnnStatus(hipdnnStatus_t status, const char* what) {
  TORCH_CHECK(
      status == HIPDNN_STATUS_SUCCESS,
      "hipDNN ",
      what,
      " failed: ",
      hipdnnGetErrorString(status));
}

void createHipdnnHandle(hipdnnHandle_t* handle) {
  checkHipdnnStatus(hipdnnCreate(handle), "hipdnnCreate");
}

void destroyHipdnnHandle(hipdnnHandle_t /* handle */) {
  // Left deliberately empty, matching ATen/cudnn/Handle.cpp: at process
  // teardown the GPU context can already be gone by the time these pooled
  // handles are reclaimed.
}

using HipdnnPoolType = at::cuda::DeviceThreadHandlePool<
    hipdnnHandle_t,
    createHipdnnHandle,
    destroyHipdnnHandle>;

} // namespace

hipdnnHandle_t getHipdnnHandle() {
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
      hipdnnSetStream(handle, at::cuda::getCurrentCUDAStream()),
      "hipdnnSetStream");
  return handle;
}

} // namespace at::native

#endif // AT_HIPDNN_ENABLED()
