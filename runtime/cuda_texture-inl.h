//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "cuda_util.h"

namespace vox {

template<typename T, size_t N, typename Derived>
CudaTextureBase<T, N, Derived>::CudaTextureBase() {}

template<typename T, size_t N, typename Derived>
CudaTextureBase<T, N, Derived>::CudaTextureBase(const TensorView<const T, N> &view) {
    set(view);
}

template<typename T, size_t N, typename Derived>
CudaTextureBase<T, N, Derived>::CudaTextureBase(const CudaTensorView<const T, N> &view) {
    set(view);
}

template<typename T, size_t N, typename Derived>
CudaTextureBase<T, N, Derived>::CudaTextureBase(const CudaTextureBase &other) {
    set(static_cast<const Derived &>(other));
}

template<typename T, size_t N, typename Derived>
CudaTextureBase<T, N, Derived>::CudaTextureBase(CudaTextureBase &&other) noexcept {
    *this = std::move(other);
}

template<typename T, size_t N, typename Derived>
CudaTextureBase<T, N, Derived>::~CudaTextureBase() {
    clear();
}

template<typename T, size_t N, typename Derived>
void CudaTextureBase<T, N, Derived>::clear() {
    if (_array != nullptr) {
        cudaFreeArray(_array);
        _array = nullptr;
    }

    if (_tex != 0) {
        cudaDestroyTextureObject(_tex);
        _tex = 0;
    }

    _shape = CudaStdArray<size_t, N>{};
}

template<typename T, size_t N, typename Derived>
void CudaTextureBase<T, N, Derived>::set(const TensorView<const T, N> &view) {
    static_cast<Derived *>(this)->set(view, cudaMemcpyHostToDevice);
}

template<typename T, size_t N, typename Derived>
void CudaTextureBase<T, N, Derived>::set(const CudaTensorView<const T, N> &view) {
    static_cast<Derived *>(this)->set(view, cudaMemcpyDeviceToDevice);
}

template<typename T, size_t N, typename Derived>
void CudaTextureBase<T, N, Derived>::set(const Derived &other) {
    static_cast<Derived *>(this)->set(other);
}

template<typename T, size_t N, typename Derived>
cudaTextureObject_t CudaTextureBase<T, N, Derived>::textureObject() const {
    return _tex;
}

template<typename T, size_t N, typename Derived>
CudaTextureBase<T, N, Derived> &CudaTextureBase<T, N, Derived>::operator=(
    const CudaTextureBase &other) {
    set(other);
    return *this;
}

template<typename T, size_t N, typename Derived>
cudaTextureObject_t CudaTextureBase<T, N, Derived>::createTexture(
    cudaArray_t array, cudaTextureFilterMode filterMode,
    bool shouldNormalizeCoords) {
    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = filterMode;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = shouldNormalizeCoords;

    // Create texture object
    cudaTextureObject_t tex = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));

    return tex;
}

//

template<typename T>
CudaTexture1<T>::CudaTexture1() : Base() {}

template<typename T>
CudaTexture1<T>::CudaTexture1(const ConstTensorView1<T> &view) : Base(view) {}

template<typename T>
CudaTexture1<T>::CudaTexture1(const ConstCudaTensorView1<T> &view)
    : Base(view) {}

template<typename T>
CudaTexture1<T>::CudaTexture1(const CudaTexture1 &other) : Base(other) {}

template<typename T>
CudaTexture1<T>::CudaTexture1(CudaTexture1 &&other) noexcept : Base() {
    *this = std::move(other);
}

template<typename T>
size_t CudaTexture1<T>::shape() const {
    return _shape[0];
}

template<typename T>
CudaTexture1<T> &CudaTexture1<T>::operator=(CudaTexture1 &&other) noexcept {
    clear();
    std::swap(_shape, other._shape);
    std::swap(_array, other._array);
    std::swap(_tex, other._tex);
    return *this;
}

template<typename T>
void CudaTexture1<T>::resize(const CudaStdArray<size_t, 1> &size) {
    if (size[0] == 0) {
        clear();
        return;
    }

    if (_shape[0] != size[0]) {
        clear();

        _shape = size;

        // Allocate CUDA array in device memory
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
        JET_CUDA_CHECK(cudaMallocArray(&_array, &channelDesc, _shape[0], 1));
    }
}

template<typename T>
template<typename View>
void CudaTexture1<T>::set(const View &view, cudaMemcpyKind memcpyKind) {
    resize(view.shape());

    if (view.length() == 0) {
        return;
    }

    // Copy to device memory to CUDA array
    JET_CUDA_CHECK(cudaMemcpyToArray(_array, 0, 0, view.data(),
                                     sizeof(T) * view.length(), memcpyKind));

    // Create texture
    _tex = createTexture(_array, cudaFilterModeLinear, false);
}

template<typename T>
void CudaTexture1<T>::set(const CudaTexture1 &other) {
    resize(other._shape);

    if (other._shape[0] == 0) {
        return;
    }

    // Copy to device memory to CUDA array
    JET_CUDA_CHECK(cudaMemcpyArrayToArray(_array, 0, 0, other._array, 0, 0,
                                          sizeof(T) * other._shape[0],
                                          cudaMemcpyDeviceToDevice));

    // Create texture
    _tex = createTexture(_array, cudaFilterModeLinear, false);
}

//

template<typename T>
CudaTexture2<T>::CudaTexture2() : Base() {}

template<typename T>
CudaTexture2<T>::CudaTexture2(const ConstTensorView2<T> &view) : Base(view) {}

template<typename T>
CudaTexture2<T>::CudaTexture2(const ConstCudaTensorView2<T> &view)
    : Base(view) {}

template<typename T>
CudaTexture2<T>::CudaTexture2(const CudaTexture2 &other) : Base(other) {}

template<typename T>
CudaTexture2<T>::CudaTexture2(CudaTexture2 &&other) noexcept : Base() {
    *this = std::move(other);
}

template<typename T>
CudaStdArray<size_t, 2> CudaTexture2<T>::shape() const {
    // TODO: Size2 should be specialization (or alias) of Size<N=2>
    return CudaStdArray<size_t, 2>(width(), height());
}

template<typename T>
size_t CudaTexture2<T>::width() const {
    return _shape[0];
}

template<typename T>
size_t CudaTexture2<T>::height() const {
    return _shape[1];
}

template<typename T>
CudaTexture2<T> &CudaTexture2<T>::operator=(CudaTexture2 &&other) noexcept {
    clear();
    std::swap(_shape, other._shape);
    std::swap(_array, other._array);
    std::swap(_tex, other._tex);
    return *this;
}

template<typename T>
void CudaTexture2<T>::resize(const CudaStdArray<size_t, 2> &size) {
    if (size[0] * size[1] == 0) {
        clear();
        return;
    }

    if (_shape != size) {
        clear();

        _shape = size;

        // Allocate CUDA array in device memory
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
        JET_CUDA_CHECK(
            cudaMallocArray(&_array, &channelDesc, _shape[0], _shape[1]));
    }
}

template<typename T>
template<typename View>
void CudaTexture2<T>::set(const View &view, cudaMemcpyKind memcpyKind) {
    // TODO: Size2 should be specialization (or alias) of Size<N=2>
    resize(CudaStdArray<size_t, 2>(view.width(), view.height()));

    if (view.width() * view.height() == 0) {
        return;
    }

    // Copy to device memory to CUDA array
    JET_CUDA_CHECK(cudaMemcpyToArray(_array, 0, 0, view.data(),
                                     sizeof(T) * view.width() * view.height(),
                                     memcpyKind));

    // Create texture
    _tex = createTexture(_array, cudaFilterModeLinear, false);
}

template<typename T>
void CudaTexture2<T>::set(const CudaTexture2 &other) {
    // TODO: Size2 should be specialization (or alias) of Size<N=2>
    resize(other._shape);

    if (other.width() * other.height() == 0) {
        return;
    }

    // Allocate CUDA array in device memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    JET_CUDA_CHECK(
        cudaMallocArray(&_array, &channelDesc, other.width(), other.height()));

    // Copy to device memory to CUDA array
    JET_CUDA_CHECK(cudaMemcpyArrayToArray(
        _array, 0, 0, other._array, 0, 0,
        sizeof(T) * other.width() * other.height(), cudaMemcpyDeviceToDevice));

    // Create texture
    _tex = createTexture(_array, cudaFilterModeLinear, false);
}

//

template<typename T>
CudaTexture3<T>::CudaTexture3() : Base() {}

template<typename T>
CudaTexture3<T>::CudaTexture3(const ConstTensorView3<T> &view) : Base(view) {}

template<typename T>
CudaTexture3<T>::CudaTexture3(const ConstCudaTensorView3<T> &view)
    : Base(view) {}

template<typename T>
CudaTexture3<T>::CudaTexture3(const CudaTexture3 &other) : Base(other) {}

template<typename T>
CudaTexture3<T>::CudaTexture3(CudaTexture3 &&other) noexcept : Base() {
    *this = std::move(other);
}

template<typename T>
CudaStdArray<size_t, 3> CudaTexture3<T>::shape() const {
    // TODO: Size3 should be specialization (or alias) of Size<N=3>
    return CudaStdArray<size_t, 3>(width(), height(), depth());
}

template<typename T>
size_t CudaTexture3<T>::width() const {
    return _shape[0];
}

template<typename T>
size_t CudaTexture3<T>::height() const {
    return _shape[1];
}

template<typename T>
size_t CudaTexture3<T>::depth() const {
    return _shape[2];
}

template<typename T>
CudaTexture3<T> &CudaTexture3<T>::operator=(CudaTexture3 &&other) noexcept {
    clear();
    std::swap(_shape, other._shape);
    std::swap(_array, other._array);
    std::swap(_tex, other._tex);
    return *this;
}

template<typename T>
void CudaTexture3<T>::resize(const CudaStdArray<size_t, 3> &size) {
    if (size[0] * size[1] * size[2] == 0) {
        clear();
        return;
    }

    if (_shape != size) {
        clear();

        _shape = size;

        // Allocate CUDA array in device memory
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
        // Note that the first param is not in bytes in this non-linear memory
        // case.
        cudaExtent ext = make_cudaExtent(size[0], size[1], size[2]);
        JET_CUDA_CHECK(cudaMalloc3DArray(&_array, &channelDesc, ext));
    }
}

template<typename T>
template<typename View>
void CudaTexture3<T>::set(const View &view, cudaMemcpyKind memcpyKind) {
    // TODO: Size3 should be specialization (or alias) of Size<N=3>
    CudaStdArray<size_t, 3> size(view.width(), view.height(), view.depth());
    resize(size);

    if (view.width() * view.height() * view.depth() == 0) {
        return;
    }

    // Copy to device memory to CUDA array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr =
        make_cudaPitchedPtr((void *)view.data(), view.width() * sizeof(T),
                            view.width(), view.height());
    copyParams.dstArray = _array;
    copyParams.extent = make_cudaExtent(size[0], size[1], size[2]);
    copyParams.kind = memcpyKind;
    CUDA_CHECK(cudaMemcpy3D(&copyParams));

    // Create texture
    _tex = createTexture(_array, cudaFilterModeLinear, false);
}

template<typename T>
void CudaTexture3<T>::set(const CudaTexture3 &other) {
    // TODO: Size3 should be specialization (or alias) of Size<N=3>
    CudaStdArray<size_t, 3> size{other.width(), other.height(), other.depth()};
    resize(size);

    if (other.width() * other.height() * other.depth() == 0) {
        return;
    }

    // Copy to device memory to CUDA array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcArray = other._array;
    copyParams.dstArray = _array;
    copyParams.extent = make_cudaExtent(size[0], size[1], size[2]);
    copyParams.kind = cudaMemcpyDeviceToDevice;
    CUDA_CHECK(cudaMemcpy3D(&copyParams));

    // Create texture
    _tex = createTexture(_array, cudaFilterModeLinear, false);
}

}// namespace vox
