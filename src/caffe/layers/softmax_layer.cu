// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>
#include <thrust/device_vector.h>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Softmax Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Softmax Layer takes a single blob as output.";
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  sum_multiplier_.Reshape(1, bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  for (int i = 0; i < sum_multiplier_.count(); ++i) {
    multiplier_data[i] = 1.;
  }
  scale_.Reshape(bottom[0]->num(), 1, 1, 1);
};

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  memcpy(top_data, bottom_data, sizeof(Dtype) * bottom[0]->count());
  // we need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < num; ++i) {
    scale_data[i] = bottom_data[i*dim];
    for (int j = 0; j < dim; ++j) {
      scale_data[i] = max(scale_data[i], bottom_data[i * dim + j]);
    }
  }
  // subtraction
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
    scale_data, sum_multiplier_.cpu_data(), 1., top_data);
  // Perform exponentiation
  caffe_exp<Dtype>(num * dim, top_data, top_data);
  // sum after exp
  caffe_cpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., top_data,
      sum_multiplier_.cpu_data(), 0., scale_data);
  // Do division
  for (int i = 0; i < num; ++i) {
    caffe_scal<Dtype>(dim, Dtype(1.) / scale_data[i], top_data + i * dim);
  }

  // LOG(INFO) << "SoftmaxLayer: " << (*top)[0]->num() << ", " << (*top)[0]->channels() << ", " << (*top)[0]->height() << ", " << (*top)[0]->width();
  // LOG(INFO) << "SoftmaxLayer: " << ((*top)[0])->cpu_data()[0] << ", " << ((*top)[0])->cpu_data()[1] << ", " << ((*top)[0])->cpu_data()[2];
}

template <typename Dtype>
__global__ void kernel_get_max(const int num, const int dim,
    const Dtype* data, Dtype* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < num) {
    Dtype maxval = -FLT_MAX;
    for (int i = 0; i < dim; ++i) {
      maxval = max(data[index * dim + i], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_softmax_div(const int num, const int dim,
    const Dtype* scale, Dtype* data) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < num * dim) {
    int n = index / dim;
    data[index] /= scale[n];
  }
}

template <typename Dtype>
__global__ void kernel_exp(const int num, const Dtype* data, Dtype* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < num) {
    out[index] = exp(data[index]);
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  CUDA_CHECK(cudaMemcpy(top_data, bottom_data,
      sizeof(Dtype) * bottom[0]->count(), cudaMemcpyDeviceToDevice));
  // we need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  // Compute max
  kernel_get_max<Dtype><<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(
      num, dim, bottom_data, scale_data);
  // subtraction
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      scale_data, sum_multiplier_.gpu_data(), 1., top_data);
  // Perform exponentiation
  kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(num * dim), CAFFE_CUDA_NUM_THREADS>>>(
      num * dim, top_data, top_data);
  // sum after exp
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num, dim, 1., top_data,
      sum_multiplier_.gpu_data(), 0., scale_data);
  // Do division
  kernel_softmax_div<Dtype><<<CAFFE_GET_BLOCKS(num * dim), CAFFE_CUDA_NUM_THREADS>>>(
      num, dim, scale_data, top_data);

  // LOG(INFO) << "SoftmaxLayer: " << (*top)[0]->num() << ", " << (*top)[0]->channels() << ", " << (*top)[0]->height() << ", " << (*top)[0]->width();
  // LOG(INFO) << "SoftmaxLayer: " << ((*top)[0])->gpu_data()[0] << ", " << ((*top)[0])->gpu_data()[1] << ", " << ((*top)[0])->gpu_data()[2];
}

template <typename Dtype>
Dtype SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int num = top[0]->num();
  int dim = top[0]->count() / top[0]->num();
  memcpy(bottom_diff, top_diff, sizeof(Dtype) * top[0]->count());
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff
  for (int i = 0; i < num; ++i) {
    scale_data[i] = caffe_cpu_dot<Dtype>(dim, top_diff + i * dim,
        top_data + i * dim);
  }
  // subtraction
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      scale_data, sum_multiplier_.cpu_data(), 1., bottom_diff);
  // elementwise multiplication
  caffe_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
  return Dtype(0);
}

// TODO(Yangqing): implement the GPU version of softmax.
template <typename Dtype>
Dtype SoftmaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  int num = top[0]->num();
  int dim = top[0]->count() / top[0]->num();
  CUDA_CHECK(cudaMemcpy(bottom_diff, top_diff,
      sizeof(Dtype) * top[0]->count(), cudaMemcpyDeviceToDevice));
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff
  // cuda dot returns the result to cpu, so we temporarily change the pointer
  // mode
  CUBLAS_CHECK(cublasSetPointerMode(Caffe::cublas_handle(),
      CUBLAS_POINTER_MODE_DEVICE));
  Dtype* scale_data = scale_.mutable_gpu_data();
  for (int i = 0; i < num; ++i) {
    caffe_gpu_dot<Dtype>(dim, top_diff + i * dim,
        top_data + i * dim, scale_data + i);
  }
  CUBLAS_CHECK(cublasSetPointerMode(Caffe::cublas_handle(),
      CUBLAS_POINTER_MODE_HOST));
  // subtraction
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      scale_.gpu_data(), sum_multiplier_.gpu_data(), 1., bottom_diff);
  // elementwise multiplication
  caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
  return Dtype(0);
}

INSTANTIATE_CLASS(SoftmaxLayer);


}  // namespace caffe
