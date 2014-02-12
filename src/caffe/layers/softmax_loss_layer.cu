// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "SoftmaxLoss Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 0) << "SoftmaxLoss Layer takes no blob as output.";
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
};

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  // LOG(INFO) << "SoftmaxWithLossLayer: " << (int)prob_.num() << ", " << (int)prob_.channels() << ", " << (int)prob_.height() << ", " << (int)prob_.width();
  // LOG(INFO) << "SoftmaxWithLossLayer: " << (float)(prob_.cpu_data()[0]) << ", " << (float)(prob_.cpu_data()[1]) << ", " << (float)(prob_.cpu_data()[2]);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  // LOG(INFO) << "SoftmaxWithLossLayer: " << (int)prob_.num() << ", " << (int)prob_.channels() << ", " << (int)prob_.height() << ", " << (int)prob_.width();
  // LOG(INFO) << "SoftmaxWithLossLayer: " << (float)(prob_.gpu_data()[0]) << ", " << (float)(prob_.gpu_data()[1]) << ", " << (float)(prob_.gpu_data()[2]);
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  batch_accuracy_ = 0;
  // First, compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* prob_data = prob_.cpu_data();  // bottom_data
  memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
  const Dtype* label = (*bottom)[1]->cpu_data();  // bottom_label
  int num = prob_.num();
  int dim = prob_.count() / num;
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    // Accuracy
    Dtype maxval = -FLT_MAX;
    int max_id = 0;
    for (int j = 0; j < dim; ++j) {
      if (prob_data[i * dim + j] > maxval) {
        maxval = prob_data[i * dim + j];
        max_id = j;
      }
    }
    if (max_id == (int)label[i]) {
      ++batch_accuracy_;
    }

    bottom_diff[i * dim + static_cast<int>(label[i])] -= 1;
    loss += -log(max(prob_data[i * dim + static_cast<int>(label[i])], FLT_MIN));
  }

  batch_accuracy_ /= num;
  // LOG(INFO) << "Training Accuracy in Layer:" << batch_accuracy_;

  // Scale down gradient
  caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
  return loss / num;
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  // TODO(Yangqing): implement the GPU version of softmax.
  return Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_CLASS(SoftmaxWithLossLayer);


}  // namespace caffe
