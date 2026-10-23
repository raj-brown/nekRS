#pragma once
#include <torch/torch.h>

struct Net : torch::nn::Module {
  static constexpr int dim_in = 9;
  static constexpr int dim_hidden = 50;
  static constexpr int dim_out = 9;

  torch::nn::Linear W1{nullptr}, W2{nullptr}, W3{nullptr}, W4{nullptr}, W5{nullptr}, W6{nullptr};
  Net()
  {
    W1 = register_module("W1", torch::nn::Linear(dim_in, dim_hidden));
    W2 = register_module("W2", torch::nn::Linear(dim_hidden, dim_hidden));
    W3 = register_module("W3", torch::nn::Linear(dim_hidden, dim_hidden));
    W4 = register_module("W4", torch::nn::Linear(dim_hidden, dim_out));
    W5 = register_module("W5", torch::nn::Linear(dim_in, dim_hidden));
    W6 = register_module("W6", torch::nn::Linear(dim_in, dim_hidden));
  }
  torch::Tensor forward(torch::Tensor z)
  {
    auto H1 = torch::tanh(W1->forward(z));
    auto H2 = torch::tanh(W2->forward(H1));
    auto G1 = torch::tanh(W5->forward(z));
    auto H3 = G1*H2;
    auto H4 = torch::tanh(W3->forward(H3));
    auto G2 = torch::tanh(W6->forward(z));
    auto H5 = G2 * H4;
    return W4->forward(H5);

  }

};

inline constexpr torch::ScalarType kModelDtype = torch::kFloat64;
