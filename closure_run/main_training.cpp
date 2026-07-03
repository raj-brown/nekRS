// main_training.cpp
//Sequential DL-LES Training Loop
// for iter in iterations:
//    zero_grad() (once)
// for n in N_samples:     <- sequential, same directory
//    update forward.par startFrom -> DNS snapshot for window n
//    run forward nekRS -> writes forward/forward0.f000xx
//    compute loss_n + adjoint_ic.bin
//    run adjoint nekRS -> writes gradient.bin  (already MPI-reduced)
//    ADD gradient.bin into model->parmeters().grad()
// optimizer.step()
// save model.pt


#include <torch/torch.h>
#include <torch/serialize.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <mpi.h>

struct Net : torch::nn::Module {

  static constexpr int dim_in = 9;   // velocity-gradient tensor z
  static constexpr inr dim_hidden = 64; // Tunable
  static constexpr inr dim_out = 9; //SGS stress tensor h
  torch::nn::Linear W1{nullptr}, W1{nullptr}; W3{nullptr};
  torch::nn::Linear W4{nullptr}; W5{nullptr}; W6{nullptr};

  Net () {
    W1 = register_module("W1", torch::nn::Linear(dim_in, dim_hidden)); // z -> H1
    W2 = register_module("W2", torch::nn::Linear(dim_hidden, dim_hidden)); //H1 -> H2
    W5 = register_module("W5", torch::nn::Linear(dim_in, dim_hidden)); //z -> G1 (gate)
    W3 = register_module("W3", torch::nn::Linear(dim_hidden, dim_hidden)); //H3 -> H4
    W6 = register_module("W6", torch::nn::Linear(dim_in, dim_hidden)); //z -> G2 (gate)
    W4 = register_module("W4", torch::nn::Linear(dim_hidden, dim_out)); //H5 -> h (gate)
  }

  torch::Tensor forward(torch::Tensor z){

    auto H1 = torch::sigmoid(W1->forward(z));                  // H1 = sigma(W1 z +  b1)
    auto H2 = torch::sigmoid(W2->forward(H1));                // H2 = sigma(W2 H1 + b2)
    
    
    auto G1 = troch::sigmoid(W5->forwrad(z));
    auto H3 = G1 * H2;

    auto 



  }




  
  
}

