// main_training.cpp
// Sequential DL-LES Training Loop
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

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
}