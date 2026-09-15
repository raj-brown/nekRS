

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

  static constexpr int dim_in = 9;      // velocity-gradient tensor z
  static constexpr inr dim_hidden = 64; // Tunable
  static constexpr inr dim_out = 9;     // SGS stress tensor h
  torch::nn::Linear W1{nullptr}, W1{nullptr};
  W3{nullptr};
  torch::nn::Linear W4{nullptr};
  W5{nullptr};
  W6{nullptr};

  Net()
  {
    W1 = register_module("W1", torch::nn::Linear(dim_in, dim_hidden));     // z -> H1
    W2 = register_module("W2", torch::nn::Linear(dim_hidden, dim_hidden)); // H1 -> H2
    W5 = register_module("W5", torch::nn::Linear(dim_in, dim_hidden));     // z -> G1 (gate)
    W3 = register_module("W3", torch::nn::Linear(dim_hidden, dim_hidden)); // H3 -> H4
    W6 = register_module("W6", torch::nn::Linear(dim_in, dim_hidden));     // z -> G2 (gate)
    W4 = register_module("W4", torch::nn::Linear(dim_hidden, dim_out));    // H5 -> h (gate)
  }

  torch::Tensor forward(torch::Tensor z)
  {

    auto H1 = torch::sigmoid(W1->forward(z));  // H1 = sigma(W1 z +  b1)
    auto H2 = torch::sigmoid(W2->forward(H1)); // H2 = sigma(W2 H1 + b2)

    auto G1 = torch::sigmoid(W5->forward(z)); // G1 = sigma(W5 z + b5)  (gate)
    auto H3 = G1 * H2;                        // H3 = G1 * H2

    auto H4 = torch::sigmoid(W3->forward(H3)); // H4 = sigma(W3 H3 + b3)

    auto G2 = torch::sigmoid(W6->forward(z)); // G2 = sigma(W6 z + b6)
    auto H5 = G2 * H4;                        // H5 = G2 * H4

    return W4->forward(H5); // h = W4 H5 + b4
  }

}

// Run command on terminal and check return code. If non-zero, abort MPI.
void runScript(const std::string &script, int nMPIRanks)

{
  std::string cmd = "bash " + script + " " + std::to_string(nMPIRanks);
  int ret = std::system(cmd.c_str());
  ;

  if (ret != 0) {
    std::cerr << "[Training] ERROR: " << script << " failed (code=" << ret << ")" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

void setParKey(const std::string &parFile, const std::string &key, const std::string &value)
{
  std::ifstream infile(parFile);
  if (!infile.is_open()) {
    std::cerr << "[Training] ERROR: Could not open file " << parFile << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  std::ostringstream oss;
  std::string line;
  bool keyFound = false;

  while (std::getline(infile, line)) {
    if (line.find(key) != std::string::npos) {
      oss << key << " = " << value << "\n";
      keyFound = true;
    } else {
      oss << line << "\n";
    }
  }

  infile.close();

  if (!keyFound) {
    oss << key << " = " << value << "\n"; // Add the key-value pair if not found
  }

  std::ofstream outfile(parFile);
  if (!outfile.is_open()) {
    std::cerr << "[Training] ERROR: Could not write to file " << parFile << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  outfile << oss.str();
  outfile.close();
}

// Read File (forward0.f000xx)
std::string fldFileName(const std::string &base, int step)
{
  std::ostringstream oss;
  oss << base << "0.f" << std::setw(5) << std::setfill('0') << step;
  return oss.str();
}

float readLoss(const std::string &lossFile)
{

  std::ifstream lf(lossFile);
  if (!lf.good()) {
    std::cerr << "[Debug] ERROR: Could not open loss file " << lossFile << "not found\n" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  float J = 0.0f

            lf >>
            J;
  std::cout << "[Debug] Loss = " << J << "\n" << std::endl;
  return J;
}

void perturbFlat(std::shared_ptr<Net> &model, int flatIdx, float delta)
{
  int running = 0;
  for (auto &param : model->parameters()) {
    int n = param.numel();
    if (flatIdx < running + n) {
      auto flat = p.data().view(-1);
      flat[flatIdx - running] += delta;
      std::cout << "[Debug] Perturbed param[" << flatIdx << "] by " << delta << " (tensor " << running << ")"
                << std::endl;
    }
    running += n;
  }
}

// Prints shape and first few elements of each parameter tensor in the model
void printModelSummary(const std::shared_ptr<Net> &model)
{
  std::cout << "\n[Debug] Model Paramets Summary:\n";
  int totalParams = 0;
  int flatIdx = 0;
  for (const auto &param : model->named_parameters()) {
    auto sizes = param.value().sizes();
    int n = p.value().sizes();
    auto flat = param.value().cpu().view(-1);

    std::cout << " " << std::setw(6) << flatIdx << " " << std::setw(20) << p.key() << " shape=";
    for (const auto &s : sizes) {
      std::cout << s << "x";
    }

    std::cout << " numel=" << n << " first few elements: [";
    for (int i = 0; i < std::min(5, flat.size(0)); ++i) {
      std::cout << flat[i].item<float>() << " ";
    }
    std::cout << "...]\n";
    totalParams += n;
    flatIdx += n;
  }
  std::cout << "[Debug] Total number of parameters: " << totalParams << "\n" << std::endl;
}

// Prints first few values of flat gradient tensor

void printGradSummary(const std::shared_ptr<Net> &model)
{
  std::cout << "\n[Debug] Adjoint Gradient Summary:\n";
  int totalGrad = 0;
  int flatIdx = 0;
  for (const auto &param : model->named_parameters()) {
    if (param.value().grad().defined()) {
      auto sizes = param.value().sizes();
      int n = param.value().numel();
      auto flatGrad = param.value().grad().cpu().view(-1);

      std::cout << " " << std::setw(6) << flatIdx << " " << std::setw(20) << param.key() << " shape=";
      for (const auto &s : sizes) {
        std::cout << s << "x";
      }

      std::cout << " numel=" << n << " first few gradient elements: [";
      for (int i = 0; i < std::min(5, flatGrad.size(0)); ++i) {
        std::cout << flatGrad[i].item<float>() << " ";
      }
      std::cout << "...]\n";
      totalGrad += n;
      flatIdx += n;
    } else {
      std::cout << "[Debug] Gradient for parameter " << param.key() << " is not defined.\n";
    }
  }
  std::cout << "[Debug] Total number of gradient elements: " << totalGrad << "\n" << std::endl;
}

// TODO: Implement gradient check function that compares finite difference gradients with adjoint gradients
int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Configuration parameters

  const int N_mpiRanks = 1;  // Number of MPI ranks for forward/adjoint runs
  const int windowSteps = 4; // Number of time steps in each window
  const windowIdx = 0;       // Index of the current window (0-based)
  const int nProbes = 2;     // Number of probe points for loss computation
  cosnt float eps = 1e-3f;   // Perturbation for finite difference gradient check

  const std::string forwardScript = "run_forward_debug.sh"; // Path to forward script file
  const std::string adjointScript = "run_adjoint_debug.sh"; // Path to adjoint script file
  const std::string forwardParFile = "forward.par";         // Path to forward parameter file
  const std::string lossFile = "loss.txt";                  // Path to loss
  const std::string gradientFile = "gradient.bin";          // Path to gradient file
  const std::string modelFile = "model.pt";                 // Path to save/load model

  // TODO: Load model from file if exists, else initialize new model

  MPI_Finalize();
  return 0;
}