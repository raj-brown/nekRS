#include "net.hpp"
#include <iostream>


int main(int argc, char **argv)
{
  const std::string cmd = argc > 1 ? argv[1] : "";
  auto m = std::make_shared<Net>();
  if (cmd=="init"){
    torch::manual_seed(1);
    auto m = std::make_shared<Net>();
    m->to(kModelDtype);
    int64_t n = 0;
    for (auto &p:m->parameters()) n +=p.numel();
    torch::save(m, "model.pt");
    std::cout << "wrote model.pt (" << n << " parameters)\n";
    return 0;
  }

  std::cerr << "usgae: dles_driver init\n";
  return 1;

}
