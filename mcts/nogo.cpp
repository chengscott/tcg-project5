#include "agent.hpp"
#include "gtp.hpp"

int main(int argc, char **argv) {
  std::string model_file = "model.pt";
  if (argc > 1) {
    model_file = argv[1];
  }

  auto &gtp = GTPHelper::getInstance();
  gtp.registerAgent(model_file);
  while (gtp.execute()) {
    ;
  }
}