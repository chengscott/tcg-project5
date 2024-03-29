CXX := g++
CXXFLAGS := -std=c++17
CXXFLAGS += -O3 -march=native #-flto
#CXXFLAGS += -O0 -g #-pg

LIBTORCH := /home/cgilab/GO/adam/Downloads/libtorch
#LIBTORCH := /opt/conda/lib/python3.6/site-packages/torch
MCTS_TARGET := nogo
EVALUATOR_TARGET := evaluator
BOARD_TARGET := _board.cpython-37m-x86_64-linux-gnu.so

.PHONY: all mcts-sp mcts board format check clean
all: mcts

mcts-sp: CPPFLAGS += -DSELF_PLAY
mcts-sp: mcts

mcts: SRC = nogo.cpp
mcts: CPPFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1
mcts: CPPFLAGS += -DSHOW_INFO
mcts: CPPFLAGS += -isystem $(LIBTORCH)/include
mcts: CPPFLAGS += -isystem $(LIBTORCH)/include/csrc/api/include
mcts: CXXFLAGS += -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wpedantic
mcts: CXXFLAGS += -Wunused -Wsign-conversion -Wdouble-promotion
mcts: LDFLAGS += -L$(LIBTORCH)/lib
mcts: LDFLAGS += -ltorch -lc10 -fopenmp -Wl,-rpath=$(LIBTORCH)/lib
mcts: LDFLAGS += -lcuda
mcts:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SRC) -o $(MCTS_TARGET) $(LDFLAGS)

evaluator: SRC = evaluator.cpp
evaluator: CPPFLAGS += -DSELF_PLAY
evaluator: CPPFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1
evaluator: CPPFLAGS += -isystem $(LIBTORCH)/include
evaluator: CPPFLAGS += -isystem $(LIBTORCH)/include/csrc/api/include
evaluator: CXXFLAGS += -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wpedantic
evaluator: CXXFLAGS += -Wunused -Wsign-conversion -Wdouble-promotion
evaluator: LDFLAGS += -L$(LIBTORCH)/lib
evaluator: LDFLAGS += -ltorch -lc10 -fopenmp -Wl,-rpath=$(LIBTORCH)/lib
evaluator: LDFLAGS += -lcuda
evaluator:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SRC) -o $(EVALUATOR_TARGET) $(LDFLAGS)

board: SRC = module.cpp
# pybind11
board: INCLUDES += `python -m pybind11 --includes`
board: CXXFLAGS += -shared -fPIC
board:
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(BOARD_TARGET)
	cp $(BOARD_TARGET) ../supervised

format:
	clang-format -i *.cpp *.hpp

check: SRC = nogo.cpp
check: CPPFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1
check: CPPFLAGS += -isystem $(LIBTORCH)/include
check: CPPFLAGS += -isystem $(LIBTORCH)/include/csrc/api/include
check: CHECKS = -checks=bugprone-*,clang-analyzer-*,modernize-*,performance-*,readability-*
check:
	clang-tidy $(SRC) *.hpp $(CHECKS) -- $(CPPFLAGS) $(CXXFLAGS)

run: $(MCTS_TARGET)
	gogui-server -port 5566 "./$(MCTS_TARGET) ../supervised/model/model.pt"

clean:
	rm -rf $(MCTS_TARGET) $(EVALUATOR_TARGET) $(BOARD_TARGET)