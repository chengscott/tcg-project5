CXX := g++
CXXFLAGS := -std=c++17
CXXFLAGS += -O3 -march=native #-flto
#CXXFLAGS += -O0 -g #-pg
CXXFLAGS += -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wpedantic
CXXFLAGS += -Wunused -Wsign-conversion -Wdouble-promotion

LIBTORCH := /home/cgilab/GO/adam/Downloads/libtorch
MCTS_TARGET := nogo
BOARD_TARGET := _board.cpython-37m-x86_64-linux-gnu.so

.PHONY: all mcts board format check clean
all: mcts

mcts: SRC = nogo.cpp
mcts: CPPFLAGS += -D_GLIBCXX_USE_CXX11_ABI=1
mcts: CPPFLAGS += -isystem $(LIBTORCH)/include
mcts: CPPFLAGS += -isystem $(LIBTORCH)/include/csrc/api/include
mcts: LDFLAGS += -L$(LIBTORCH)/lib
mcts: LDFLAGS += -ltorch -lc10 -fopenmp -Wl,-rpath=$(LIBTORCH)/lib
mcts: LDFLAGS += -lcuda
mcts:
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(SRC) -o $(MCTS_TARGET) $(LDFLAGS)

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
	rm -rf $(MCTS_TARGET) $(BOARD_TARGET)