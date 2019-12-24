CXX := g++
CXXFLAGS ?= -std=c++17 -O3 -Wall
SRC = module.cpp
TARGET = _board.cpython-37m-x86_64-linux-gnu.so

# pybind11
INCLUDES += `python -m pybind11 --includes`
CXXFLAGS += -shared -fPIC

.PHONY: all format clean
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $(TARGET) $(LDFLAGS)

format: $(SRC)
	clang-format -i $(SRC)

clean:
	rm -rf $(TARGET)