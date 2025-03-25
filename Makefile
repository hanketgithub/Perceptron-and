# Makefile at root level

CXX = g++
CXXFLAGS = -std=c++11 -Wall -O2

# folders
SRC_DIR = src
BUILD_DIR = build
BIN = perceptron-and

# source files
SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# default target
all: $(BIN)

# how to build binary
$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# how to compile .cpp into .o
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# clean build
clean:
	rm -rf $(BUILD_DIR) $(BIN)

.PHONY: all clean
