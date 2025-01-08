# Compiler and flags
CXX := clang++
CXXFLAGS := -std=c++17 -O3 -g -Xpreprocessor -fopenmp
LDFLAGS := -lomp
INCLUDE := -Iinclude -I$(shell brew --prefix libomp)/include
LIBS := -L$(shell brew --prefix libomp)/lib

# Target binary
TARGET := matmul_test_app
SRC := main.cc

# Profile target
profile: $(TARGET)
	samply record --rate 1000000 ./$(TARGET) --profile

# Build target
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LIBS) -o $@ $^ $(LDFLAGS)

neon:
	clang++ -DNEON -DNDEBUG -std=c++17 -O3 -Iinclude -o matmul_test_app main.cc
	./matmul_test_app --profile
	./matmul_test_app --profile
	./matmul_test_app --profile

i8mm:
	clang++ -DNEON_I8MM -march=armv9+i8mm -DNDEBUG -std=c++17 -O3 -Iinclude -o matmul_test_app main.cc
	./matmul_test_app --profile
	./matmul_test_app --profile
	./matmul_test_app --profile


# Clean target
clean:
	rm -f $(TARGET)
