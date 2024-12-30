build:
	g++ -std=c++17 -O3 -I include -o matmul_test_app main.cc

profile:
	samply record --rate 100000 ./build/debug/matmul_test_app
