build:
	g++ -std=c++17 -O3 -I include -o matmul_test_app main.cc

profile:
	g++ -std=c++17 -O3 -g -I include -o matmul_test_app main.cc
	samply record --rate 100000 ./matmul_test_app --profile
