all : gpu scalar conv_single_thread conv_multi_thread

gpu : gpu.cu
	nvcc -o $@ $^

scalar : scalar_conv.c
	gcc -o $@ $^

conv_single_thread : single_avx.c
	gcc -o $@ $^ -mavx2

conv_multi_thread : multi_avx.c
	gcc -o $@ $^ -mavx2 -pthread

testall : test_s test_thd test_thds

test_s : test_1_1 test_2_1 test_3_1 test_4_1 test_5_1

test_thd : test_1_2 test_2_2 test_3_2 test_4_2 test_5_2

test_thds : test_1_3 test_2_3 test_3_3 test_4_3 test_5_3

test_1_1 : scalar
	@echo "SISD testfile 1"; \
	./scalar sample/test1/input.txt sample/test1/kernel.txt sample/test1/output.txt

test_2_1 :
	@echo "SISD testfile 2"; \
	./scalar sample/test2/input.txt sample/test2/kernel.txt sample/test2/output.txt

test_3_1 :
	@echo "SISD testfile 3"; \
	./scalar sample/test3/input.txt sample/test3/kernel.txt sample/test3/output.txt

test_4_1 :
	@echo "SISD testfile 4"; \
	./scalar sample/test4/input.txt sample/test4/kernel.txt sample/test4/output.txt

test_5_1 :
	@echo "SISD testfile 5"; \
	./scalar sample/test5/input.txt sample/test5/kernel.txt sample/test5/output.txt


test_1_2 : conv_single_thread
	@echo "#################################"; \
	echo "SIMD Single thread testfile 1"; \
	./conv_single_thread sample/test1/input.txt sample/test1/kernel.txt sample/test1/output.txt

test_2_2 :
	@echo "SIMD Single thread testfile 2"; \
	./conv_single_thread sample/test2/input.txt sample/test2/kernel.txt sample/test2/output.txt

test_3_2 :
	@echo "SIMD Single thread testfile 3"; \
	./conv_single_thread sample/test3/input.txt sample/test3/kernel.txt sample/test3/output.txt

test_4_2 :
	@echo "SIMD Single thread testfile 4"; \
	./conv_single_thread sample/test4/input.txt sample/test4/kernel.txt sample/test4/output.txt

test_5_2 :
	@echo "SIMD Single thread testfile 5"; \
	./conv_single_thread sample/test5/input.txt sample/test5/kernel.txt sample/test5/output.txt

test_1_3 : conv_multi_thread
	@echo "#################################"; \
	echo "SIMD Multi thread testfile 1"; \
	./conv_multi_thread sample/test1/input.txt sample/test1/kernel.txt sample/test1/output.txt

test_2_3 :
	@echo "SIMD Multi thread testfile 2"; \
	./conv_multi_thread sample/test2/input.txt sample/test2/kernel.txt sample/test2/output.txt

test_3_3 :
	@echo "SIMD Multi thread testfile 3"; \
	./conv_multi_thread sample/test3/input.txt sample/test3/kernel.txt sample/test3/output.txt

test_4_3 :
	@echo "SIMD Multi thread testfile 4"; \
	./conv_multi_thread sample/test4/input.txt sample/test4/kernel.txt sample/test4/output.txt

test_5_3 :
	@echo "SIMD Multi thread testfile 5"; \
	./conv_multi_thread sample/test5/input.txt sample/test5/kernel.txt sample/test5/output.txt

clean:
	rm  scalar
	rm  conv_single_thread
	rm  conv_multi_thread
	rm gpu
