## Project-2 : 3D Convolution

### Intro

GPU는 CPU보다 병렬 계산에 특화되어 있기 때문에 Deep learning, graphic에 주로 쓰인다. 본 프로젝트에서는 Deep learning에 사용되는 3D Convolution을 single thread(AVX), multi-thread (with AVX), gpu (CUDA)를 이용해서 구현한다. 각 방식에 따른 성능 차이를 비교해 본다.

&nbsp;

### How to make executable file

```bash
$ ls
Makefile gpu.cu multi_avx.c sample scalar_conv.c single_avx.c

$ make all

```

실행파일 이름

scalar : scalar
single-thread with AVX : conv_single_thread
multi-thread with AVX : conv_multi_thread
gpu : gpu

&nbsp;

### How to run and test

```bash
Scalar and AVX 
[example]
$ ./conv_single_thread <input file> <kernel file> <output file>
$ ./conv_single_thread sample/test1/input.txt sample/test1/kernel.txt sample/test1/output.txt

or if you want to test 5 test cases,

[test scalar]
$ make test_s		

[test single-thread]
$ make test_thd	

[test multi-thread]
$ make test_thds	

this will test scalar, avx with single-thread, avx with multi-thread at once
$ make testall


CUDA gpu
[example]
$ ./gpu <file directory> <tile size>
$ ./gpu sample/test1 3
```

multi-thread with AVX에서 thread의 개수를 변경하고 싶으면 multi_avx.c 파일에서 NUM_THREADS 값을 수정한 후 make

