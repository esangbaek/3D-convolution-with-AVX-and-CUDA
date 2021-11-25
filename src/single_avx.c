/***********************************************
 * Intensive programming-2 project-2
 * 3D Convolution
 * Single-thread AVX (Advanced Vector eXtension)
 * Contributor : Lee Sang Baek
 ***********************************************/

/*
2021/11/18	3d convolution한 result와 output 비교 구현(with AVX)
2021/11/19 	시간측정 함수 추가함
2021/11/23	width, height, depth 변수 추가하여 코드 간결화
*/

#include <stdio.h>
#include <immintrin.h>
#include <math.h>
#include <sys/time.h>

/*
   Vector size : AVX 256bit / 32bit float = 8 elements
*/
#define V_SIZE 8
#define USECPSEC 1000000ULL
#define MS 1000ULL

float *kernel;
float *input_mat, *result, *output_mat;
int X_SIZE, Y_SIZE, Z_SIZE;
int width, height, depth;

u_int64_t dtime_usec(u_int64_t start)
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}

/*
한번에 8개 성분을 convolution함
하나의 성분들은 scalar_conv.c 의 sum 함수와 동일하게 연산되어짐
*/
void sum(int z, int y, int x, int padding)
{
	__m256 ker;
	__m256 input;
	__m256 result_sum;

	float *temp_result;

	result_sum = _mm256_set1_ps(0.0);
	int kernel_idx=0;

	for(int i=z-padding/2;i<=z+padding/2;i++)
		for(int j=y-padding/2;j<=y+padding/2;j++)
			for(int k=x-padding/2;k<=x+padding/2;k++)
			{
				//load 8 elements from input matrix
				input = _mm256_loadu_ps(&input_mat[k+j*width+i*width*height]);

				//load kernel's one element
				ker = _mm256_set1_ps(kernel[kernel_idx++]);

				//multiplication & add result to result_sum
				result_sum = _mm256_add_ps(result_sum, _mm256_mul_ps(input, ker));
			}
	
	temp_result = (float*)&result_sum;
	for(int i=0; i<V_SIZE; i++)
	{
		result[x+y*width+z*width*height+i] = temp_result[i];
	}
}

int main(int argc, char *argv[])
{
	if(argc!=4)
	{
		printf("input error\n");
		printf("%s <test/input> <test/kernel> <test/output>\n",argv[0]);
		exit(0);
	}

	FILE *fp;
	int kernel_size, padding, not_same=0, idx;
	u_int64_t difft;

	fp = fopen(argv[2],"r");
	if(fp == NULL)
	{
		perror("error fopen\n");
		return 0;
	}

	//get kernel data
	fscanf(fp, "%d", &kernel_size);
	padding = kernel_size-1;		//input size 와 output size가 같기 때문에 padding 필요
	kernel = aligned_alloc(32, sizeof(float)*kernel_size*kernel_size*kernel_size);
	for(int i=0; i<kernel_size*kernel_size*kernel_size; i++)
		fscanf(fp,"%f",&kernel[i]);

	fclose(fp);

	fp = fopen(argv[1],"r");
	if(fp == NULL)
	{
		perror("error fopen\n");
		return 0;
	}
	fscanf(fp,"%d",&Z_SIZE);
	fscanf(fp,"%d",&Y_SIZE);
	fscanf(fp,"%d",&X_SIZE);
	
	//padding 과 input을 고려한 전체 크기
	width = X_SIZE+padding;
	height = Y_SIZE+padding;
	depth = Z_SIZE+padding;

	//get input data
	input_mat = aligned_alloc(32, sizeof(float)*width*height*depth);
	result = aligned_alloc(32, sizeof(float)*width*height*depth);

	for(int z=0; z<depth; z++)
		for(int y=0; y<height; y++)
			for(int x=0; x<width; x++)
			{
				if(z<padding/2 || y<padding/2 || x<padding/2 || z>Z_SIZE+padding/2-1 || y>Y_SIZE+padding/2-1 || x>X_SIZE+padding/2-1)
					//padding 영역의 값은 0 으로 설정
					input_mat[x + y*width + z*width*height]=0;
				else
				{
					fscanf(fp,"%f",(input_mat + x + y*width + z*width*height));
				}
			}

	fclose(fp);

	fp = fopen(argv[3],"r");
	if(fp == NULL)
	{
		perror("error fopen\n");
		return 0;
	}

	//input과 output크기가 같기 때문에 필요없지만 output.txt 데이터 형식 고려
	fscanf(fp,"%d",&Z_SIZE);
	fscanf(fp,"%d",&Y_SIZE);
	fscanf(fp,"%d",&X_SIZE);

	//get expected output data
	output_mat = aligned_alloc(32, sizeof(float)*width*height*depth);

	for(int z=padding/2; z < Z_SIZE + padding/2; z++)
		for(int y=padding/2; y < Y_SIZE + padding/2; y++)
			for(int x=padding/2; x < X_SIZE + padding/2; x++)
			{
				fscanf(fp,"%f",&output_mat[x + y*width + z*width*height]);
			}

	fclose(fp);

	//한번에 8개의 성분을 계산하기 때문에 V_SIZE 8 씩 증가
	difft = dtime_usec(0);		//start timer
	for(int z=padding/2; z < Z_SIZE + padding/2; z++)
		for(int y=padding/2; y < Y_SIZE + padding/2; y++)
			for(int x=padding/2; x < X_SIZE + padding/2; x+=V_SIZE)
			{
				sum(z,y,x,padding);
			}

	difft = dtime_usec(difft);		//end timer

	for(int z=padding/2; z < Z_SIZE + padding/2; z++)
		for(int y=padding/2; y < Y_SIZE + padding/2; y++)
			for(int x=padding/2; x < X_SIZE + padding/2; x++)
			{
				idx = x+y*width+z*width*height;
				//check result and expected output
				if(fabs(result[idx] - output_mat[idx]) > 0.001f )
					not_same++;
			}

	printf("Data dimension size (x,y,z) : %d x %d x %d\n", X_SIZE, Y_SIZE, Z_SIZE);
	printf("Not same : %d / %d\n",not_same, X_SIZE*Y_SIZE*Z_SIZE);
	printf("Execution time(ms): %f\n\n",difft/(float)MS);

	free(kernel);
	free(input_mat);
	free(output_mat);
	free(result);

	return 0;
}
