/***********************************************
 * Intensive programming-2 project-2
 * 3D Convolution
 * Single-thread Scalar
 * Contributor : Lee Sang Baek
 ***********************************************/
 
/*
2021/11/17	3d convolution한 result와 output 비교 구현
2021/11/19 	시간측정 함수 추가함
2021/11/23	width, height, depth, padding_width 변수 추가하여 코드 간결화
*/

#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <stdlib.h>

#define USECPSEC 1000000ULL
#define MS 1000ULL

float *kernel;
float *input_mat, *output_mat, *result;
int width, height, depth, padding_width;

u_int64_t dtime_usec(u_int64_t start)
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return ((tv.tv_sec*USECPSEC)+tv.tv_usec)-start;
}
/*
convolution하는 함수
해당 좌표에서 padding_width 만큼 떨어진 좌표들 모두 kernel과 계산하여 더함
*/
float sum(int z, int y, int x)
{
	float res=0.0;
	int kernel_idx=0;
	for(int i=z-padding_width;i<=z+padding_width;i++)
		for(int j=y-padding_width;j<=y+padding_width;j++)
			for(int k=x-padding_width;k<=x+padding_width;k++)
			{
				res += kernel[kernel_idx++]*input_mat[k+width*j+width*height*i];
			}
	return res;
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
	int Z_SIZE,Y_SIZE,X_SIZE;
	u_int64_t difft;
	int kernel_size, not_same=0, idx;

	fp = fopen(argv[2],"r");
	if(fp == NULL)
	{
		perror("error fopen\n");
		return 0;
	}

	fscanf(fp, "%d", &kernel_size);
	padding_width = (kernel_size-1)/2;		//input size = output size
	
	//get kernel data
	kernel = (float*)malloc(sizeof(float)*kernel_size*kernel_size*kernel_size);
	for(int i=0;i<kernel_size*kernel_size*kernel_size;i++)
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
	
	//padding과 input을 고려한 전체 크기
	width = X_SIZE + padding_width * 2;
	height = Y_SIZE + padding_width * 2;
	depth = Z_SIZE + padding_width * 2;

	//get input data
	input_mat = (float*)malloc(sizeof(float)*width*height*depth);
	for(int z=0;z<depth;z++)
		for(int y=0;y<height;y++)
			for(int x=0;x<width;x++)
			{
				if(z<padding_width || y<padding_width || x<padding_width || z>Z_SIZE+padding_width-1 || y>Y_SIZE+padding_width-1 || x>X_SIZE+padding_width-1)
					//padding 영역의 값은 0 으로 설정
					input_mat[x + y*width + z*width*height]=0;
				else
				{
					fscanf(fp,"%f",(input_mat + x + y*width + z*width*height));
				}
			}

	fclose(fp);

	fp = fopen(argv[3],"r");
	//input과 output크기가 같기 때문에 필요없지만 output.txt 데이터 형식 고려
	fscanf(fp,"%d",&Z_SIZE);
	fscanf(fp,"%d",&Y_SIZE);
	fscanf(fp,"%d",&X_SIZE);
	
	
	output_mat = (float*)malloc(sizeof(float)*width*height*depth);
	for(int z=padding_width;z<Z_SIZE+padding_width;z++)
		for(int y=padding_width;y<Y_SIZE+padding_width;y++)
			for(int x=padding_width;x<X_SIZE+padding_width;x++)
			{
				fscanf(fp,"%f",&output_mat[x+y*width+z*width*height]);
			}

	fclose(fp);

	result = (float*)malloc(sizeof(float)*width*height*depth);
	difft = dtime_usec(0);	//start timer	
	for(int z=padding_width;z<Z_SIZE+padding_width;z++)
		for(int y=padding_width;y<Y_SIZE+padding_width;y++)
			for(int x=padding_width;x<X_SIZE+padding_width;x++)
			{
				result[x+y*width+z*width*height] = sum(z,y,x);
			}
	difft = dtime_usec(difft);	//end timer

	for(int z=padding_width;z<Z_SIZE+padding_width;z++)
		for(int y=padding_width;y<Y_SIZE+padding_width;y++)
			for(int x=padding_width;x<X_SIZE+padding_width;x++)
			{
				idx = x+y*width+z*width*height;
				//check result and expected output
				if(fabs(result[idx] - output_mat[idx])>0.001f)
					not_same++;
			}

	printf("Data dimension size (x,y,z) : %d x %d x %d\n", X_SIZE, Y_SIZE, Z_SIZE);
	printf("Not same : %d / %d\n", not_same, X_SIZE*Y_SIZE*Z_SIZE);
	printf("Execution time(seconds): %f\n\n", difft/(float)MS);

	free(kernel);
	free(input_mat);
	free(output_mat);
	free(result);

	return 0;
}
