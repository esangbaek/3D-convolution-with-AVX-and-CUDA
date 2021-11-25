/***********************************************
 * Intensive programming-2 project-2
 * 3D Convolution
 * GPU (CUDA)
 * Contributor : Yongha Kwon
 ***********************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define MAX_KERNEL_SIZE 7

 

__constant__ float Mc[MAX_KERNEL_SIZE * MAX_KERNEL_SIZE * MAX_KERNEL_SIZE];

// 2021 11 17 add gpu convolution
__global__ void Conv3D(float *input, float *output, int width, int height, int channel, int kernel_size, int block_size, int tile_size){


    //Ns[block_size][block_size][block_size]
    extern __shared__ float Ns[];
    //printf("shared\n");

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int row_o = blockIdx.y * tile_size + ty;
    int col_o = blockIdx.x * tile_size + tx;
    int ch_o = blockIdx.z * tile_size + tz;

    int row_i = row_o - kernel_size/2;
    int col_i = col_o - kernel_size/2;
    int ch_i = ch_o - kernel_size/2;

    float out = 0.0f;
    if((ch_i < 0) || (ch_i >= channel) || (row_i < 0) || (row_i >= height) || (col_i < 0) || (col_i >= width)){
        //printf("%d\n",tz*block_size*block_size + ty*block_size + tx);
        Ns[tz*block_size*block_size + ty*block_size + tx] = 0.0f;
    }
    else{
        //printf("%d\n",tz*block_size*block_size + ty*block_size + tx);
        Ns[tz*block_size*block_size + ty*block_size + tx] = input[ch_i * height * width + row_i * width + col_i];
        
    }
    __syncthreads();
    //printf("sync end\n");
    
    if(ty < tile_size && tx < tile_size && tz < tile_size){
        for(int i = 0; i < kernel_size; i++){
            for(int j = 0; j < kernel_size; j++){
                for(int k = 0; k < kernel_size; k++){
                    
                    out += Mc[i*kernel_size*kernel_size+j*kernel_size+k] * Ns[(i+tz)*block_size*block_size + (j+ty) * block_size + (k+tx)];
                    //printf("%d %d %d %d %d %d\n", i, j, k, tz, ty, tx);
                }
            }
        }
        // printf("%d %f\n",row_o * width + col_o ,output);
        if(row_o < height && col_o < width && ch_o < channel)
            output[ch_o * height*width + row_o * width + col_o] = out;
    }
}

//2021 11 17 add verification function
void verification(const float *ans, const float *ret, int channel, int height, int width){
    for(int i = 0; i < channel * height * width; i++){
        if(abs(ret[i] - ans[i]) >= 0.001f){
            printf("Not Equal\n");
            return;
        }
    }

    printf("equal\n");
    return;

}


//2021 11 18 add cuda run function
void run_cuda(const float *input_tmp, const float *kernel_tmp, float *ret, const int channel, const int height, const int width, const int kernel_size, const int tile_size){
    float *input;
    float *output;
    int block_size;
    cudaEvent_t start, end;
    float time_ms;

    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    cudaMalloc((void**)&input, sizeof(float) * channel * height * width);
    cudaMalloc((void**)&output, sizeof(float) * channel * height * width);
    //ret =(float*)malloc(sizeof(float)*channel*height*width);

    cudaMemcpyToSymbol(Mc, kernel_tmp, sizeof(float) * kernel_size * kernel_size * kernel_size);
    cudaMemcpy(input, input_tmp, sizeof(float) * channel * height * width, cudaMemcpyHostToDevice);

    block_size = tile_size + (kernel_size - 1);
    //printf("%d\n", block_size);
    printf("\n\ngpu block size: %d, total_size: %d\n",block_size, block_size * block_size*block_size);
    dim3 dimBlock(block_size, block_size, block_size);
    dim3 dimGrid(ceil(width / (tile_size * 1.0)), ceil(height / (tile_size * 1.0)), ceil(channel / (tile_size * 1.0)));

    cudaEventRecord(start, 0);
    Conv3D <<< dimGrid, dimBlock, sizeof(float)*block_size*block_size*block_size >>> (input, output, width, height, channel, kernel_size, block_size, tile_size);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&time_ms, start, end);

    cudaDeviceSynchronize();
    
    cudaMemcpy(ret, output, sizeof(float) * channel * height * width, cudaMemcpyDeviceToHost);

    cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		// somthing's gone wrong
		// print out the CUDA error as a string
		fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(error));

		// we can't recover from the error -- exit the program
		return;
	}
    printf("time : %f ms\n", time_ms);
}


//2021 11 17 setting and run conv3d
//2021 11 18 modify to not call direct conv3d. instead call run_cuda
int main(int argc, char *argv[]){
    //blcok size 가 11이 되면 되지 않음.

    int channel, width, height, kernel_size, tile_size;
    int i = 0;
    float tmp_data;

    char input_path[30];
    char output_path[30];
    char kernel_path[30];

    strcpy(input_path, argv[1]);
    strcpy(kernel_path, argv[1]);
    strcpy(output_path, argv[1]);

    
    float *input_tmp, *kernel_tmp, *ret;
    float *ans;
    
    tile_size = atoi(argv[2]);
    
    strcat(input_path, "input.txt");
    strcat(output_path, "output.txt");
    strcat(kernel_path, "kernel.txt");

    FILE *fp1 = fopen(input_path,"r");
    if(fp1){
        printf("open file: ");
    }
    else{
        printf("can't open file\n");
        return -1;
    }
    printf("%s\n",input_path);
    fscanf(fp1, "%d %d %d ", &channel, &height, &width);
    input_tmp = (float*)malloc(sizeof(float)*width*height*channel);
    
    ans = (float*)malloc(sizeof(float)*width*height*channel);

    i = 0;
    
    while(fscanf(fp1, "%f ", &tmp_data) > 0){
        input_tmp[i++] = tmp_data;
        //printf("%f\n",tmp_data);
    }
    
    fclose(fp1);

    FILE *fp2 = fopen(output_path, "r");

    fscanf(fp2, "%d %d %d ", &channel, &height, &width);

    i = 0;
    while(fscanf(fp2, "%f ", &tmp_data) > 0){
        ans[i++] = tmp_data;
    }
    fclose(fp2);

    FILE *fp3 = fopen(kernel_path, "r");
    fscanf(fp3, "%d ", &kernel_size);
    kernel_tmp = (float*)malloc(sizeof(float) * kernel_size * kernel_size * kernel_size);
    i = 0;
    while(fscanf(fp3, "%f ", &tmp_data) > 0){
        kernel_tmp[i++] = tmp_data;
    }
    fclose(fp3);

    ret =(float*)malloc(sizeof(float)*channel*height*width);

    if((tile_size + kernel_size - 1) > 10){
        printf("block size is %d((%d)tile_size + ((%d)kernel_size - 1)\n",(tile_size + kernel_size - 1), tile_size, kernel_size);
        printf("block size must less than 11\n");
        return -1;
    }

    run_cuda(input_tmp, kernel_tmp, ret, channel, height, width, kernel_size, tile_size);
    verification(ans, ret, channel, height, width);

    return 0;
}