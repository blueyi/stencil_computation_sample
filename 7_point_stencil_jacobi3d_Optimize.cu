#include <iostream>
#include <string>
#include <iomanip>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// #define imin(a, b) (a<b?a:b)

//定义X,Y,Z各维的长度
const int dimX = 10;
const int dimY = 10;
const int dimZ = 10;
const int SIZE = dimX * dimY * dimZ;

//设置每个线程块中线程数量，此处设置三维一样
const int threadPerBlock = 32;

//设置迭代次数
const int times = 10;

//设置stencil边界处邻居的值
__device__ const double BORDER = 0.0;

int count = 0;


//设定线程格中线程块的数量, 避免启动过多线程块
int blockPerGrid(const int dim, const int threadPerBlock)
{
   //由于暂时一个线程只计算一个stencil，所以暂时不能指定线程块的限制
   int temp = dim / threadPerBlock;
   if (dim % threadPerBlock != 0) {
      temp += 1; 
   }
   return temp;
}

//错误处理
#define CHECK_ERROR(error) checkCudaError(error, __FILE__, __LINE__)
#define CHECK_STATE(msg) checkCudaState(msg, __FILE__, __LINE__)

inline void checkCudaError(cudaError_t error, const char *file, const int line)
{
   if (error != cudaSuccess) {
      std::cerr << "CUDA CALL FAILED:" << file << "( " << line << ")- " << cudaGetErrorString(error) << std::endl;
      exit(EXIT_FAILURE);
   }
}
inline void checkCudaState(const char *msg, const char *file, const int line)
{
   cudaError_t error = cudaGetLastError();
   if (error != cudaSuccess) {
      std::cerr << "---" << msg << " Error---" << std::endl;
      std::cerr << file << "( " << line << ")- " << cudaGetErrorString(error) << std::endl;
      exit(EXIT_FAILURE);
   }
}


//计算线程与元素的唯一偏移，以x为行，y为列，z为高
__device__ __host__ int offset(int x, int y, int z) 
{
   return (((x + dimX) % dimX) + ((y + dimY) % dimY) * dimX + ((z + dimZ) % dimZ) * dimX * dimY);
}

__global__ void kernel(double *dev_grid_in, double *dev_grid_out)
{
   //使用shared memory存储每个线程块中的计算
   __shared__ double cache[threadPerBlock * threadPerBlock * 1];
   int cacheIndex = threadIdx.x * threadIdx.y * threadIdx.z;
   cache[cacheIndex] = 0.0;
   __syncthreads();

   //线程索引
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;
   int z = threadIdx.z + blockIdx.z * blockDim.z;

   //设置stencil中各元素值
   double center = dev_grid_in[offset(x, y, z)];
   double up     = (z < (dimZ - 1)) ? dev_grid_in[offset(x, y, z + 1)] : BORDER;
   double down   = (z > 0) ? dev_grid_in[offset(x, y, z - 1)] : BORDER;
   double west   = (x > 0) ? dev_grid_in[offset(x - 1, y, z)] : BORDER;
   double east   = (x < (dimX - 1)) ? dev_grid_in[offset(x + 1, y, z)] : BORDER;
   double south  = (y > 0) ? dev_grid_in[offset(x, y - 1, z)] : BORDER;
   double north  = (y < (dimY - 1)) ? dev_grid_in[offset(x, y + 1, z)] : BORDER;

   //    dev_grid_out[offset(x, y, z)] = 1.0;
//   dev_grid_out[offset(x, y, z)] = (center + up + down + west + east + south + north) * (1.0 / 7.0);
   cache[cacheIndex] = (center + up + down + west + east + south + north) * (1.0 / 7.0);
   __syncthreads();

   //显然此处没有加速效果
   dev_grid_out[offset(x, y, z)] = cache[cacheIndex];
}

//初始化输入，输出
void init(double *grid, int dimX, int dimY, int dimZ)
{
   for (int z=0; z<dimZ; ++z) {
      for (int y=0; y<dimY; ++y) {
         for (int x=0; x<dimX; ++x) {
            if ((x*y*z == 0) || (x == dimX-1) || (y == dimY-1) || (z == dimZ-1)) {
               grid[offset(x, y, z)] = 7;
            }
            else {
               grid[offset(x, y, z)] = 0;
               //      grid[offset(x, y, z)] = count;
            }
            count++;
         }
      }
   }
}

void print(double *grid)
{
   for (int z=0; z<dimZ; ++z) {
      std::cout << z << ":\n\n";
      for (int y=0; y<dimY; ++y) {
         for (int x=0; x<dimX; ++x) {
            std::cout << std::fixed << std::setprecision(3) << grid[offset(x, y, z)] << "\t";
         }
         std::cout << std::endl;
      }
      std::cout << std::endl;
   }
}

void debug(int test, std::string str)
{

   if (test != 0) {
      std::cout << "-----------" << str  << "--------------" << std::endl;
      std::cout << test << std::endl;
   }
   else {
      std::cout << "-----------" << str  << "--------------" << std::endl;
   }
}

int main(void)
{
   CHECK_ERROR(cudaSetDevice(0));
   //由于blocks不能大于1024，所以最后一维设备为1
   dim3 blocks(threadPerBlock, threadPerBlock, 1);
   dim3 grids(blockPerGrid(dimX, blocks.x), blockPerGrid(dimY, blocks.y), blockPerGrid(dimZ, blocks.z));

   double *grid_in, *grid_out;
   grid_in = (double *)malloc(SIZE * sizeof(double));
   grid_out = (double *)malloc(SIZE * sizeof(double));

   double *dev_grid_in, *dev_grid_out;
   CHECK_ERROR(cudaMalloc((void**)&dev_grid_in, SIZE * sizeof(double)));
   CHECK_ERROR(cudaMalloc((void**)&dev_grid_out, SIZE * sizeof(double)));

   init(grid_in, dimX, dimY, dimZ);
   init(grid_out, dimX, dimY, dimZ);

//   debug(0, "input");
//   print(grid_in);

   //统计用于GPU计算的时间
   cudaEvent_t start, stop;
   CHECK_ERROR(cudaEventCreate(&start));
   CHECK_ERROR(cudaEventCreate(&stop));
   CHECK_ERROR(cudaEventRecord(start, 0));
   CHECK_ERROR(cudaEventSynchronize(start));

   CHECK_ERROR(cudaMemcpy(dev_grid_in, grid_in, SIZE * sizeof(double), cudaMemcpyHostToDevice));
   CHECK_ERROR(cudaMemcpy(dev_grid_out, grid_out, SIZE * sizeof(double), cudaMemcpyHostToDevice));

   for (int i=0; i<times; ++i) {
      kernel<<<grids, blocks>>>(dev_grid_in, dev_grid_out);
      std::swap(dev_grid_in, dev_grid_out);
   }
   cudaDeviceSynchronize();

   CHECK_STATE("kernel call");

   CHECK_ERROR(cudaMemcpy(grid_in, dev_grid_in, SIZE * sizeof(double), cudaMemcpyDeviceToHost));

   //计算统计的时间
   CHECK_ERROR(cudaEventRecord(stop, 0));
   CHECK_ERROR(cudaEventSynchronize(stop));
   float elapsedTime;
   CHECK_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

   debug(0, "output");
   print(grid_in);

   std::cout << "Time elapsed: " << std::fixed << std::setprecision(6) << elapsedTime << " ms"  << std::endl;

   CHECK_ERROR(cudaEventDestroy(start));
   CHECK_ERROR(cudaEventDestroy(stop));

   free(grid_in);
   free(grid_out);
   CHECK_ERROR(cudaFree(dev_grid_in));
   CHECK_ERROR(cudaFree(dev_grid_out));
   return 0;
}


