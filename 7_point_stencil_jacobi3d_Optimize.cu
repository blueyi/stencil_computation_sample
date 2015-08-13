#include <iostream>
#include <string>
//格式化输入输出
#include <iomanip>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//通过文件进行数据的输入输出
#include <fstream>
// 使用字符串流读取文件中的数据
#include <sstream>

/*
   对于本机GTX850M来说，当Threads Per Block = 1024，registers per thread = 32, shared memory per
   blocks(bytes)=1024时可以获得最佳性能。
 */

// #define imin(a, b) (a<b?a:b)

//定义X,Y,Z各维的长度
const int dimX = 10;
const int dimY = 10;
const int dimZ = 10;
const int SIZE = dimX * dimY * dimZ;

//设置每个线程块中线程数量，此处设置三维一样
const int threadPerBlock = 10;

//设置迭代次数
const int times = 90;

//设置stencil边界处邻居的值
__device__ const double BORDER = 0.0;

//生成递增的输入数据
int count = 0;

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

//输出当前使用的设备属性
void print_device(const int id)
{
   cudaDeviceProp props;
   cudaGetDeviceProperties(&props, id);
   std::cout << "---Property of currently device used---" << std::endl;
   std::cout  << "Device " << id <<  ": " << props.name << std::endl;
   std::cout  << "CUDA Capability: " << props.major << "." << props.minor << std::endl;
}

//选择multiprocessor最多的CUDA设备
void setCudaDevice(int id)
{
   int numDevices = 0;
   cudaGetDeviceCount(&numDevices);
   if (numDevices > 1) {
      cudaDeviceProp props;
      cudaGetDeviceProperties(&props, id);
      int maxMultiProcessors = props.multiProcessorCount;
      for (int device=1; device<numDevices; ++device) {
         cudaGetDeviceProperties(&props, device);
         if (maxMultiProcessors < props.multiProcessorCount) {
            maxMultiProcessors = props.multiProcessorCount;
            id = device;
         }
      }
   }
   CHECK_ERROR(cudaSetDevice(id));
   print_device(id);
}

//计算元素的唯一线性偏移，以x为行，y为列，z为高
__device__ __host__ int offset(int x, int y, int z) 
{
   return (((x + dimX) % dimX) + ((y + dimY) % dimY) * dimX + ((z + dimZ) % dimZ) * dimX * dimY);
}

__global__ void kernel(double *dev_grid_in, double *dev_grid_out)
{
   //线程索引
   int x = threadIdx.x + blockIdx.x * blockDim.x;
   int y = threadIdx.y + blockIdx.y * blockDim.y;
   int z = threadIdx.z + blockIdx.z * blockDim.z;

   //使用shared memory存储输入
   __shared__ double cache_in[threadPerBlock * threadPerBlock * threadPerBlock];
   //使用同样的方法将线程块中的线程索引线性化
   int cacheIndex = offset(threadIdx.x, threadIdx.y, threadIdx.z);
   cache_in[cacheIndex] = 0.0;
   __syncthreads();
   cache_in[cacheIndex] = dev_grid_in[offset(x, y, z)];
   __syncthreads();

   //可以考虑通过存储位置来减少register使用
   //   int center = offset(x, y, z);
   //   int up     = offset(x, y, z + 1);
   //   int down   = offset(x, y, z - 1);
   //   int west   = offset(x - 1, y, z);
   //   int east   = offset(x + 1, y, z);
   //   int south  = offset(x, y - 1, z);
   //   int north  = offset(x, y + 1, z);

   //设置stencil中各元素值
   double center = cache_in[offset(threadIdx.x, threadIdx.y, threadIdx.z)];
   double up     = (threadIdx.z < (dimZ - 1)) ? cache_in[offset(threadIdx.x, threadIdx.y, threadIdx.z + 1)] : BORDER;
   double down   = (threadIdx.z > 0) ? cache_in[offset(threadIdx.x, threadIdx.y, threadIdx.z - 1)] : BORDER;
   double west   = (threadIdx.x > 0) ? cache_in[offset(threadIdx.x - 1, threadIdx.y, threadIdx.z)] : BORDER;
   double east   = (threadIdx.x < (dimX - 1)) ? cache_in[offset(threadIdx.x + 1, threadIdx.y, threadIdx.z)] : BORDER;
   double south  = (threadIdx.y > 0) ? cache_in[offset(threadIdx.x, threadIdx.y - 1, threadIdx.z)] : BORDER;
   double north  = (threadIdx.y < (dimY - 1)) ? cache_in[offset(threadIdx.x, threadIdx.y + 1, threadIdx.z)] : BORDER;

   dev_grid_out[offset(x, y, z)] = (center + up + down + west + east + south + north) * (1.0 / 7.0);
}

//初始化输入，输出
void init(double *grid, int dimX, int dimY, int dimZ)
{
   for (int z=0; z<dimZ; ++z) {
      for (int y=0; y<dimY; ++y) {
         for (int x=0; x<dimX; ++x) {
            if ((x*y*z == 0) || (x == dimX-1) || (y == dimY-1) || (z == dimZ-1)) {
               grid[offset(x, y, z)] = 7.0;
            }
            else {
               grid[offset(x, y, z)] = 0.0;
               //      grid[offset(x, y, z)] = count;
            }
            count++;
         }
      }
   }
}

//显示结果
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

//将数据写入到文件
void outFile(double *grid)
{
   unsigned int data_count = 0;
   std::string file_name = "out.txt";
   std::fstream ofile(file_name.c_str(), std::fstream::out);
   if (!ofile){
      std::cout << "Error to open output file: "  << file_name << std::endl;
   }
   else {
      for (int z=0; z<dimZ; ++z) {
         ofile << z << ":\n";
         for (int y=0; y<dimY; ++y) {
            for (int x=0; x<dimX; ++x) {
               ofile << std::fixed << std::setprecision(3) << grid[offset(x, y, z)] << "\t";
               data_count++;
            }
            ofile << std::endl;
         }
         ofile << std::endl;
      }
   }
   ofile.close();
   if (data_count == SIZE) {
      std::cout << "Output " << data_count << " Data Successfully" << std::endl;
   }
   else {
      std::cout << "Error, Output " << data_count << " Data" << std::endl;
   }
}

//从文件中读取数据
double * inFile(double *grid)
{
   unsigned int data_count = 0;
   std::string file_name = "in.txt";
   std::fstream ifile(file_name.c_str(), std::fstream::in);
   if (!ifile){
      std::cout << "Error to open output file: "  << file_name << std::endl;
   }
   else {
      std::string str;
      std::istringstream stream;
      for (int z=0; z<dimZ; ++z) {
         for (int y=0; y<dimY; ++y) {
            getline(ifile, str);
            //跳过空行
            while (str.empty()) {
               getline(ifile, str);
            }
            stream.str(str);
            for (int x=0; x<dimX && !stream.eof(); ++x) {
               stream >> grid[offset(x, y, z)];
               data_count++;
            }
            stream.clear();
            stream.sync();
         }
      }
   }
   ifile.close();
   if (data_count == SIZE) {
      std::cout << "Input " << data_count << " Data Successfully" << std::endl;
   }
   else {
      std::cout << "Error, Input " << data_count << " Data" << std::endl;
   }
   return grid;
}

int main(void)
{
   setCudaDevice(0);
   //此处有1000个数据，所以一共启动1000个线程，一个线程块，使一个线程对应一个数据
   dim3 blocks(threadPerBlock, threadPerBlock, threadPerBlock);
   //此处该方法会浪费大量线程，因为各维的线程块并不相等，导致有些线程块中会有很多空闲线程
   dim3 grids(blockPerGrid(dimX, blocks.x), blockPerGrid(dimY, blocks.y), blockPerGrid(dimZ, blocks.z));

   double *grid_in, *grid_out;
   grid_in = (double *)malloc(SIZE * sizeof(double));
   grid_out = (double *)malloc(SIZE * sizeof(double));

   double *dev_grid_in, *dev_grid_out;
   CHECK_ERROR(cudaMalloc((void**)&dev_grid_in, SIZE * sizeof(double)));
   CHECK_ERROR(cudaMalloc((void**)&dev_grid_out, SIZE * sizeof(double)));

   //初始化输入输出
   //   init(grid_in, dimX, dimY, dimZ);
   //   init(grid_out, dimX, dimY, dimZ);
   inFile(grid_in);

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

   //   debug(0, "output");
   //   print(grid_in);
   outFile(grid_in);

   std::cout << "Time elapsed: " << std::fixed << std::setprecision(6) << elapsedTime << " ms"  << std::endl;

   CHECK_ERROR(cudaEventDestroy(start));
   CHECK_ERROR(cudaEventDestroy(stop));

   free(grid_in);
   free(grid_out);
   CHECK_ERROR(cudaFree(dev_grid_in));
   CHECK_ERROR(cudaFree(dev_grid_out));
   return 0;
}


