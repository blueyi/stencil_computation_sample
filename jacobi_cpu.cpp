#include <iostream>
#include <cmath>
#include <iomanip>

const float ERROR = 0.000001;
const int SIZE = 3;
//线性方程组为ax=b, 迭代结果数组为y, 初始迭代数组为x, 数组大小为SIZE，返回总次数
int jacobi_kernel(float (*a)[SIZE], float *b, float *x, float *y, int SIZE, float ERROR);

int main(void)
{
    float a[3][3] = {5,2,1,-1,4,2,2,-3,10};
    float b[3]={-12,20,3};
    float x[3]={0,0,0};
    float y[3]={0,0,0};
    //int size = sizeof(b) / sizeof(float);
    int times = 0;  //记录总次数

    times = jacobi_kernel(a, b, x, y, SIZE, ERROR);

    std::cout << "-----------result------------------" << std::endl;
    std::cout << "Total time: " << times << std::endl;
    for (int i=0; i<3; i++)
       std::cout << std::fixed << std::setprecision(6) << y[i] << std::endl;
    return 0;
}

int jacobi_kernel(float (*a)[SIZE], float *b, float *x, float *y, int SIZE, float ERROR)
{
    float sum = 0.0;  //临时求和
    float abs = 0.0;  
    int times = 0;  //记录总次数
    while (true)
    {
        for (int i=0; i<SIZE; ++i)
        {
            for (int j=0; j<SIZE; ++j)
            {
                sum += a[i][j] * x[j];
            }
            sum -= a[i][i] * x[i];
            y[i] = (b[i] - sum) / a[i][i];
            sum = 0.0;
        }
        int i = 0;
        while (i < SIZE)
        {
            abs = fabs(y[i] - x[i]);
            if (abs > ERROR)
                break;
            else
                i++;
        }
        if (i != SIZE)
        {
            times++;
            std::cout << times << " times, currently error: " << abs << std::endl;
            for (int i=0; i<SIZE; i++)
                x[i] = y[i];
        }
        else
            break;
    }
    return times;
}
