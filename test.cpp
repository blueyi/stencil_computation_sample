#include <iostream>

using namespace std;
int main(void)
{
    int dimX, dimY, dimZ;
    dimX = dimY = dimZ = 10;
    int X, Y, Z;
    cout << "Enter 3 number: ";
    cin >> X >> Y >> Z;
    cout << "Result:" << endl;
    cout << ((X + dimX) % dimX) + ((Y + dimY) % dimY) * dimX + ((Z + dimZ) % dimZ) * dimX * dimY << endl;
    return 0;
}
