for (int n = 0; n < num_nodes; n++)
{ 
  B[n] = All_points[n-1] + All_points[n+1] + All_points[n-nx] + All_points[n+nx] + All_points[n-nxny] + All_points[n+nxny];
  All_points[n] = new value;
}


//Why not only one dimension in you cuda? 
__global__ void FTCS3D( float *u,int HEIGHT, int WIDTH, int DEPTH)
{   
  int x = threadIdx.x+blockDim.x*blockIdx.x;
  int totid = HEIGHT * WIDTH * DEPTH;

  if (x < totid)
  {
    if (x==1 )
    u[x]=5.0;
  }

}


//Listing 1.1. An example of a stencil motif showing a Jacobi solver that uses a 7-point
//stencil to approximate the solution to the Poisson equation in 3 dimensions.
// Unew, U, and rhs are N x N x N g r i d s
for t = 1 to num steps { 
  for k = 1 to N - 2
  for j = 1 to N - 2
  for i = 1 to N - 2 // l e a d i n g dimension
  Unew[k, j, i] = a * (U[k-1, j, i]+U[k+1, j, i]+U[k ,j-1, i] + U[k , j+1, i]+U[k, j, i-1]+U[k, j, i + 1])-b*rhs[k, j, i] ;
  swap (U,Unew)
}

// Example: Heat Equation
for (t=0; t<timesteps; t++)  {  // time step loop
  for (k=1; k<nz-1; k++) {
    for (j=1; j<ny-1; j++) {
      for (i=1; i<nx-1; i++) {
        // 3-d 7-point stencil
        B[i][j][k] = A[i][j][k+1] + A[i][j][k-1] +
          A[i][j+1][k] + A[i][j-1][k] + A[i+1][j][k] + 
          A[i-1][j][k] – 6.0 * A[i][j][k] / (fac*fac);
      }
    }
  }
  temp_ptr = A;
  A = B;
  B = temp_ptr;
}  


//Heat Equation, Add Tiling
for (t=0; t<timesteps; t++)  {  // time step loop
  for (jj = 1; jj < ny-1; jj+=TJ) {
    for (ii = 1; ii < nx - 1; ii+=TI) {
      for (k=1; k<nz-1; k++) {
        for (j = jj; j < MIN(jj+TJ,ny - 1); j++) {
          for (i = ii; i < MIN(ii+TI,nx - 1); i++) {
            // 3-d 7-point stencil
            B[i][j][k] = A[i][j][k+1] + A[i][j][k-1] +
              A[i][j+1][k] + A[i][j-1][k] + A[i+1][j][k] + 
              A[i-1][j][k] – 6.0 * A[i][j][k] / (fac*fac);
          }
        }
      }
      temp_ptr = A;
      A = B;
      B = temp_ptr;
    }  
  }
}

// Heat Equation, Time Skewing
for (kk=1; kk < nz-1; kk+=tz) {
  for (jj = 1; jj < ny-1; jj+=ty) {
    for (ii = 1; ii < nx - 1; ii+=tx) {
      for (t=0; t<timesteps; t++)  {  // time step loop
        … calculate bounds from t and slope …
          for (k=blockMin_z; k < blockMax_z; k++) {
            for (j=blockMin_y; j < blockMax_y; j++) {
              for (i=blockMin_x; i < blockMax_x; i++) {
                // 3-d 7-point stencil
                B[i][j][k] = A[i][j][k+1] + A[i][j][k-1] +
                  A[i][j+1][k] + A[i][j-1][k] + A[i+1][j][k] + 
                  A[i-1][j][k] – 6.0 * A[i][j][k] / (fac*fac);
              }
            }
          }
        temp_ptr = A;
        A = B;
        B = temp_ptr;
      }  
    }
  }
}
