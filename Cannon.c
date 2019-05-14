#include <string.h>
#include <math.h>
#include<stdlib.h>
#include<stdio.h>
#include"mpi.h"
#include <x86intrin.h>

void fillData(int * A, int * B, int * C, int blockSize) {
  int counter = 0;
  for (int i = 0; i < blockSize; i++) {
    for (int j = 0; j < blockSize; j++) {
      counter++;
      A[i * blockSize + j] = counter;
      B[i * blockSize + j] = 1;
      C[i * blockSize + j] = 0;

    }
  }
}
void fillData1(int * A, int * B, int * C, int blockSize) {
  int counter = 0;
  for (int i = 0; i < blockSize; i++) {
    for (int j = 0; j < blockSize; j++) {
      counter++;
      A[i * blockSize + j] = 1;
      B[i * blockSize + j] = 1;
      C[i * blockSize + j] = 0;

    }
  }
}
void printMatrix(int * M, int matrixSize) {
  for (int i = 0; i < matrixSize; i++) {
    for (int j = 0; j < matrixSize; j++) {
      printf(" %d ", M[i * matrixSize + j]);
    }
    printf("\n");
  }
}

void multiply(int * a, int * b, int * c, int n) {
  int i, j, k;
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
      for (k = 0; k < n; k++) {
        c[i * n + j] += a[i * n + k] * b[k * n + j];
      }
    }
  }
}
void PrepareForScater(int * Main_A, int * arrangedBuf, int startFrom, int strid, int blockSize, int matrixSize, int numtasks, int * displs) {

  int counter = 0;
  for (int i = 0; i < numtasks; i++) {
    startFrom = displs[i];
    //printf("start from is %d  " , startFrom);
    for (int j = 0; j < blockSize; j++) {
      for (int k = startFrom; k < startFrom + blockSize; k++) {
        arrangedBuf[counter] = Main_A[k];
        counter++;
      }
      startFrom = startFrom + strid;
    }
  }

}
void RestoreOrder(int * Main_C, int * arrangedBuf, int startFrom, int strid, int blockSize, int matrixSize, int numtasks, int * displs) {

  int counter = 0;
  for (int i = 0; i < numtasks; i++) {
    startFrom = displs[i];
    //printf("start from is %d  " , startFrom);
    for (int j = 0; j < blockSize; j++) {
      for (int k = startFrom; k < startFrom + blockSize; k++) {
        Main_C[k] = arrangedBuf[counter];
        counter++;
      }
      startFrom = startFrom + strid;
    }
  }
}

int main(int argc, char * argv[]) {
  int rank, count = 0, i = 0, j = 0, k = 0;
  char ch;
  int * A, * B, * C, n;

  int numtasks;
  int ndim = 2;
  int periodic[2] = {
    1,
    1
  };
  int dimensions[2];
  int coord_2d[2];
  int rank_2d;
  int belongs[2];
  int coords1D, colID, rowID;
  char msg[20];
  int matrixSize, blockSize;
  MPI_Status status;
  MPI_Comm cartcomm, commrow, commcol;
  MPI_Request requestA0, requestA1, requestB0, requestB1;
  MPI_Datatype MatrixBlockType, ScatterType1, ScatterType2;
  int * sendcounts; // array describing how many elements to send to each process
  int * displs; // array describing the displacements where each segment begins
  double startTime, endTime;
  int * Main_A = NULL, * Main_B = NULL, * Main_C = NULL;
  unsigned long long int start, end, residu;
  unsigned long long int av;
  unsigned int exp;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, & numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, & rank);

  sendcounts = malloc(sizeof(int) * numtasks);
  displs = malloc(sizeof(int) * numtasks);

  int dim = sqrt(numtasks);
  dimensions[0] = dimensions[1] = dim; // set dimension of the GRID
  if(rank == 0){
    printf("\nnumber of tasks is : %d , dim %d %d \n", numtasks, dimensions[0], dimensions[1]);
  }
  int remain_dims[2];

  if (rank == 0) {
    do {
      printf("Matrix is NxN it should be N mod dim = 0   enter N: \n");
      scanf("%d", & matrixSize);
      printf("\nmatrix columns size is : %d , process columns size is %d\n", matrixSize, dim);
    } while (matrixSize % dim != 0);
  }
  if (rank == 0) {
    start = _rdtsc();
    end = _rdtsc();
    residu = end - start;
    startTime = MPI_Wtime();

  }

  start = _rdtsc();

  MPI_Bcast( & matrixSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

  blockSize = matrixSize / dim; // small block dimension


  MPI_Type_contiguous(blockSize * blockSize, MPI_INT, & MatrixBlockType); // create matrix datatype to transfer
  MPI_Type_commit( & MatrixBlockType);

  MPI_Type_vector(2, 2, 4, MPI_INT, & ScatterType1); // for the MPI_Scatterv
  MPI_Type_create_resized(ScatterType1, 0, 2 * sizeof(int), & ScatterType2);
  MPI_Type_commit( & ScatterType2);

  //printf("\nnumber of blocks is %d \t block size is %d \t strid is %d\n" ,blockSize,blockSize, dimensions[0] * blockSize);
//each process has matrix A & B & C
  A = (int * ) malloc(blockSize * blockSize * sizeof(int * ));
  B = (int * ) malloc(blockSize * blockSize * sizeof(int * ));
  C = (int * ) malloc(blockSize * blockSize * sizeof(int * ));

  MPI_Cart_create(MPI_COMM_WORLD, ndim, dimensions, periodic, 0, & cartcomm);
  MPI_Cart_coords(cartcomm, rank, ndim, coord_2d);
  MPI_Cart_rank(cartcomm, coord_2d, & rank_2d);
  //Initialize the matrices
  fillData1(A, B, C, blockSize);

  //printf("\nProcess %d Matrix A\n", rank);
  //printMatrix(A, blockSize);
  //printf("\nProcess %d Matrix B\n", rank);
  //printMatrix(B, blockSize);

  int * buffA0, * buffA1, * buffB0, * buffB1;

  buffA0 = malloc(sizeof(int) * blockSize * blockSize);
  buffA1 = malloc(sizeof(int) * blockSize * blockSize);
  buffB0 = malloc(sizeof(int) * blockSize * blockSize);
  buffB1 = malloc(sizeof(int) * blockSize * blockSize);

  int LS, RS, US, DS;

  MPI_Cart_shift(cartcomm, 1, coord_2d[0], & LS, & RS);// to know the left and right neighbors
  MPI_Cart_shift(cartcomm, 0, coord_2d[1], & US, & DS);//to know the top and bottom neighbors

//  printf("\nleft = %d \t right = %d \t coods (%d , %d)\n", LS, RS, coord_2d[0], coord_2d[1]);

  for (int l = 0; l < blockSize; l++) {
    for (int o = 0; o < blockSize; o++) {
      buffA0[l * blockSize + o] = A[l * blockSize + o];
    }
  }
  for (int l = 0; l < blockSize; l++) {
    for (int o = 0; o < blockSize; o++) {
      buffB0[l * blockSize + o] = B[l * blockSize + o];
    }
  }

  MPI_Isend(buffA0, 1, MatrixBlockType, LS, 1, cartcomm, & requestA0);
  MPI_Irecv(buffA1, 1, MatrixBlockType, RS, 1, cartcomm, & requestA1);

  MPI_Isend(buffB0, 1, MatrixBlockType, US, 1, cartcomm, & requestB0);
  MPI_Irecv(buffB1, 1, MatrixBlockType, DS, 1, cartcomm, & requestB1);

  MPI_Wait( & requestA0, & status);
  MPI_Wait( & requestA1, & status);
  MPI_Wait( & requestB0, & status);
  MPI_Wait( & requestB1, & status);

  for (int l = 0; l < blockSize; l++) {
    for (int o = 0; o < blockSize; o++) {
      A[l * blockSize + o] = buffA1[l * blockSize + o];
    }
  }
  for (int l = 0; l < blockSize; l++) {
    for (int o = 0; o < blockSize; o++) {
      B[l * blockSize + o] = buffB1[l * blockSize + o];
    }
  }

  multiply(A, B, C, blockSize);

  for (i = 1; i < dim; i++) {
    //get neighbors
    MPI_Cart_shift(cartcomm, 1, 1, & LS, & RS);
    MPI_Cart_shift(cartcomm, 0, 1, & US, & DS);

    for (int l = 0; l < blockSize; l++) {
      for (int o = 0; o < blockSize; o++) {
        buffA0[l * blockSize + o] = A[l * blockSize + o];
      }
    }
    for (int l = 0; l < blockSize; l++) {
      for (int o = 0; o < blockSize; o++) {
        buffB0[l * blockSize + o] = B[l * blockSize + o];
      }
    }
    //perform shifting
    MPI_Isend(buffA0, 1, MatrixBlockType, LS, 1, cartcomm, & requestA0);
    MPI_Irecv(buffA1, 1, MatrixBlockType, RS, 1, cartcomm, & requestA1);

    MPI_Isend(buffB0, 1, MatrixBlockType, US, 1, cartcomm, & requestB0);
    MPI_Irecv(buffB1, 1, MatrixBlockType, DS, 1, cartcomm, & requestB1);

    MPI_Wait( & requestA0, & status);
    MPI_Wait( & requestA1, & status);
    MPI_Wait( & requestB0, & status);
    MPI_Wait( & requestB1, & status);

    for (int l = 0; l < blockSize; l++) {
      for (int o = 0; o < blockSize; o++) {
        A[l * blockSize + o] = buffA1[l * blockSize + o];
      }
    }
    for (int l = 0; l < blockSize; l++) {
      for (int o = 0; o < blockSize; o++) {
        B[l * blockSize + o] = buffB1[l * blockSize + o];
      }
    }
    // do submatrix multiplication
    multiply(A, B, C, blockSize);

  }
  end = _rdtsc();

  exp = end - start;

  /*printf("Process=%d MATRIX C \n",rank);
  for (int i = 0; i < blockSize; i++) {
    for (int j = 0; j < blockSize; j++) {
      printf("%d ", C[i * blockSize + j]);
    }
    printf("\n");
  }*/

  if(rank == 0){// to get the cycles and the time
      printf("\n Cannon algo  \t\t\t %Ld cycles\n\n", exp - residu);
      endTime = MPI_Wtime();
      printf("\nTotal time it took to do the multiplication is :%f \n", endTime - startTime);
  }



  MPI_Finalize();
  return 0;
}
