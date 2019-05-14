#include <string.h>
#include <math.h>
#include<stdlib.h>
#include<stdio.h>
#include"mpi.h"
#include <x86intrin.h>


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

void fillData(int * A, int * B, int * C, int blockSize) {// for matrices initialization
  for (int i = 0; i < blockSize; i++) {
    for (int j = 0; j < blockSize; j++) {
      A[i * blockSize + j] = 1;
      B[i * blockSize + j] = 1;
      C[i * blockSize + j] = -0;
    }
  }
}

void printMatrices(int * A, int * B, int * C, int blockSize) {// for printing matrices A & B & C
  printf("\nProcess 0: MATRIX A\n");
  for (int i = 0; i < blockSize; i++) {
    for (int j = 0; j < blockSize; j++) {
      printf(" %d ", A[i * blockSize + j]);
    }
    printf("\n");
  }
  printf("\nProcess 0: MATRIX B\n");
  for (int i = 0; i < blockSize; i++) {
    for (int j = 0; j < blockSize; j++) {
      printf(" %d ", B[i * blockSize + j]);
    }
    printf("\n");
  }
  printf("\nProcess 0: MATRIX C\n");
  for (int i = 0; i < blockSize; i++) {
    for (int j = 0; j < blockSize; j++) {
      printf(" %d ", C[i * blockSize + j]);
    }
    printf("\n");
  }
}
void fillMatrix_Main(int * A, int * B, int * C, int size) {// initializing main matrices
  int c = 1;
  for (int i = 0; i < size * size; i++, c++) {
    A[i] = c;
    B[i] = 1;
    C[i] = 0;

  }
}
// produce an arrangedBuf that has  elements ordered by process
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
// restores the order of the matrix
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
void printMatrix(int * M, int matrixSize) {
  for (int i = 0; i < matrixSize; i++) {
    for (int j = 0; j < matrixSize; j++) {
      printf(" %d ", M[i * matrixSize + j]);
    }
    printf("\n");
  }
}

void main(int argc, char * argv[]) {
  unsigned long long int start, end, residu;
  unsigned long long int av;
  unsigned int exp;

  int numtasks;
  int ndim = 2;
  int periodic[2] = {
    1,
    1
  };
  int dimensions[2];
  int coord_2d[2] = {
    0,
    0
  };
  int rank_2d;
  int rank;
  int up, down;
  int belongs[2];
  int coords1D, colID, rowID;
  int * A, * B, * C;
  char msg[20];
  int matrixSize, blockSize;
  MPI_Status status;
  MPI_Comm cartcomm, commrow, commcol;
  MPI_Request request, request2;
  MPI_Datatype MatrixBlockType, ScatterType;
  int * sendcounts; // array describing how many elements to send to each process
  int * displs; // array describing the displacements where each segment begins
  double startTime, endTime;

  MPI_Init( & argc, & argv);
  MPI_Comm_size(MPI_COMM_WORLD, & numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, & rank);

  sendcounts = malloc(sizeof(int) * numtasks);
  displs = malloc(sizeof(int) * numtasks);
  //printf("\nnum task is %d", numtasks);

  int dim = sqrt(numtasks);

  dimensions[0] = dimensions[1] = dim; // set dimension of the GRID
  if(rank == 0){
    printf("\nnumber of tasks is : %d , dim %d %d \n", numtasks, dimensions[0], dimensions[1]);
  }
  int remain_dims[2];

  int * Main_A = NULL, * Main_B = NULL, * Main_C = NULL;

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

  if (rank == 0) {
    Main_A = (int * ) malloc(matrixSize * matrixSize * sizeof(int));
    Main_B = (int * ) malloc(matrixSize * matrixSize * sizeof(int));
    Main_C = (int * ) malloc(matrixSize * matrixSize * sizeof(int));
    fillMatrix_Main(Main_A, Main_B, Main_C, matrixSize);
    //printMatrices(Main_A, Main_B, Main_C, matrixSize);
  }

  MPI_Type_contiguous(blockSize * blockSize, MPI_INT, & MatrixBlockType); // create matrix datatype to transfer
  MPI_Type_commit( & MatrixBlockType);
  MPI_Type_vector(blockSize, blockSize, dimensions[0] * blockSize, MPI_INT, & ScatterType); // for the MPI_Scatterv
  MPI_Type_commit( & ScatterType);

  A = (int * ) malloc(blockSize * blockSize * sizeof(int * ));
  B = (int * ) malloc(blockSize * blockSize * sizeof(int * ));
  C = (int * ) malloc(blockSize * blockSize * sizeof(int * ));

  fillData(A, B, C, blockSize);

  MPI_Cart_create(MPI_COMM_WORLD, ndim, dimensions, periodic, 0, & cartcomm);
  MPI_Cart_coords(cartcomm, rank, ndim, coord_2d);
  MPI_Cart_rank(cartcomm, coord_2d, & rank_2d);

  for (int i = 0; i < numtasks; i++) {
    sendcounts[i] = (matrixSize * matrixSize) / numtasks;
  }
  int c = 0;
  for (int i = 0; i < dimensions[0]; i++) {
    for (int j = 0; j < dimensions[1]; j++) {
      sendcounts[c] = (matrixSize * matrixSize) / numtasks;
      displs[c] = ((i * blockSize) * matrixSize) + j * blockSize;
      c++;
    }
  }
  fillData(A , B , C , blockSize);

  int * arrangedBufA = NULL, * arrangedBufB = NULL;

  //printf("Process %d: MATRIX A",rank);
  //printMatrix(A, blockSize);

  belongs[0] = 0;
  belongs[1] = 1;

  MPI_Cart_sub(cartcomm, belongs, & commrow); // create function that returns this commrow
  MPI_Comm_rank(commrow, & rowID);
  MPI_Cart_coords(commrow, rowID, 1, & coords1D);

  belongs[0] = 1;
  belongs[1] = 0;
  MPI_Cart_sub(cartcomm, belongs, & commcol);
  MPI_Comm_rank(commcol, & colID);
  MPI_Cart_coords(commcol, colID, 1, & coords1D);

  int * tempa = (int * ) malloc(blockSize * blockSize * sizeof(int * ));

  int * buff = (int * ) malloc(blockSize * blockSize * sizeof(int));
  int * buff2 = (int * ) malloc(blockSize * blockSize * sizeof(int));
  int source, dest;

  MPI_Cart_shift(commcol, 0, 1, & dest, & source);
  //printf("\ncoords %d %d source dest %d %d rank %d\n", coord_2d[0], coord_2d[1], source, dest, rank_2d);

  int root;
  for (int stage = 0; stage < dim; stage++) {
    root = (coord_2d[0] + stage) % dim;

    if (root == coord_2d[1]) {
      MPI_Bcast(A, 1, MatrixBlockType, root, commrow);

      multiply(A, B, C, blockSize);
    } else {
      MPI_Bcast(tempa, 1, MatrixBlockType, root, commrow);
      multiply(tempa, B, C, blockSize);
    }

    for (int l = 0; l < blockSize; l++) {
      for (int o = 0; o < blockSize; o++) {
        buff[l * blockSize + o] = B[l * blockSize + o];
      }
    }

    MPI_Isend(buff, 1, MatrixBlockType, dest, 1, commcol, & request2);
    MPI_Irecv(buff2, 1, MatrixBlockType, source, 1, commcol, & request);

    MPI_Wait( & request, & status);
    MPI_Wait( & request2, & status);

    for (int l = 0; l < blockSize; l++) {
      for (int o = 0; o < blockSize; o++) {
        B[l * blockSize + o] = buff2[l * blockSize + o];
      }
    }

  }
  end = _rdtsc();

  exp = end - start;

  /*printf("\nProcess %d Matrix C \n", rank);
  for (int i = 0; i < blockSize; i++) {
      for (int j = 0; j < blockSize; j++) {
        printf("%d ", C[i * blockSize + j]);
      }
      printf("\n");
    }*/
if(rank == 0){
    printf("\n Fox algo  \t\t\t %Ld cycles\n\n", exp - residu);
    endTime = MPI_Wtime();
    printf("\nTotal time it took to do the multiplication is :%f \n", endTime - startTime);
}

  MPI_Finalize();
}
