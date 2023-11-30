// write a sample mpi code 
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size, i;
    int *sendbuf, *recvbuf;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get size

    sendbuf = (int *)malloc(size * sizeof(int));
    recvbuf = (int *)malloc(size * sizeof(int));

    for (i = 0; i < size; i++)
        sendbuf[i] = rank * size + i;

    MPI_Alltoall(sendbuf, 1, MPI_INT, recvbuf, 1, MPI_INT, MPI_COMM_WORLD);

    printf("rank = %d, recvbuf = ", rank);
    for (i = 0; i < size; i++)
        printf("%d ", recvbuf[i]);
    printf("\n");

    MPI_Finalize();
    return 0;
}