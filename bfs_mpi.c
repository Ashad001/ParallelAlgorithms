#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MAX_VERTICES 500
#define ROOT_PROCESS 0

void readGraphFromCSV(const char *filename, int graph[MAX_VERTICES][MAX_VERTICES], int vertices)
{
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < vertices; ++i)
    {
        for (int j = 0; j < vertices; ++j)
        {
            if (fscanf(file, "%d,", &graph[i][j]) != 1)
            {
                perror("Error reading from file");
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(file);
}

void parallelBFS(int rank, int size, int graph[MAX_VERTICES][MAX_VERTICES], int vertices, int startVertex) {
    int *visited = (int *)malloc(vertices * sizeof(int));
    for (int i = 0; i < vertices; ++i) {
        visited[i] = 0;
    }

    // double startTime, endTime;

    MPI_Barrier(MPI_COMM_WORLD);
    // startTime = MPI_Wtime();

    if (rank == ROOT_PROCESS) {
        visited[startVertex] = 1;
    }

    MPI_Bcast(visited, vertices, MPI_INT, ROOT_PROCESS, MPI_COMM_WORLD);

    for (int level = 0; level < vertices; ++level) {
        for (int i = rank; i < vertices; i += size) {
            if (visited[i] == 1) {
                // Process vertex i
                printf("Process %d visits vertex %d at level %d\n", rank, i, level);

                // Broadcast to neighbors
                for (int j = 0; j < vertices; ++j) {
                    if (graph[i][j] == 1 && visited[j] == 0) {
                        visited[j] = 1;
                    }
                }
            }
        }
        MPI_Bcast(visited, vertices, MPI_INT, ROOT_PROCESS, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // endTime = MPI_Wtime();

    // if (rank == ROOT_PROCESS) {
    //     printf("Total time taken: %f seconds\n", endTime - startTime);
    // }

    free(visited);
}




int main(int argc, char *argv[]) {
    int rank, size;
    int vertices = 6;
    int graph[MAX_VERTICES][MAX_VERTICES];

    const char *filename = "./data/matrices/mat3.csv";

    readGraphFromCSV(filename, graph, vertices);

    int startVertex = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double startTime, endTime;
    startTime = MPI_Wtime();
    parallelBFS(rank, size, graph, vertices, startVertex);
    endTime = MPI_Wtime();

    printf("Total time taken: %f seconds\n", endTime - startTime);

    MPI_Finalize();

    return 0;
}
