#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAX_VERTICES 100

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

void parallelBFS(int graph[MAX_VERTICES][MAX_VERTICES], int vertices, int startVertex)
{
    int *visited = (int *)malloc(vertices * sizeof(int));
    for (int i = 0; i < vertices; ++i)
    {
        visited[i] = 0;
    }

    double startTime, endTime;

    startTime = omp_get_wtime();

    visited[startVertex] = 1;

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        for (int level = 0; level < vertices; ++level)
        {
#pragma omp for
            for (int i = 0; i < vertices; ++i)
            {
                if (visited[i] == 1)
                {
                    // Process vertex i
                    printf("Thread %d visits vertex %d at level %d\n", tid, i, level);

#pragma omp parallel for
                    for (int j = 0; j < vertices; ++j)
                    {
                        if (graph[i][j] == 1 && visited[j] == 0)
                        {
                            visited[j] = 1;
                        }
                    }
                }
            }
#pragma omp barrier
        }
    }

    endTime = omp_get_wtime();

    printf("Total time taken: %f seconds\n", endTime - startTime);

    free(visited);
}
int main()
{
    int vertices = 6;
    int graph[MAX_VERTICES][MAX_VERTICES];

    const char *filename = "./data/matrices/mat4.csv";

    readGraphFromCSV(filename, graph, vertices);

    int startVertex = 1;

    parallelBFS(graph, vertices, startVertex);

    return 0;
}