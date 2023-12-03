#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <conio.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <mpi.h>
#include <omp.h>

#define NUMBER_OF_POINTS 1000000 // Number of points
#define D 6                      // Dimensions of data
#define K 12                     // Number of cluserts

#define POINTS_PER_PROCESS NUMBER_OF_POINTS / D // Number of processes

// Function prototypes
float *create_rand_data(int num_points, int num_proc)
{
    float *data = (float *)malloc(num_points * sizeof(float));
    FILE *fp;
    char filename[100];
    sprintf(filename, "./data/input/mpi_data_%d_%d_%d%s", num_proc, (num_points / num_proc) / D, D, ".csv");
    fp = fopen(filename, "w");

    for (int i = 0; i < num_points; i++)
    {
        data[i] = (float)rand() / (float)RAND_MAX;
        fprintf(fp, "%f", data[i]);

        if (i < num_points - 1)
        {
            fprintf(fp, ",");
        }

        if ((i + 1) % D == 0 && i < num_points - 1)
        {
            fprintf(fp, "\n");
        }
    }

    fclose(fp);
    return data;
}

float distance(float *vec1, float *vec2, int dim)
{
    float dist = 0.0;
    for (int i = 0; i < dim; i++)
    {
        dist += ((vec1[i] - vec2[i]) * (vec1[i] - vec2[i]));
    }
    return dist;
}

int assign_point(float *point, float *centroids, int k, int dim)
{
    int cluster = 0;
    float dist = distance(point, centroids, dim);
    float *centroid = centroids + dim;
    for (int c = 1; c < k; c++, centroid += dim)
    {
        float temp_dist = distance(point, centroid, dim);
        if (temp_dist < dist)
        {
            dist = temp_dist;
            cluster = c;
        }
    }
    return cluster;
}

void add_point(float *point, float *sum, int dim)
{
    for (int i = 0; i < dim; i++)
    {
        sum[i] += point[i];
    }
}

void print_centroids(float *centroids, int k, int dim)
{
    float *p = centroids;
    printf("--------------------------CENTROIDS--------------------------\n");
    for (int i = 0; i < k; i++)
    {
        printf("Centroid %d: ", i);
        for (int j = 0; j < dim; j++, p++)
        {
            printf("%f ", *p);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char const *argv[])
{

    srand(12345);
    MPI_Init(NULL, NULL);
    int rank, num_process;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);

    double start_time_total = MPI_Wtime();

    float *points, *sums, *centroids;
    int *counts, *labels;
    points = malloc(POINTS_PER_PROCESS * D * sizeof(float));
    sums = malloc(K * D * sizeof(float));
    counts = malloc(K * sizeof(int));
    labels = malloc(POINTS_PER_PROCESS * sizeof(int));
    centroids = malloc(K * D * sizeof(float));

    float *all_points = NULL;
    float *all_sums = NULL;
    int *all_counts = NULL;

    int *all_labels;

    if (rank == 0)
    {
        all_points = create_rand_data(POINTS_PER_PROCESS * num_process * D, num_process);
        for (int i = 0; i < K * D; i++)
        {
            centroids[i] = all_points[i];
        }
        // print_centroids(centroids, K, D);
        all_sums = malloc(K * D * sizeof(float));
        all_counts = malloc(K * sizeof(int));
        all_labels = malloc(POINTS_PER_PROCESS * num_process * sizeof(int));
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time_computation = MPI_Wtime();

    MPI_Scatter(all_points, POINTS_PER_PROCESS * D, MPI_FLOAT, points, POINTS_PER_PROCESS * D, MPI_FLOAT, 0, MPI_COMM_WORLD);

    double end_time_computation = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time_communication = MPI_Wtime();

    float norm = 1.0;

    while (norm > 0.0001)
    {
        MPI_Bcast(centroids, K * D, MPI_FLOAT, 0, MPI_COMM_WORLD);
        for (int i = 0; i < K * D; i++)
        {
            sums[i] = 0.0;
        }
        for (int i = 0; i < K; i++)
        {
            counts[i] = 0;
        }
        float *point = points;
        for (int i = 0; i < POINTS_PER_PROCESS; i++, point += D)
        {
            int label = assign_point(point, centroids, K, D);
            counts[label]++;
            add_point(point, &sums[label * D], D);
            // labels[i] = label;
        }
        MPI_Reduce(sums, all_sums, K * D, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(counts, all_counts, K, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            for (int i = 0; i < K; i++)
            {
                for (int j = 0; j < D; j++)
                {
                    all_sums[i * D + j] = all_sums[i * D + j] / all_counts[i];
                }
                // all_labels[i] = all_labels[i] / all_counts[i];
            }
            norm = distance(all_sums, centroids, K * D);
            // printf("Normalized distance: %f\n", norm);
            for (int i = 0; i < K * D; i++)
            {
                centroids[i] = all_sums[i];
            }
            print_centroids(centroids, K, D);
        }
        MPI_Bcast(&norm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    float *point = points;
    for (int i = 0; i < POINTS_PER_PROCESS; i++, point += D)
    {
        int label = assign_point(point, centroids, K, D);
        labels[i] = label;
    }
    MPI_Gather(labels, POINTS_PER_PROCESS, MPI_INT, all_labels, POINTS_PER_PROCESS, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time_communication = MPI_Wtime();

    if ((rank == 0) && 1)
    {
        printf("Computation time: %f seconds\n", end_time_computation - start_time_computation);
        printf("Communication time: %f seconds\n", end_time_communication - start_time_communication);
        printf("Total time: %f seconds\n", MPI_Wtime() - start_time_total);

        // Save labels in labels csv
        float *point = all_points;
        FILE *fp;
        char filename[100];
        sprintf(filename, "./data/output/mpi_labels_%d_%d_%d%s", num_process, POINTS_PER_PROCESS, D, ".csv");
        fp = fopen(filename, "w");
        for (int i = 0; i < num_process * POINTS_PER_PROCESS; i++, point += D)
        {
            for (int j = 0; j < D; j++)
            {
                fprintf(fp, "%f,", points[i * D + j]);
            }
            fprintf(fp, "%d\n", labels[i]);
        }
    }

    free(all_points);
    free(sums);
    free(counts);
    free(labels);
    free(centroids);
    free(all_sums);
    free(all_counts);
    free(all_labels);

    MPI_Finalize();

    return 0;
}
