#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <conio.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <omp.h>

#define NUMBER_OF_POINTS 10000 // Number of points
#define D 3                    // Dimensions of data
#define K 4                    // Number of cluserts

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
    // Parallel Implementation of Kmeans algorithm using OpenMP

    // Initialize data
    float *data = create_rand_data(NUMBER_OF_POINTS, D);
    float *centroids = (float *)malloc(K * D * sizeof(float));

#pragma omp parallel for
    for (int i = 0; i < K * D; i++)
    {
        centroids[i] = data[i + rand() % NUMBER_OF_POINTS];
    }

    // Initialize clusters
    int *clusters = (int *)malloc(NUMBER_OF_POINTS * sizeof(int));
    for (int i = 0; i < NUMBER_OF_POINTS; i++)
    {
        clusters[i] = -1;
    }

    // Initialize cluster sizes
    int *cluster_sizes = (int *)malloc(K * sizeof(int));
    for (int i = 0; i < K; i++)
    {
        cluster_sizes[i] = 0;
    }

    // Initialize cluster sums
    float *cluster_sums = (float *)malloc(K * D * sizeof(float));
    for (int i = 0; i < K * D; i++)
    {
        cluster_sums[i] = 0.0;
    }

    // Initialize cluster sums
    float *cluster_means = (float *)malloc(K * D * sizeof(float));
    for (int i = 0; i < K * D; i++)
    {
        cluster_means[i] = 0.0;
    }

    // Algorithms
    int num_iterations = 0;
    int converged = 0;
    double start_time = omp_get_wtime();

    while (!converged)
    {
// Assign points to clusters
#pragma omp parallel for
        for (int i = 0; i < NUMBER_OF_POINTS; i++)
        {
            clusters[i] = assign_point(data + i * D, centroids, K, D);
        }

// Reset cluster sizes
#pragma omp parallel for
        for (int i = 0; i < K; i++)
        {
            cluster_sizes[i] = 0;
        }

// Reset cluster sums
#pragma omp parallel for
        for (int i = 0; i < K * D; i++)
        {
            cluster_sums[i] = 0.0;
        }

// Add points to clusters
#pragma omp parallel for
        for (int i = 0; i < NUMBER_OF_POINTS; i++)
        {
            int cluster = clusters[i];
            add_point(data + i * D, cluster_sums + cluster * D, D);
            cluster_sizes[cluster]++;
        }

// Compute cluster means
#pragma omp parallel for
        for (int i = 0; i < K; i++)
        {
            for (int j = 0; j < D; j++)
            {
                cluster_means[i * D + j] = cluster_sums[i * D + j] / cluster_sizes[i];
            }
        }

        // Check for convergence
        // Check for convergence (use a threshold instead of exact equality)
        converged = 1;
        for (int i = 0; i < K * D; i++)
        {
            if (isnan(cluster_means[i]) || fabs(cluster_means[i] - centroids[i]) > 0.0001)
            {
                converged = 0;
                break;
            }
        }

        // Update centroids only for non-empty clusters
        for (int i = 0; i < K * D; i++)
        {
            if (!isnan(cluster_means[i]) && cluster_sizes[i / D] > 0)
            {
                centroids[i] = cluster_means[i];
            }
        }

        num_iterations++;
    }

    double end_time = omp_get_wtime();

    // Print results
    printf("Number of iterations: %d\n", num_iterations);
    printf("Time taken: %f seconds\n", end_time - start_time);
    print_centroids(centroids, K, D);

    return 0;
}
