#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NUMBER_OF_POINTS 2000 // Number of points
#define D 4                    // Dimensions of data
#define K 5                    // Number of clusters

float *create_rand_data(int num_points)
{
    float *data = (float *)malloc(num_points * D * sizeof(float));
    FILE *fp;
    char filename[100];
    sprintf(filename, "./data/input/omp_data_%d_%d%s", num_points, D, ".csv");
    fp = fopen(filename, "w");

    for (int i = 0; i < num_points; i++)
    {
        for (int j = 0; j < D; j++)
        {
            data[i * D + j] = (float)rand() / (float)RAND_MAX;
            fprintf(fp, "%f", data[i * D + j]);

            if (j < D - 1)
            {
                fprintf(fp, ",");
            }
        }

        fprintf(fp, "\n");
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

#pragma omp parallel for
    for (int c = 1; c < k; c++)
    {
        float temp_dist = distance(point, centroid, dim);
        if (temp_dist < dist)
        {
#pragma omp critical
            {
                dist = temp_dist;
                cluster = c;
            }
        }
        centroid += dim;
    }

    return cluster;
}

void add_point(float *point, float *sum, int dim)
{
#pragma omp parallel for
    for (int i = 0; i < dim; i++)
    {
#pragma omp atomic
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

int main()
{
    srand(12345);

    float *points, *sums, *centroids;
    int *counts, *labels;
    points = (float *)malloc(NUMBER_OF_POINTS * D * sizeof(float));
    sums = (float *)malloc(K * D * sizeof(float));
    counts = (int *)malloc(K * sizeof(int));
    labels = (int *)malloc(NUMBER_OF_POINTS * sizeof(int));
    centroids = (float *)malloc(K * D * sizeof(float));

    points = create_rand_data(NUMBER_OF_POINTS);
    double starttime = omp_get_wtime();
#pragma omp parallel for
    for (int i = 0; i < K * D; i++)
    {
        centroids[i] = points[i];
    }

    float norm = 1.0;

    while (norm > 0.0001)
    {
#pragma omp parallel for
        for (int i = 0; i < K * D; i++)
        {
            sums[i] = 0.0;
        }

#pragma omp parallel for
        for (int i = 0; i < K; i++)
        {
            counts[i] = 0;
        }

#pragma omp parallel for
        for (int i = 0; i < NUMBER_OF_POINTS; i++)
        {
            int label = assign_point(points + i * D, centroids, K, D);
#pragma omp atomic write
            labels[i] = label;
#pragma omp atomic
            counts[label]++;
            add_point(points + i * D, sums + label * D, D);
        }

#pragma omp parallel for
        for (int i = 0; i < K * D; i++)
        {
            sums[i] = sums[i] / counts[i / D];
        }

        norm = distance(sums, centroids, K * D);
        // printf("Normalized distance: %f\n", norm);

#pragma omp parallel for
        for (int i = 0; i < K * D; i++)
        {
            centroids[i] = sums[i];
        }

        // print_centroids(centroids, K, D);
    }

    double endtime = omp_get_wtime();
    printf("Time taken: %f seconds\n", endtime - starttime);

    // Save labels in labels csv
    FILE *fp;
    char filename[100];
    sprintf(filename, "./data/output/omp_labels_%d_%d%s", NUMBER_OF_POINTS, D, ".csv");
    fp = fopen(filename, "w");

    for (int i = 0; i < NUMBER_OF_POINTS; i++)
    {
        for (int j = 0; j < D; j++)
        {
            fprintf(fp, "%f,", points[i * D + j]);
        }
        fprintf(fp, "%d\n", labels[i]);
    }

    fclose(fp);
    free(points);
    free(sums);
    free(counts);
    free(labels);
    free(centroids);

    return 0;
}
