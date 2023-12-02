#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NUMBER_OF_POINTS 10000 // Number of points
#define D 3                    // Dimensions of data
#define K 4                    // Number of clusters

float *create_rand_data(int num_points)
{
    float *data = (float *)malloc(num_points * sizeof(float));
    FILE *fp;
    char filename[100];
    sprintf(filename, "./data/input/sequantial_data_%d_%d_%d%s", (num_points / D), D, ".csv");
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

int main()
{
    srand(12345);

    float *points, *sums, *centroids;
    int *counts, *labels;
    points = malloc(NUMBER_OF_POINTS * D * sizeof(float));
    sums = malloc(K * D * sizeof(float));
    counts = malloc(K * sizeof(int));
    labels = malloc(NUMBER_OF_POINTS * sizeof(int));
    centroids = malloc(K * D * sizeof(float));

    points = create_rand_data(NUMBER_OF_POINTS);

    for (int i = 0; i < K * D; i++)
    {
        centroids[i] = points[i];
    }

    float norm = 1.0;

    while (norm > 0.0001)
    {
        for (int i = 0; i < K * D; i++)
        {
            sums[i] = 0.0;
        }
        for (int i = 0; i < K; i++)
        {
            counts[i] = 0;
        }
        float *point = points;
        for (int i = 0; i < NUMBER_OF_POINTS; i++, point += D)
        {
            int label = assign_point(point, centroids, K, D);
            counts[label]++;
            add_point(point, &sums[label * D], D);
            labels[i] = label;
        }

        for (int i = 0; i < K * D; i++)
        {
            sums[i] = sums[i] / counts[i / D];
        }
        norm = distance(sums, centroids, K * D);
        printf("Normalized distance: %f\n", norm);

        for (int i = 0; i < K * D; i++)
        {
            centroids[i] = sums[i];
        }
        print_centroids(centroids, K, D);
    }

    // Save labels in labels csv
    FILE *fp;
    char filename[100];
    sprintf(filename, "./data/output/sequential_labels_%d_%d%s", NUMBER_OF_POINTS, D, ".csv");
    fp = fopen(filename, "w");
    for (int i = 0; i < NUMBER_OF_POINTS; i++)
    {
        for (int j = 0; j < D; j++)
        {
            fprintf(fp, "%f,", points[i * D + j]);
        }
        printf("Label: %d\n", labels[i]);
        fprintf(fp, "%d\n", labels[i]);
    }

    free(points);
    free(sums);
    free(counts);
    free(labels);
    free(centroids);

    return 0;
}
