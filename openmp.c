#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

// CONSTANTS
#define MAX_PAGES 800 // Integer in [MIN_PAGES, +inf)
#define MIN_PAGES 2 // Integer in [2, MAX_PAGES]
#define WEIGHT 0.15 // Real in (0, 1), best at 0.15
#define ERROR 0.0001 // Real in (0, +inf), best at 0.0001

// PROTOTYPES
int get_num_pages();
void init_link_matrix(const char *filename, int num_pages, float link_matrix[][num_pages]);
void scalar_multiplication(float *matrix, int num_rows, int num_cols, float scalar);
void column_multiplication(float *matrix, int num_rows, int num_cols, float *column);
void addition(float *matrix1, float *matrix2, int num_rows, int num_cols);
float norm(float *column, int num_rows);
void print_standings(float score_column[], int num_pages);

int main() {
    // INPUT
    printf("Let's start by creating a model of the web.\n");
    int num_pages = MAX_PAGES;
    float link_matrix[num_pages][num_pages];
    init_link_matrix("input.txt", num_pages, link_matrix);
    
    clock_t start, end;
    double cpu_time_used;

    start = clock();

    // CONVERGENCE LOOP

    // Initialize mean column and score column
    float mean_column[num_pages], score_column[num_pages];
    for (int i = 0; i < num_pages; i++) {
        float entry = 1 / (float) num_pages;
        mean_column[i] = entry;
        score_column[i] = entry;
    }

    // Weigh link matrix and mean column
    scalar_multiplication((float *)link_matrix, num_pages, num_pages, 1 - WEIGHT);
    scalar_multiplication(mean_column, num_pages, 1, WEIGHT);

    float score_norm;
    do {
        // Store score column before operations
        float score_diff[num_pages];
        for (int i = 0; i < num_pages; i++)
            score_diff[i] = score_column[i];

        // Multiply score column by weighted link matrix
        column_multiplication((float *)link_matrix, num_pages, num_pages, score_column);

        // Add weighted mean column to score column
        addition(score_column, mean_column, num_pages, 1);

        // Subtract previous score column from score column
        scalar_multiplication(score_diff, num_pages, 1, -1);
        addition(score_diff, score_column, num_pages, 1);

        // Calculate norm of the difference
        score_norm = norm(score_diff, num_pages);

        // Repeat until score norm is smaller than error
    } while (score_norm > ERROR);
    
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    // OUTPUT
    printf("Here are the standings:\n");
    print_standings(score_column, num_pages);
    
    printf("Execution time: %f seconds\n", cpu_time_used);
}

// INPUT

// Get number of pages
int get_num_pages() {
    int num_pages;
    do {
        printf("How many pages does your web have? ");
        scanf("%d", &num_pages);
        if (num_pages < MIN_PAGES)
            printf("Your web has too few pages, try %d or more.\n", MIN_PAGES);
        else if (num_pages > MAX_PAGES)
            printf("Your web has too many pages, try %d or less.\n", MAX_PAGES);
    } while (num_pages < MIN_PAGES || num_pages > MAX_PAGES);
    return num_pages;
}

// Initialize link matrix
void init_link_matrix(const char *filename, int num_pages, float link_matrix[][num_pages]) {
    // Set all entries to 0
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < num_pages; i++) {
        for (int j = 0; j < num_pages; j++) {
            link_matrix[i][j] = 0;
        }
    }
    
    FILE *file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file\n");
        exit(EXIT_FAILURE);
    }

    // Links
    for (int i = 0; i < num_pages; i++) {
        // Number of links
        int num_links;
        fscanf(file, "%d", &num_links);
        // Linked pages
        for (int j = 0; j < num_links; j++) {
            int page_num;
            fscanf(file, "%d", &page_num);
            link_matrix[page_num - 1][i] = 1 / (float) num_links;
        }
    }
    fclose(file);
}

// MATRIX OPERATIONS

// Multiply matrix by scalar
void scalar_multiplication(float *matrix, int num_rows, int num_cols, float scalar) {
    int num_entries = num_rows * num_cols;
    #pragma omp parallel for
    for (int i = 0; i < num_entries; i++) {
        matrix[i] *= scalar;
    }
}

// Multiply matrix by column
void column_multiplication(float *matrix, int num_rows, int num_cols, float *column) {
    float product[num_cols];
    #pragma omp parallel for
    for (int i = 0; i < num_rows; i++) {
        float sum = 0;
        for (int j = 0; j < num_cols; j++) {
            sum += matrix[i * num_cols + j] * column[j];
        }
        product[i] = sum;
    }
    #pragma omp parallel for
    for (int i = 0; i < num_cols; i++) {
        column[i] = product[i];
    }
}

// Sum two matrices
void addition(float *matrix1, float *matrix2, int num_rows, int num_cols) {
    int num_entries = num_rows * num_cols;
    #pragma omp parallel for
    for (int i = 0; i < num_entries; i++) {
        matrix1[i] += matrix2[i];
    }
}

// Return Euclidean norm of column
float norm(float *column, int num_rows) {
    float sum = 0;
    for (int i = 0; i < num_rows; i++) {
        sum += pow(column[i], 2);
    }
    return sqrt(sum);
}

// OUTPUT

// Print standings
void print_standings(float score_column[], int num_pages) {
    for (int i = 0; i < num_pages; i++) {
        float max_score = -1.0;
        int page_num = -1;
        for (int j = 0; j < num_pages; j++) {
            if (score_column[j] > max_score) {
                max_score = score_column[j];
                page_num = j;
            }
        }
        score_column[page_num] = 0; // To avoid picking the same page twice
        printf("%d. Page %d: %f\n", i + 1, page_num + 1, max_score);
    }
}

