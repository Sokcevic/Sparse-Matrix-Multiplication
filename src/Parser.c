#include "Parser.h"
#include "ELL.h"

ELLMatrix* textFileToEll(char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) return NULL;
    if (feof(file)) {
        fclose(file);
        printf("Could not open file %s", filename);
        return NULL;
    }
    uint64_t rows;
    uint64_t cols;
    uint64_t nonZero;
    fscanf(file, "%lu,%lu,%lu\n", &rows, &cols, &nonZero);
    if (!rows || !cols || !nonZero) {
        fclose(file);
        printf("Format of first line was wrong");
        return NULL;
    }
    uint64_t size = cols * nonZero;
    ELLMatrix* result = createELL(rows, cols, nonZero);
    if (result == NULL) {
        fclose(file);
        return NULL;
    }
    for (uint64_t i = 0; i < size - 1; i++) {
        char peek = fgetc(file);
        if (peek == EOF) {
            fclose(file);
            freeELL(result);
            printf("Values array too short!");
            return NULL;
        }
        if (peek == '*') {
            fgetc(file);
        } else {
            ungetc(peek, file);
            fscanf(file, "%f,", &(result->values[i]));
        }
    }
    fscanf(file, "%f", &(result->values[size - 1]));
    fgetc(file);
    for (uint64_t i = 0; i < size - 1; i++) {
        char peek = fgetc(file);
        if (peek == EOF) {
            fclose(file);
            freeELL(result);
            printf("Indices array too short!");
            return NULL;
        }
        if (peek == '*') {
            fgetc(file);
        } else {
            ungetc(peek, file);
            fscanf(file, "%lu,", &(result->indicies[i]));
        }
    }
    fscanf(file, "%lu", &(result->indicies[size - 1]));
    fclose(file);
    return result;
}

void EllToTextFile(ELLMatrix* ell, char* filename) {
    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Could not open file %s", filename);
        return;
    }
    fprintf(file, "%lu,%lu,%lu\n", ell->rows, ell->cols, ell->nonZero);
    uint64_t size = ell->cols * ell->nonZero;
    for (uint64_t i = 0; i < size - 1; i++) {
        float next = ell->values[i];
        if (next == 0.0) {
            fprintf(file, "*,");
        } else {
            fprintf(file, "%f,", next);
        }
    }
    float lastF = ell->values[size-1];
    if (lastF == 0.0) {
        fprintf(file, "*\n");
    } else {
        fprintf(file, "%f\n", lastF);
    }

    for (uint64_t i = 0; i < size - 1; i++) {
        if (ell->values[i] == 0.0) {
            fprintf(file, "*,");
        } else {
            fprintf(file, "%lu,", ell->indicies[i]);
        }
    }
    if (ell->values[size - 1] == 0.0) {
        fprintf(file, "*");
    } else {
        fprintf(file, "%lu", ell->indicies[size - 1]);
    }
}
