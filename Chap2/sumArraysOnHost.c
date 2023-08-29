#include <stdlib.h>
#include <time.h>

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
}

void init(float *A, float *B, const int N)
{
    for (int i = 0; i < N; i++)
    {
        A[i] = ((float)i) + 0.1335f;
        B[i] = 1.50f * ((float)i) + 0.9383f;
    }
}

int main(int argc, char **argv)
{
    int nElem = 1024;

    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nElem * sizeof(float));
    h_B = (float *)malloc(nElem * sizeof(float));
    h_C = (float *)malloc(nElem * sizeof(float));

    init(h_A, h_B, nElem);

    sumArraysOnHost(h_A, h_B, h_C, nElem);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}