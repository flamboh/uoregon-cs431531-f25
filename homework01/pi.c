#include <stdlib.h> 
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "common.h"
#include <inttypes.h>
#include <time.h>


#define PI 3.1415926535


void usage(int argc, char** argv);
double calcPi_Serial(int num_steps);
double calcPi_P1Atomic(int num_steps);
double calcPi_P1Critical(int num_steps);
double calcPi_P2Atomic(int num_steps);
double calcPi_P2Critical(int num_steps);


int main(int argc, char** argv)
{
    // get input values
    uint32_t num_steps = 100000;
    if(argc > 1) {
        num_steps = atoi(argv[1]);
    } else {
        usage(argc, argv);
        printf("using %"PRIu32"\n", num_steps);
    }
    // fprintf(stdout, "The first 10 digits of Pi are %0.10f\n", PI);


    // set up timer
    uint64_t start_t;
    uint64_t end_t;
    InitTSC();

    // printf("num_threads,num_steps,time_serial,time_atomic,time_critical,time_monte_carlo_atomic,time_monte_carlo_critical,pi_serial,pi_atomic,pi_critical,pi_monte_carlo_atomic,pi_monte_carlo_critical\n");
    printf("%d,%d,", omp_get_max_threads(), num_steps);
    // calculate in serial 
    start_t = ReadTSC();
    double Pi0 = calcPi_Serial(num_steps);
    end_t = ReadTSC();
    printf("%g,", ElapsedTime(end_t - start_t));
        
    // calculate in parallel with integration
    start_t = ReadTSC();
    double Pi1A = calcPi_P1Atomic(num_steps);
    end_t = ReadTSC();
    printf("%g,", ElapsedTime(end_t - start_t));
    
    // calculate in parallel with integration
    start_t = ReadTSC();
    double Pi1C = calcPi_P1Critical(num_steps);
    end_t = ReadTSC();
    printf("%g,", ElapsedTime(end_t - start_t));
        
    // calculate in parallel with Monte Carlo atomic
    start_t = ReadTSC();
    double Pi2A = calcPi_P2Atomic(num_steps);
    end_t = ReadTSC();
    printf("%g,", ElapsedTime(end_t - start_t));
    // calculate in parallel with Monte Carlo critical
    start_t = ReadTSC();
    double Pi2C = calcPi_P2Critical(num_steps);
    end_t = ReadTSC();
    printf("%g,", ElapsedTime(end_t - start_t));
    printf("%0.10f,", Pi0);
    printf("%0.10f,", Pi1A);
    printf("%0.10f,", Pi1C);
    printf("%0.10f,", Pi2A);
    printf("%0.10f\n", Pi2C);

    
    return 0;
}


void usage(int argc, char** argv)
{
    fprintf(stdout, "usage: %s <# steps>\n", argv[0]);
}

double calcPi_Serial(int num_steps) 
{
    // a lot faster than both parallel functions. Compiler optimizations?
    double pi = 0.0;
    double step = 1.0/num_steps;
    for (int i = 0; i < num_steps * 2; i++) {
        double x = -1 + (i * step);
        double h = sqrt(1 - (x * x));
        double w = step;
        pi += h * w;
    }    
    return pi * 2.0;
}

double calcPi_P1Atomic(int num_steps) // seems to vary a lot more than critical?
{
    double pi = 0.0;
    double step = 1.0/num_steps;
    #pragma omp parallel for
    for (int i = 0; i < num_steps * 2; i++) {
        double x = -1 + (i * step);
        double h = sqrt(1 - (x * x));
        double w = step;
        #pragma omp atomic
        pi += h * w;
    }    
    return pi * 2.0;
}

double calcPi_P1Critical(int num_steps)
{
    double pi = 0.0;
    double step = 1.0/num_steps;
    #pragma omp parallel for
    for (int i = 0; i < num_steps * 2; i++) {
        double x = -1 + (i * step);
        double h = sqrt(1 - (x * x));
        double w = step;
        #pragma omp critical
        pi += h * w;
    }    
    return pi * 2.0;
}

double calcPi_P2Atomic(int num_steps)
{
    int num_in = 0;
    #pragma omp parallel for
    for (int i = 0; i < num_steps; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;

        if (sqrt((x*x) + (y*y)) < 1) {
            #pragma omp atomic
            num_in++;
        }
    }
    return (((float)num_in) / num_steps) * 4;
}

double calcPi_P2Critical(int num_steps)
{
    int num_in = 0;
    #pragma omp parallel for
    for (int i = 0; i < num_steps; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;

        if (sqrt((x*x) + (y*y)) < 1) {
            #pragma omp critical
            num_in++;
        }
    }
    return (((float)num_in) / num_steps) * 4;
}
