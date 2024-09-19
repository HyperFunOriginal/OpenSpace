#include "printstring_helper.h"
#include "gravitation.h"
#include "raymarch.h"

#include <chrono>
double one_second()
{
    std::chrono::steady_clock clock;
    const long long now = clock.now().time_since_epoch().count();
    _sleep(1000);
    return (double)(clock.now().time_since_epoch().count() - now);
}

// Implicit octree with morton indexing and a grid of sidelength 2^N, then raymarching straight through grids or collections thereof if there are no points there, i.e. cumsum over a certain patch is not increasing 

int main()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    std::chrono::steady_clock clock;
    rng_state state(clock.now().time_since_epoch().count() >> 12u);
    gravitational_simulation simulation(100000);

    simulation.set_massive_sphere(90000, 0, 5.97E+15f, 6378.f);
    simulation.sort_spatially();
    simulation.generate_gravitational_data();
    simulation.apply_gravitation();

    while (true)
        _sleep(1000);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
