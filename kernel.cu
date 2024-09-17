#include "printstring_helper.h"
#include "io_img_helper.h"
#include "gravitation.h"

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

    simulation.set_sphere(90000, 0, 5.97E+15f, 6378.f);
    simulation.sort_spatially();
    simulation.generate_gravitational_data();
    simulation.octree.copy_to_cpu();

    for (uint i = __morton_index(make_float3(0, domain_size_km, domain_size_km) * .5f); i < grid_side_length * grid_side_length * grid_side_length; i = add_morton_indices(i, 1))
    {
        grid_cell_ensemble s = simulation.octree.cpu_buffer_ptr[start_index(grid_dimension_pow) + i];
        writeline(std::to_string(s.deviatoric_pos_km.x + __cell_pos_from_index(i, grid_dimension_pow).x) + ": " + std::to_string(s.average_acceleration_ms2.x));
    }
   
    while (true)
        _sleep(1000);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
