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
constexpr uint frames = 2100u;
constexpr float major_timestep = 40.f;
constexpr uint physics_substeps = 3u;

int main()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return 1;
    }

    if (create_folder("SaveFolder"))
    {
        std::chrono::steady_clock clock;
        rng_state state(clock.now().time_since_epoch().count() >> 12u);
        gravitational_simulation simulation(3000000);
        smart_gpu_cpu_buffer<uint> temp(512 * 512);

        double step_second = physics_substeps * one_second();
        simulation.set_massive_sphere(3000000, 0, 5.97E+15f, 10000.f, make_float3(domain_size_km * .5f), make_float3(0.f), make_float3(0.f, 0.f, 7.2E-4f));
        
        for (uint i = 0u; i < frames; i++)
        {
            const long long now = clock.now().time_since_epoch().count();
            for (uint j = 0u; j < physics_substeps; j++)
            {
                simulation.sort_spatially();
                simulation.generate_gravitational_data();
                simulation.apply_gravitation();
                simulation.apply_kinematics(major_timestep / physics_substeps);
            }
            writeline("Saving image " + std::to_string(i) + ", Time taken per substep: " + std::to_string((clock.now().time_since_epoch().count() - now) * 1000.0 / step_second) + " ms");
            save_octree_image(temp, simulation, 512, 512, ("SaveFolder/" + std::to_string(i) + ".png").c_str());
        }
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
