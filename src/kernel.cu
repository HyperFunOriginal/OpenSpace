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

// Tunable
constexpr float major_timestep = 10.f;
constexpr uint physics_substeps = 1u;
constexpr uint width = 512u;
constexpr uint height = 512u;

int main()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        Sleep(5000);
        return 1;
    }

    if (create_folder("SaveFolder"))
    {
        std::chrono::steady_clock clock;
        rng_state state(clock.now().time_since_epoch().count() >> 12u);
        smart_gpu_cpu_buffer<uint> temp(width * height);

        std::string line; writeline("Number of Particles?"); size_t particle_count = 100000u;
        std::getline(std::cin, line); particle_count = std::stoul(line);
        if (particle_count > 1000000u)
        {
            writeline("Are you sure you want to proceed with greater than 1 000 000 particles? \nRestart the program if not, and press enter otherwise.");
            std::getline(std::cin, line);
        }
        writeline("Starting Simulation:\n");

        double step_second = physics_substeps * one_second();
        gravitational_simulation simulation(particle_count);
        simulation.set_massive_cuboid(particle_count, 0, 5.97E+15f, make_float3(10000.f, 7000.f, 12000.f), domain_size_km * make_float3(.5f, .5f, .5f), make_float3(3.f, 0.f, 0.f), make_float3(-6E-4f));

        for (uint i = 0u; ; i++)
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
            save_octree_image(temp, simulation, width, height, ("SaveFolder/" + std::to_string(i) + ".png").c_str());
        }
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        Sleep(5000);
        return 1;
    }

    return 0;
}
