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
constexpr float major_timestep = 15.f;
constexpr uint physics_substeps = 1u;
constexpr uint width = 1024u;
constexpr uint height = 1024u;

uint obtain_particle_count()
{
    std::string line; writeline("Number of Particles?"); size_t particle_count = 100000u;
    std::getline(std::cin, line); particle_count = std::stoul(line);
    if (particle_count > 1000000u)
    {
        writeline("Are you sure you want to proceed with greater than 1 000 000 particles? \nRestart the program if not, and press enter otherwise.");
        std::getline(std::cin, line);
    }
    writeline("\nStarting Simulation:");
    return particle_count;
}

int main()
{
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        Sleep(5000);
        return 1;
    }

    try
    {
        if (create_folder("SaveFolder"))
        {
            std::chrono::steady_clock clock;
            rng_state state(clock.now().time_since_epoch().count() >> 12u);
            smart_gpu_cpu_buffer<uint> temp(width * height);

            uint particle_count = obtain_particle_count();
            double step_second = physics_substeps * one_second();
            gravitational_simulation simulation(particle_count);

            std::vector<initial_kinematic_object> v = std::vector<initial_kinematic_object>();
            v.push_back(initial_kinematic_object(particle_count, initial_kinematic_object::geometry::GEOM_SPHERE, { 8000.f }, 6E+15f, make_float3(domain_size_km * .5f), make_float3(0.f), make_float3(6e-4f)));
            initialize_kinematic_objects(simulation, v);

            for (uint i = 0u; ; i++)
            {
                const long long now = clock.now().time_since_epoch().count();
                for (uint j = 0u; j < physics_substeps; j++)
                {
                    simulation.sort_spatially();
                    simulation.generate_gravitational_data();
                    simulation.apply_gravitation();
                    simulation.apply_kinematics_recenter(major_timestep / physics_substeps);
                }
                writeline("Saving image " + std::to_string(i) + ", Time taken per substep: " + std::to_string((clock.now().time_since_epoch().count() - now) * 1000.0 / step_second) + " ms");
                save_octree_image(temp, simulation, width, height, ("SaveFolder/" + std::to_string(i) + ".png").c_str());
            }
        }
    }
    catch (std::exception e)
    {
        fprintf(stderr, e.what());
        Sleep(5000);
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        Sleep(5000);
        return 1;
    }

    return 0;
}
