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
constexpr uint frames = 2100u;
constexpr float major_timestep = 10.f;
constexpr uint physics_substeps = 1u;

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
        camera cam_1 = camera(domain_size_km * make_float3(.5f, .5f, -1.f), make_float3(1.f, 0.f, 0.f), make_float3(0.f, 1.f, 0.f), 1.f);
        rng_state state(clock.now().time_since_epoch().count() >> 12u);
        gravitational_simulation simulation(10000000);
        gravitational_renderer renderer(simulation);
        smart_gpu_cpu_buffer<uint> temp(512 * 512);

        double step_second = physics_substeps * one_second();
        simulation.set_massive_sphere(10000000, 0, 5.97E+15f, 8000.f, domain_size_km * make_float3(.5f, .5f, .5f), make_float3(3.f, 0.f, 0.f), make_float3(-6E-4f));

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
            //raymarch_render<gravitational_renderer>(temp, simulation, cam_1, renderer, 512, 512, ("SaveFolder/" + std::to_string(i) + ".png").c_str());
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
