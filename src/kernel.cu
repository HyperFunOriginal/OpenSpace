#include "printstring_helper.h"
#include "gravitation.h"
#include "hydrodynamics.h"
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
constexpr uint physics_substeps = 6u;
constexpr uint width  = 512u;
constexpr uint height = 512u;

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

void run_grav_sim()
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
        v.push_back(initial_kinematic_object(initial_kinematic_object::geometry::GEOM_SPHERE, { 6378.f }, 6.0E+15f, domain_size_km * make_float3(.5f, .75f, .5f), make_float3(3.5f, 0.f, 0.f), make_float3(0.f, 0.f, 1.25e-3f)));
        v.push_back(initial_kinematic_object(initial_kinematic_object::geometry::GEOM_SPHERE, { 7000.f }, 7.9321666e+15f, domain_size_km * make_float3(.5f, .25f, .5f), make_float3(-3.5f, 0.f, 0.f), make_float3(0.f, 0.f, 1.25e-3f)));
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

        uint particle_count = obtain_particle_count();
        double step_second = physics_substeps * one_second();
        hydrogravitational_simulation simulation(particle_count);

        smart_cpu_buffer<material_properties> mats(1);
        mats.cpu_buffer_ptr[0].bulk_modulus_GPa = 90.f;
        mats.cpu_buffer_ptr[0].limiting_heat_capacity_kJkgK = .7f;
        mats.cpu_buffer_ptr[0].standard_density_kgm3 = 4000.f;
        mats.cpu_buffer_ptr[0].standard_volume_m3mol = 2.5E-5f;
        mats.cpu_buffer_ptr[0].thermal_scale_K = 100.f;
        simulation.copy_materials_to_gpu(mats);

        std::vector<initial_kinematic_object> v = std::vector<initial_kinematic_object>();
        v.push_back(initial_kinematic_object(initial_kinematic_object::geometry::GEOM_SPHERE, { 6378.f }, 6.0E+15f, domain_size_km * make_float3(.3f, .3f, .5f), make_float3(0.f), make_float3(0.f, 1E-4f, 0.f)));
        v.push_back(initial_kinematic_object(initial_kinematic_object::geometry::GEOM_SPHERE, { 3000.f }, 5E+14f, domain_size_km * make_float3(.85f, .85f, .5f), -make_float3(10.f, 10.f, 0.f), make_float3(1E-4f, 0.f, 0.f)));
        initialize_kinematic_objects(simulation, v);

        for (uint i = 0u; ; i++)
        {
            const long long now = clock.now().time_since_epoch().count();
            for (uint j = 0u; j < physics_substeps; j++)
                simulation.apply_complete_timestep(major_timestep / physics_substeps, 0.002f);
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
