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
constexpr float major_timestep = 6.f;
constexpr uint physics_substeps = 5u;
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
void init_materials(smart_cpu_buffer<material_properties>& buffer)
{
    // Iron
    buffer.cpu_buffer_ptr[0].bulk_modulus_GPa = 160.f;
    buffer.cpu_buffer_ptr[0].limiting_heat_capacity_kJkgK = .47f;
    buffer.cpu_buffer_ptr[0].standard_density_kgm3 = 7000.f;
    buffer.cpu_buffer_ptr[0].molar_mass_kgmol = 5.2E-2f;
    buffer.cpu_buffer_ptr[0].thermal_scale_K = 300.f;
    buffer.cpu_buffer_ptr[0].stiffness_exponent = 4.6f;

    // Rock
    buffer.cpu_buffer_ptr[1].bulk_modulus_GPa = 90.f;
    buffer.cpu_buffer_ptr[1].limiting_heat_capacity_kJkgK = .7f;
    buffer.cpu_buffer_ptr[1].standard_density_kgm3 = 4000.f;
    buffer.cpu_buffer_ptr[1].molar_mass_kgmol = 9E-2f;
    buffer.cpu_buffer_ptr[1].thermal_scale_K = 100.f;
    buffer.cpu_buffer_ptr[1].stiffness_exponent = 4.6f;

    // H2
    buffer.cpu_buffer_ptr[2].bulk_modulus_GPa = .25f;
    buffer.cpu_buffer_ptr[2].limiting_heat_capacity_kJkgK = 12.4f;
    buffer.cpu_buffer_ptr[2].standard_density_kgm3 = 86.f;
    buffer.cpu_buffer_ptr[2].molar_mass_kgmol = 2E-3f;
    buffer.cpu_buffer_ptr[2].thermal_scale_K = 40.f;
    buffer.cpu_buffer_ptr[2].stiffness_exponent = 3.4f;

    // H2O
    buffer.cpu_buffer_ptr[3].bulk_modulus_GPa = 2.1f;
    buffer.cpu_buffer_ptr[3].limiting_heat_capacity_kJkgK = 2.1f;
    buffer.cpu_buffer_ptr[3].standard_density_kgm3 = 1000.f;
    buffer.cpu_buffer_ptr[3].molar_mass_kgmol = 1.8E-2f;
    buffer.cpu_buffer_ptr[3].thermal_scale_K = 50.f;
    buffer.cpu_buffer_ptr[3].stiffness_exponent = 4.0f;
}

void run_sph_sim()
{
    if (create_folder("SaveFolder"))
    {
        std::chrono::steady_clock clock;
        rng_state state(clock.now().time_since_epoch().count() >> 12u);
        smart_gpu_cpu_buffer<uint> temp(width * height);

        double step_second = physics_substeps * one_second();
        hydrogravitational_simulation simulation(5000000);
        init_materials(simulation.materials_cpu_copy);
        simulation.copy_materials_to_gpu();

        std::vector<initial_thermodynamic_object> v = std::vector<initial_thermodynamic_object>();
        v.push_back(initial_thermodynamic_object(initial_kinematic_object::geometry::GEOM_SPHERE, { 6378.f }, 5.97E+15f, domain_size_km * make_float3(.7f, .38f, .5f), make_float3(-1.f, 0.f, 0.f), make_float3(0.f), 300.f, 1u));
        v.push_back(initial_thermodynamic_object(initial_kinematic_object::geometry::GEOM_SPHERE, { 4000.f }, 1.976578e+15f, domain_size_km * make_float3(.3f, .43f, .5f), make_float3(7.f, 0.f, 0.f), make_float3(0.f), 300.f, 0u));
        initialize_thermodynamic_objects(simulation, v);

        for (uint i = 0u; i < 6000; i++)
        {
            const long long now = clock.now().time_since_epoch().count();
            for (uint j = 0u; j < physics_substeps; j++)
                simulation.apply_complete_timestep(major_timestep / physics_substeps, 0.00025f);
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

    run_sph_sim();

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        Sleep(5000);
        return 1;
    }

    return 0;
}
