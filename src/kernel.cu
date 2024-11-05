#include "printstring_helper.h"
#include "cuda_headers/hydrodynamics_helper.h"
#include "cuda_headers/raymarch.h"

#include <chrono>

// Tunable
constexpr float major_timestep = 30.f;
constexpr float timestep_tolerance = major_timestep * 1E-4f;
constexpr uint width  = 512u;
constexpr uint height = 512u;

double ticks_five_seconds()
{
    std::chrono::steady_clock clock;
    long long time = clock.now().time_since_epoch().count();
    Sleep(5000);
    return double(clock.now().time_since_epoch().count() - time);
}

void init_materials(hydrodynamics_simulation& simulation)
{
    // Iron
    simulation.materials_cpu_copy.cpu_buffer_ptr[0].bulk_modulus_GPa = 170.f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[0].limiting_heat_capacity_kJkgK = .47f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[0].standard_density_kgm3 = 7000.f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[0].molar_mass_kgmol = 5.2E-2f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[0].thermal_scale_K = 300.f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[0].stiffness_exponent = 4.6f;

    // Olivine
    simulation.materials_cpu_copy.cpu_buffer_ptr[1].bulk_modulus_GPa = 120.f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[1].limiting_heat_capacity_kJkgK = .7f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[1].standard_density_kgm3 = 4000.f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[1].molar_mass_kgmol = 9E-2f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[1].thermal_scale_K = 100.f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[1].stiffness_exponent = 4.6f;

    // Hydrogen
    simulation.materials_cpu_copy.cpu_buffer_ptr[2].bulk_modulus_GPa = .193f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[2].limiting_heat_capacity_kJkgK = 10.14f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[2].standard_density_kgm3 = 86.f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[2].molar_mass_kgmol = 2E-3f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[2].thermal_scale_K = 40.f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[2].stiffness_exponent = 3.1f;

    // Water
    simulation.materials_cpu_copy.cpu_buffer_ptr[3].bulk_modulus_GPa = 2.1f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[3].limiting_heat_capacity_kJkgK = 2.1f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[3].standard_density_kgm3 = 1000.f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[3].molar_mass_kgmol = 1.8E-2f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[3].thermal_scale_K = 50.f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[3].stiffness_exponent = 4.0f;

    // Helium
    simulation.materials_cpu_copy.cpu_buffer_ptr[4].bulk_modulus_GPa = .03f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[4].limiting_heat_capacity_kJkgK = 3.2f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[4].standard_density_kgm3 = 210.f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[4].molar_mass_kgmol = 4E-3f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[4].thermal_scale_K = 20.f;
    simulation.materials_cpu_copy.cpu_buffer_ptr[4].stiffness_exponent = 3.9f;
    simulation.copy_materials_to_gpu();
}
void run_sph_sim()
{
    if (create_folder("SaveFolder"))
    {
        smart_gpu_cpu_buffer<uint> temp(width * height);

        hydrogravitational_simulation simulation(1000000);
        timestep_helper timestepper = timestep_helper();
        init_materials(simulation);

        std::vector<initial_thermodynamic_object> v = std::vector<initial_thermodynamic_object>(); 
        v.push_back(initial_thermodynamic_object(initial_kinematic_object::geometry::GEOM_SPHERE, { 0.f }, 6e+15f, domain_size_km * make_float3(.3f, .3f, .5f), make_float3(1.f, 3.f, 0.f), make_float3(0.f), 60000.f, 1u));
        v.push_back(initial_thermodynamic_object(initial_kinematic_object::geometry::GEOM_SPHERE, { 0.f }, 6e+15f, domain_size_km * make_float3(.7f, .7f, .5f), make_float3(-1.f, -3.f, 0.f), make_float3(0.f), 60000.f, 1u));
        initialize_thermodynamic_objects_hydrostatic_equilibrium(simulation, v);
        writeline("Initialized. Awaiting simulation start in 5 seconds.");

        double ticks_ms = ticks_five_seconds() * 2E-4; std::chrono::steady_clock clock;
        float average_time = 0.f, curr_timestep, next_timestep = timestep_tolerance;
        for (uint i = 0u; i < 3000; i++)
        {
            uint substeps_taken = 0u;
            for (float t = 0.f; t + timestep_tolerance < major_timestep; t += curr_timestep)
            {
                substeps_taken++;
                curr_timestep = next_timestep;
                long long time = clock.now().time_since_epoch().count();
                simulation.apply_complete_timestep(curr_timestep, 1e-4f);
                time = clock.now().time_since_epoch().count() - time;
                writeline("Ran physics step of timestep " + std::to_string(curr_timestep) + "s. Time taken: " + std::to_string(time / ticks_ms) + " ms");
               
                float tolerance = major_timestep - (t + curr_timestep);
                float tol_cond = tolerance < timestep_tolerance ? major_timestep : tolerance;
                next_timestep = timestepper.maximal_timestep_hydrodynamics_simulation(simulation);
                if (next_timestep < timestep_tolerance) { goto RETURN; }
                next_timestep = fmaxf(fminf(tol_cond / ceilf(tol_cond / next_timestep),
                    tolerance < timestep_tolerance ? INFINITY : tolerance), timestep_tolerance);
            }
            writeline("Saving image " + std::to_string(i) + " with " + std::to_string(substeps_taken) + " substeps.");
            save_octree_image(temp, simulation, width, height, ("SaveFolder/" + std::to_string(i) + ".png").c_str());
        }

    RETURN:
        timestepper.destroy();
        simulation.destroy();
        temp.destroy();
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
