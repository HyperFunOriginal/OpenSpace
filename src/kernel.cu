#include "printstring_helper.h"
#include "cuda_headers/hydrodynamics_helper.h"
#include "cuda_headers/raymarch.h"

#include <chrono>

// Tunable
constexpr float major_timestep = 10.f;
constexpr float timestep_tolerance = major_timestep * 1E-4f;
constexpr uint width  = 512u;
constexpr uint height = 512u;

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

        hydrogravitational_simulation simulation(10000000);
        smart_gpu_buffer<float3> x_factor(simulation.particle_capacity);
        timestep_helper timestepper = timestep_helper();
        init_materials(simulation);

        std::vector<initial_thermodynamic_object> v = std::vector<initial_thermodynamic_object>(); 
        v.push_back(initial_thermodynamic_object(initial_kinematic_object::geometry::GEOM_SPHERE, { 6600.f }, 6E+15f, domain_size_km * make_float3(.25f), make_float3(4.f, 3.5f, 4.f), make_float3(0.f), 300.f, 1u));
        v.push_back(initial_thermodynamic_object(initial_kinematic_object::geometry::GEOM_SPHERE, { 5300.f }, 6E+15f, domain_size_km * make_float3(.75f), -make_float3(4.f, 3.5f, 4.f), make_float3(0.f), 300.f, 0u));
        initialize_thermodynamic_objects(simulation, v);

        simulation.sort_spatially();

        float average_time = 0.f, curr_timestep, next_timestep = 2.f;
        for (uint i = 0u; i < 6000; i++)
        {
            uint substeps_taken = 0u;
            for (float t = 0.f; t + timestep_tolerance < major_timestep; t += curr_timestep)
            {
                substeps_taken++;
                curr_timestep = next_timestep;
                apply_xsph_variant(x_factor, simulation, curr_timestep, 5e-4f);
                writeline("Ran physics step of timestep " + std::to_string(curr_timestep) + "s.");
               
                float tolerance = major_timestep - (t + curr_timestep);
                float tol_cond = tolerance < timestep_tolerance ? major_timestep : tolerance;
                next_timestep = timestepper.maximal_timestep_hydrodynamics_simulation(simulation);
                next_timestep = fmaxf(fminf(tol_cond / fmaxf(1.f, roundf(tol_cond / next_timestep)),
                    tolerance < timestep_tolerance ? INFINITY : tolerance), timestep_tolerance);
            }
            writeline("Saving image " + std::to_string(i) + " with " + std::to_string(substeps_taken) + " substeps.");
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
