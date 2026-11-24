/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/material_model/growing_fault.h>
#include <aspect/simulator/solver/midpoint_fault.h>
#include <aspect/phase_field/bernstein_bounds.h>
#include <aspect/geometry_model/box.h>
#include <deal.II/fe/fe_values.h>
#include <numeric>

namespace aspect::GrowingFault
{
  template <int dim>
  void Material<dim>::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Material model");
    prm.enter_subsection("Growing fault");
    prm.declare_entry("Seed tangent angle","0",Patterns::Double(),"Angle in degrees counterclockwise from x.");
    prm.declare_entry("Forward seed tip","0,0",Patterns::List(Patterns::Double(),2,2),"Physical tip of the initial contiguous seed in metres.");
    prm.declare_entry("Section quadrature order","6",Patterns::Integer(2),"Gauss points per strip subinterval and chord.");
    prm.declare_entry("Initial age","10",Patterns::Double(0),"Initial state age in seconds.");
    prm.declare_entry("Shear modulus","1e6",Patterns::Double(0),"Shear modulus in Pa.");
    prm.declare_entry("Viscosity","1e8",Patterns::Double(0),"Maxwell viscosity in Pa s.");
    prm.declare_entry("Cohesion","1000",Patterns::Double(0),"Cohesive strength in Pa.");
    prm.declare_entry("Reference friction coefficient","0.6",Patterns::Double(),"Dimensionless reference friction coefficient.");
    prm.declare_entry("Direct effect","0.015",Patterns::Double(0),"Positive direct effect coefficient a.");
    prm.declare_entry("Evolution effect","0.02",Patterns::Double(0),"Nonnegative evolution coefficient b.");
    prm.declare_entry("Reference speed","0.001",Patterns::Double(0),"Reference speed in m/s.");
    prm.declare_entry("Slip distance","0.01",Patterns::Double(0),"State evolution distance in metres.");
    prm.declare_entry("Radiation damping","1000",Patterns::Double(0),"Radiation impedance in Pa s/m.");
    prm.leave_subsection();
    prm.leave_subsection();
  }

  template <int dim>
  void Material<dim>::parse_parameters(ParameterHandler &prm)
  {
    const auto &p=this->get_parameters();
    const auto &energy=p.surface_energy_model;
    energy.validate();
    field=this->introspection().compositional_index_for_name(energy.composition);
    AssertThrow(dim==2 && &this->get_material_model()==this && !p.enable_elasticity &&
                p.nonlinear_solver==Parameters<dim>::NonlinearSolver::single_Advection_iterated_Newton_Stokes &&
                p.compositional_field_methods[field]==Parameters<dim>::AdvectionFieldMethod::midpoint_fault_phase &&
                std::count(p.compositional_field_methods.begin(),p.compositional_field_methods.end(),
                           Parameters<dim>::AdvectionFieldMethod::midpoint_fault_phase)==1 &&
                !p.explicit_compositions.selected[field] &&
                p.nonlinear_solver_failure_strategy!=Parameters<dim>::NonlinearSolverFailureStrategy::continue_with_next_timestep &&
                p.surface_pressure>0. && p.pressure_normalization=="no",
                ExcMessage("Growing fault requires a single midpoint fault phase, plane strain, native elasticity off, single Advection, iterated Newton Stokes, prescribed positive surface pressure and no pressure normalization. Nonconvergence must abort or request a retry."));
    AssertThrow(!p.mesh_deformation_enabled && !p.initial_adaptive_refinement && !p.adaptive_refinement_interval &&
                p.additional_refinement_times.empty() && p.stokes_velocity_degree==2 && !p.use_equal_order_interpolation_for_stokes &&
                !p.use_locally_conservative_discretization && !p.include_melt_transport,
                ExcMessage("Growing fault requires a fixed conforming Q2/Q1 mesh."));
    AssertThrow(!p.use_direct_stokes_solver,ExcMessage("Midpoint fault mechanics requires separate velocity and pressure blocks."));
    AssertThrow(!p.run_postprocessors_on_nonlinear_iterations,ExcMessage("Midpoint fault output requires an accepted step; iteration postprocessing is unsupported."));
    AssertThrow(!p.enable_prescribed_dilation &&
                (p.formulation_mass_conservation==Parameters<dim>::Formulation::MassConservation::incompressible ||
                 p.formulation_mass_conservation==Parameters<dim>::Formulation::MassConservation::ask_material_model),
                ExcMessage("Midpoint fault mechanics requires incompressible continuity without prescribed dilation."));
    prm.enter_subsection("Material model");
    prm.enter_subsection("Growing fault");
    degrees=prm.get_double("Seed tangent angle");
    const auto coordinates=Utilities::string_to_double(Utilities::split_string_list(prm.get("Forward seed tip")));
    anchor=Point<2>(coordinates[0],coordinates[1]);
    order=prm.get_integer("Section quadrature order");
    initial_age=prm.get_double("Initial age");
    const double G=prm.get_double("Shear modulus"), strength=prm.get_double("Cohesion");
    const double Kf=energy.fracture_energy/(energy.geometry.normalization()*energy.length_scale);
    AssertThrow(strength>0.,ExcMessage("Growing fault requires positive cohesion."));
    point_law= {G,prm.get_double("Viscosity"),Kf*2.*G/(strength*strength)};
    law= {prm.get_double("Reference friction coefficient"),prm.get_double("Direct effect"),prm.get_double("Evolution effect"),
          prm.get_double("Reference speed"),prm.get_double("Slip distance"),prm.get_double("Radiation damping")
         };
    AssertThrow(G>0. && point_law.eta>0. && initial_age>0. && law.direct_effect>0. && law.reference_speed>0. && law.slip_distance>0.,
                ExcMessage("Growing fault requires positive Maxwell and RSF scales."));
    AssertThrow(law.damping>0. || law.evolution_effect==0.,
                ExcMessage("The growing fault monotonicity bound requires positive damping for nonzero state evolution."));
    prm.leave_subsection();
    prm.leave_subsection();
    this->model_dependence.viscosity=MaterialModel::NonlinearDependence::none;
    this->model_dependence.density=MaterialModel::NonlinearDependence::none;
    this->model_dependence.compressibility=MaterialModel::NonlinearDependence::none;
    this->model_dependence.specific_heat=MaterialModel::NonlinearDependence::none;
    this->model_dependence.thermal_conductivity=MaterialModel::NonlinearDependence::none;
  }

  template <int dim>
  void Material<dim>::initialize()
  {
    this->get_signals().post_resume_load_user_data.connect([this](parallel::distributed::Triangulation<dim> &)
    {
      initialize_native_history();
    });
    this->get_signals().post_accept_timestep.connect([this](LinearAlgebra::BlockVector &)
    {
      if (trial)
        {
          accepted=std::move(*trial);
          trial.reset();
          last_balance=trial_balance;
          balance_available=true;
        }
    });
  }

  template <int dim>
  void Material<dim>::evaluate(const MaterialModel::MaterialModelInputs<dim> &in,MaterialModel::MaterialModelOutputs<dim> &out) const
  {
    for (unsigned int q=0; q<in.n_evaluation_points(); ++q)
      {
        out.viscosities[q]=point_law.eta;
        out.densities[q]=1.;
        out.specific_heat[q]=1.;
        out.thermal_conductivities[q]=0.;
        out.thermal_expansion_coefficients[q]=out.compressibilities[q]=0.;
        out.entropy_derivative_pressure[q]=out.entropy_derivative_temperature[q]=0.;
        std::fill(out.reaction_terms[q].begin(),out.reaction_terms[q].end(),0.);
      }
  }

  template <int dim>
  void Material<dim>::initialize_native_history()
  {
    if constexpr (dim==2)
      {
        AssertThrow(dynamic_cast<const GeometryModel::Box<dim>*>(&this->get_geometry_model()) &&
                    this->get_geometry_model().get_periodic_boundary_pairs().empty(),
                    ExcMessage("Growing fault requires a fixed nonperiodic box."));
        geometry=std::make_unique<MaterialSections>(this->get_triangulation(),this->get_mpi_communicator(),degrees,order);
        accepted.sections.resize(geometry->edges.size()-1);
        for (auto &s:accepted.sections) s.age=initial_age;
        for (const auto &cell:geometry->cells) accepted.cells[cell.first].resize(cell.second.quadrature.size());
        if (!restored.empty())
          {
            std::istringstream input(restored);
            aspect::iarchive ar(input);
            unsigned int n;
            double angle;
            std::vector<double> edges;
            std::vector<double> signature;
            ar >> signature;
            AssertThrow(signature==checkpoint_parameters(),ExcMessage("Growing fault physical parameters differ from the checkpoint."));
            ar >> n >> angle >> edges >> accepted.step >> accepted.time >> accepted.sections;
            AssertThrow(n==order && angle==degrees && edges==geometry->edges && accepted.sections.size()+1==edges.size(),
                        ExcMessage("Incompatible native material strip checkpoint."));
            std::map<std::string,std::pair<std::array<Point<2>,4>,std::vector<PointHistory>>> points;
            ar >> points;
            for (const auto &cell:this->get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  auto &history=accepted.cells.at(cell->id());
                  const auto &saved_cell=points.at(cell->id().to_string());
                  for (const auto v:cell->vertex_indices())
                    AssertThrow(saved_cell.first[v]==cell->vertex(v),ExcMessage("Material point locations differ from the fixed checkpoint mesh."));
                  const auto &saved=saved_cell.second;
                  AssertThrow(saved.size()==history.size(),ExcMessage("Incompatible native quadrature checkpoint."));
                  history=saved;
                }
            restored.clear();
          }
      }
  }

  template <int dim>
  SolverControl Material<dim>::solve_midpoint_fields(LinearAlgebra::BlockVector &solution)
  {
    SolverControl control(this->get_parameters().max_nonlinear_iterations,this->get_parameters().nonlinear_tolerance);
    if constexpr(dim==2)
      {
        if (!geometry) initialize_native_history();
        if (this->get_timestep_number()==0)
          {
            // Initialize the physical Q2 field with the inherited whole-cell
            // scaling. An admissible polynomial keeps every input coefficient.
            const auto &intro=this->introspection();
            const auto c=intro.component_indices.compositional_fields[field], b=intro.block_indices.compositional_fields[field];
            const auto offset=solution.get_block_indices().block_start(b);
            LinearAlgebra::Vector owned(intro.index_sets.system_partitioning[b],this->get_mpi_communicator());
            std::vector<types::global_dof_index> ids(this->get_fe().n_dofs_per_cell());
            unsigned int invalid=0;
            for (const auto &cell:this->get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned())
                {
                  cell->get_dof_indices(ids);
                  std::vector<double> patch(9);
                  for (unsigned int j=0; j<9; ++j) patch[j]=solution[ids[this->get_fe().component_to_system_index(c,j)]];
                  invalid|=!PhaseField::limit_bernstein<2>(patch,std::accumulate(patch.begin(),patch.end(),0.)/9.);
                  for (unsigned int j=0; j<9; ++j) owned[ids[this->get_fe().component_to_system_index(c,j)]-offset]=patch[j];
                }
            AssertThrow(!Utilities::MPI::max(invalid,this->get_mpi_communicator()),ExcMessage("Invalid initial phase mean."));
            owned.compress(VectorOperation::insert);
            solution.block(b)=owned;
            control.check(0,0.);
            return control;
          }
        AssertThrow(accepted.step+1==this->get_timestep_number(),ExcMessage("Growing fault accepted parent clock mismatch."));
        trial.reset();
        auto next=std::make_unique<History>();
        Balance result;
        solve_midpoint(*this,solution,*next,result);
        trial=std::move(next);
        trial_balance=result;
        control.check(result.iterations,result.mechanical);
      }
    return control;
  }

  template <int dim>
  std::vector<double> Material<dim>::checkpoint_parameters() const
  {
    const auto &energy=this->get_parameters().surface_energy_model;
    return {energy.fracture_energy,energy.length_scale,this->get_parameters().discontinuous_penalty,
            this->get_parameters().surface_pressure,anchor[0],anchor[1],point_law.G,point_law.eta,point_law.m,
            law.reference_coefficient,law.direct_effect,law.evolution_effect,law.reference_speed,law.slip_distance,law.damping
           };
  }

  template <int dim>
  void Material<dim>::save(std::map<std::string,std::string> &status) const
  {
    if (!geometry) return;
    using CellRecord=std::pair<std::array<Point<2>,4>,std::vector<PointHistory>>;
    std::map<std::string,CellRecord> local;
    if constexpr(dim==2)
      for (const auto &cell:this->get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            auto &record=local[cell->id().to_string()];
            for (const auto v:cell->vertex_indices()) record.first[v]=cell->vertex(v);
            record.second=accepted.cells.at(cell->id());
          }
    auto parts=Utilities::MPI::gather(this->get_mpi_communicator(),local);
    if (Utilities::MPI::this_mpi_process(this->get_mpi_communicator())==0)
      {
        std::map<std::string,CellRecord> all;
        for (auto &part:parts) all.merge(part);
        std::ostringstream out;
        {
          aspect::oarchive ar(out);
          ar << checkpoint_parameters() << order << degrees << geometry->edges << accepted.step << accepted.time << accepted.sections << all;
        }
        status["Native growing fault v1"]=out.str();
        if (balance_available)
          {
            std::ostringstream bytes;
            {
              aspect::oarchive archive(bytes);
              archive << last_balance;
            }
            status["Growing fault accepted balance v1"]=bytes.str();
          }
      }
  }

  template <int dim>
  void Material<dim>::load(const std::map<std::string,std::string> &status)
  {
    restored=status.at("Native growing fault v1");
    const auto entry=status.find("Growing fault accepted balance v1");
    balance_available=entry!=status.end();
    if (balance_available)
      {
        std::istringstream bytes(entry->second);
        aspect::iarchive archive(bytes);
        archive >> last_balance;
      }
  }

  ASPECT_REGISTER_MATERIAL_MODEL(Material,"growing fault",
                                 "Fixed material strips with coupled plane-strain Maxwell, section RSF and irreversible midpoint AT1 evolution.")
}
