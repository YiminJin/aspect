/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/material_model/growing_fault.h>
#include <aspect/material_model/rheology/rate_and_state_friction.h>
#include <aspect/geometry_model/box.h>
#include <deal.II/fe/fe_values.h>
#include <numeric>

namespace aspect::GrowingFault
{
  namespace
  {
    // The repaired v1 archive interleaved RSF age with each physical strip.
    // Decode this representation only when restoring that format.
    struct ArchivedRsfStrip : SectionHistory
    {
      double age;
      template <class Archive> void serialize(Archive &ar,const unsigned int)
      {
        ar &age &slip &angle &rate &resistance &formed;
      }
    };
  }

  template <int dim>
  void Material<dim>::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Material model");
    prm.enter_subsection("Growing fault");
    prm.declare_entry("Prescribed fault compression", "", Patterns::List(Patterns::Double(0.),0,1),
                      "Positive compression in Pa used by the signed fault resistance law. An empty entry "
                      "uses the explicitly supplied Surface pressure from legacy growing fault inputs.");
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
    AssertThrow(&this->get_material_model()==this && p.compositional_field_methods[field]==Parameters<dim>::AdvectionFieldMethod::midpoint_fault_phase,
                ExcMessage("Growing fault requires a top-level midpoint material and a midpoint fault phase."));
    const bool legacy_compression = !prm.get("Surface pressure").empty();
    prm.enter_subsection("Material model");
    prm.enter_subsection("Growing fault");
    compression = p.surface_pressure;
    if (!prm.get("Prescribed fault compression").empty())
      {
        compression = prm.get_double("Prescribed fault compression");
        AssertThrow(!legacy_compression || compression == p.surface_pressure,
                    ExcMessage("Conflicting Prescribed fault compression and Surface pressure in growing fault input."));
      }
    AssertThrow(std::isfinite(compression) && compression > 0.,
                ExcMessage("Growing fault requires a finite, positive prescribed fault compression in Pa."));
    degrees=prm.get_double("Seed tangent angle");
    const auto coordinates=Utilities::string_to_double(Utilities::split_string_list(prm.get("Forward seed tip")));
    anchor=Point<2>(coordinates[0],coordinates[1]);
    order=prm.get_integer("Section quadrature order");
    const double G=prm.get_double("Shear modulus"), strength=prm.get_double("Cohesion");
    const double Kf=energy.fracture_energy/(energy.geometry.normalization()*energy.length_scale);
    AssertThrow(strength>0.,ExcMessage("Growing fault requires positive cohesion."));
    point_law= {G,prm.get_double("Viscosity"),Kf*2.*G/(strength*strength)};
    AssertThrow(G>0. && point_law.eta>0.,ExcMessage("Growing fault requires positive Maxwell scales."));
    law=make_resistance(prm);
    law->validate_midpoint();
    prm.leave_subsection();
    prm.leave_subsection();
    this->model_dependence.viscosity=MaterialModel::NonlinearDependence::none;
    this->model_dependence.density=MaterialModel::NonlinearDependence::none;
    this->model_dependence.compressibility=MaterialModel::NonlinearDependence::none;
    this->model_dependence.specific_heat=MaterialModel::NonlinearDependence::none;
    this->model_dependence.thermal_conductivity=MaterialModel::NonlinearDependence::none;
  }

  template <int dim>
  std::unique_ptr<MaterialModel::Rheology::FaultResistance> Material<dim>::make_resistance(ParameterHandler &prm) const
  {
    const MaterialModel::Rheology::RateAndStateFriction parameters= {prm.get_double("Reference friction coefficient"),prm.get_double("Direct effect"),prm.get_double("Evolution effect"),
                                                                     prm.get_double("Reference speed"),prm.get_double("Slip distance"),prm.get_double("Radiation damping")
                                                                    };
    return std::make_unique<MaterialModel::Rheology::RateStateHistory>(parameters,prm.get_double("Initial age"));
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
          for (unsigned int k=0; k<trial->sections.size(); ++k)
            law->accept_rate(k,trial->sections[k].rate,this->get_timestep());
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
        law->initialize(accepted.sections.size());
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
            ar >> n >> angle >> edges >> accepted.step >> accepted.time;
            if (legacy_checkpoint)
              {
                AssertThrow(law->name()=="rate and state",ExcMessage("The v1 growing fault checkpoint requires RSF."));
                std::vector<ArchivedRsfStrip> records;
                ar >> records;
                std::vector<double> ages;
                accepted.sections.assign(records.begin(),records.end());
                for (const auto &record:records) ages.push_back(record.age);
                law->restore_state(std::move(ages));
              }
            else
              {
                std::string name;
                std::vector<double> state;
                ar >> name >> accepted.sections >> state;
                AssertThrow(name==law->name(),ExcMessage("Growing fault checkpoint resistance law changed."));
                law->restore_state(std::move(state));
              }
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
  void Material<dim>::begin_attempt()
  {
    trial.reset();
    if (!geometry) initialize_native_history();
  }

  template <int dim>
  void Material<dim>::stage_candidate(History &&candidate,const Balance &ledger)
  {
    trial=std::make_unique<History>(std::move(candidate));
    trial_balance=ledger;
  }

  template <int dim>
  std::vector<double> Material<dim>::checkpoint_parameters() const
  {
    const auto &energy=this->get_parameters().surface_energy_model;
    std::vector<double> signature= {energy.fracture_energy,energy.length_scale,this->get_parameters().phase_discontinuous_penalty,
                                    compression,anchor[0],anchor[1],point_law.G,point_law.eta,point_law.m
                                   };
    const auto parameters=law->parameters();
    signature.insert(signature.end(),parameters.begin(),parameters.end());
    return signature;
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
          ar << checkpoint_parameters() << order << degrees << geometry->edges << accepted.step << accepted.time << law->name() << accepted.sections << law->state() << all;
        }
        status["Native growing fault v2"]=out.str();
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
    legacy_checkpoint=status.find("Native growing fault v2")==status.end();
    restored=status.at(legacy_checkpoint ? "Native growing fault v1" : "Native growing fault v2");
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
