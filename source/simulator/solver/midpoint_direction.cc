/* This file is part of ASPECT, distributed under the GNU GPL v2 or later. */
#include <aspect/simulator/solver/midpoint_system.h>
#include <aspect/reconstructed_fault/onset_score.h>

namespace aspect::GrowingFault
{
  namespace
  {
    // Verify the tip against the accepted DG field before installing any
    // formed plane. A taper may vanish at the tip: inspect the incoming
    // centreline over half a strip width, capped at one phase length. The
    // strip width is at most the smallest Cartesian cell altitude.
    void check_initial_seed_tip(const SimulatorAccess<2> &sim, const Point<2> &tip,
                                const Tensor<1,2> &tangent, const double reach,
                                const unsigned int field)
    {
      const auto &fe = sim.get_fe();
      const auto phase = sim.introspection().extractors.compositional_fields[field];
      const QGauss<1> line(3); // Q2 restricted to an oblique line is at most quartic.
      bool tip_in_mesh = false;
      double phase_in_band = 0.;
      for (const auto &cell : sim.get_dof_handler().active_cell_iterators())
        if (cell->is_locally_owned())
          {
            // Work in unit coordinates: a face tolerance is measured in
            // roundoff multiples of the corresponding physical cell width.
            const auto &lo = cell->vertex(0);
            const auto &hi = cell->vertex(3);
            constexpr double face_tolerance = 64.*std::numeric_limits<double>::epsilon();
            double first = 0., last = reach;
            Point<2> unit_tip;
            Tensor<1,2> direction;
            for (unsigned int d=0; d<2; ++d)
              {
                unit_tip[d] = (tip[d]-lo[d])/(hi[d]-lo[d]);
                direction[d] = -tangent[d]/(hi[d]-lo[d]);
                if (direction[d] == 0.)
                  {
                    if (unit_tip[d] < -face_tolerance || unit_tip[d] > 1.+face_tolerance)
                      last = 0.;
                  }
                else
                  {
                    const double a = -unit_tip[d]/direction[d];
                    const double b = (1.-unit_tip[d])/direction[d];
                    first = std::max(first,std::min(a,b));
                    last = std::min(last,std::max(a,b));
                  }
              }
            tip_in_mesh |= GeometryInfo<2>::is_inside_unit_cell(unit_tip,face_tolerance);
            if (last <= first)
              continue;
            std::vector<Point<2>> points;
            for (const auto &q : line.get_points())
              points.push_back(GeometryInfo<2>::project_to_unit_cell(unit_tip+(first+(last-first)*q[0])*direction));
            FEValues<2> values(sim.get_mapping(),fe,Quadrature<2>(points),update_values);
            values.reinit(cell);
            std::vector<double> trace(points.size());
            values[phase].get_function_values(sim.get_old_solution(),trace);
            // A DG face can have two traces. Positive support on either
            // incident cell suffices; no averaging across MPI owners.
            double integral = 0.;
            for (unsigned int q=0; q<trace.size(); ++q)
              integral += line.weight(q)*trace[q];
            phase_in_band = std::max(phase_in_band,integral*((last-first)/reach));
          }
      const auto comm = sim.get_mpi_communicator();
      AssertThrow(Utilities::MPI::max(static_cast<unsigned int>(tip_in_mesh),comm),
                  ExcMessage("Forward seed tip lies outside the mesh."));
      AssertThrow(Utilities::MPI::max(phase_in_band,comm) > 64.*std::numeric_limits<double>::epsilon(),
                  ExcMessage("No local phase support immediately behind the forward seed tip."));
    }
  }

  void MidpointSystem::freeze_direction_neighborhood()
  {
    const auto &geometry=material.sections();
    const double ell=sim.get_parameters().surface_energy_model.length_scale;
    const auto &seed=material.seed_tip();
    const double tip_xi=geometry.coordinate(seed);
    std::vector<double> stations;
    for (unsigned int k=0; k<formed.size(); ++k) stations.push_back(.5*(geometry.edges[k]+geometry.edges[k+1]));
    const auto station=[&](const double xi)->unsigned int
    {return std::lower_bound(stations.begin(),stations.end(),xi)-stations.begin();};
    const unsigned int anchor=station(tip_xi-ell);
    if (material.history().step==0)
      {
        check_initial_seed_tip(sim, seed, geometry.tangent, std::min(ell,0.5*geometry.width(0)), field);
        std::vector<unsigned int> positive(formed.size());
        for (const auto &[id,cell]:cells)
          for (const auto &p:cell.points) if (p.old_phase>0.) positive[p.strip]=1;
        Utilities::MPI::max(positive,comm,positive);
        AssertThrow(anchor<stations.size() && stations[anchor]<=tip_xi && positive[anchor],ExcMessage("The forward tip needs contiguous positive seed support one phase length behind the anchor."));
        unsigned int first=anchor,last=anchor+1;
        while (first>0 && positive[first-1]) --first;
        while (last<stations.size() && stations[last]<=tip_xi && positive[last]) ++last;
        std::fill(formed.begin()+first,formed.begin()+last,1u);
      }

    Point<2> center=seed;
    const unsigned int beyond=std::upper_bound(stations.begin(),stations.end(),tip_xi)-stations.begin();
    if (material.history().step && beyond<formed.size() && formed[beyond])
      {
        const auto &t=geometry.tangent;
        const Tensor<1,2> normal({-t[1],t[0]});
        double z=normal*(seed-geometry.origin),last_xi=tip_xi;
        const auto &fe=sim.get_fe();
        const auto c=sim.introspection().component_indices.compositional_fields[field];
        const auto &element=fe.base_element(fe.component_to_base_index(c).first);
        std::vector<Index> ids(fe.n_dofs_per_cell());
        for (unsigned int k=anchor; k<formed.size() && (k<beyond || formed[k]); ++k)
          {
            if (stations[k]>tip_xi)
              {
                z+=(stations[k]-last_xi)*std::tan(angles[k]);
                last_xi=stations[k];
              }
            const Point<2> p=geometry.origin+(stations[k]*t+z*normal);
            double trace=0.;
            for (const auto &cell:sim.get_dof_handler().active_cell_iterators())
              if (cell->is_locally_owned() && cell->bounding_box().point_inside(p,1.e-12))
                {
                  const auto unit=sim.get_mapping().transform_real_to_unit_cell(cell,p);
                  if (!GeometryInfo<2>::is_inside_unit_cell(unit,1.e-12)) continue;
                  cell->get_dof_indices(ids);
                  double value=0.;
                  for (unsigned int i=0; i<element.n_dofs_per_cell(); ++i)
                    value+=element.shape_value(i,unit)*sim.get_old_solution()[ids[fe.component_to_system_index(c,i)]];
                  trace=std::max(trace,value);
                }
            if (Utilities::MPI::max(trace,comm)<.05) break;
            if (stations[k]>tip_xi) center=p;
          }
      }
    // Freeze both eligibility and the compact kernel from this accepted graph.
    const double radius=2.*ell;
    for (auto &[id,cell]:cells)
      for (auto &p:cell.points)
        {
          const double factor=std::max(0.,1.-p.position.distance_square(center)/(radius*radius));
          p.kernel=factor*factor;
          if (factor>0. && !formed[p.strip]) eligible[p.strip]=1;
        }
    Utilities::MPI::max(eligible,comm,eligible);
  }

  bool MidpointSystem::select_prospective_planes(const Values &x)
  {
    const auto &geometry=material.sections();
    if (std::none_of(eligible.begin(),eligible.end(),[](unsigned int a)
    {
      return a!=0;
    })) return false;
    // First three moments use the entire strip. Only the last six apply the
    // compact physical-area kernel to the squared candidate traction.
    Values moments(9*angles.size()), centered(3*angles.size());
    struct WeightedStress
    {
      unsigned int k;
      double weight,u,v;
    };
    std::vector<WeightedStress> samples;
    const auto &t=geometry.tangent;
    const double cos2=t[0]*t[0]-t[1]*t[1],sin2=2.*t[0]*t[1];
    for (const auto &[id,cell]:cells)
      for (unsigned int q=0; q<cell.points.size(); ++q)
        {
          const auto &p=cell.points[q];
          const auto k=p.strip;
          if (!eligible[k]) continue;
          const auto &old=material.history().cells.at(id)[q];
          AssertThrow(old.cohesive_strain.norm()==0. && old.absolute_shear==0.,ExcMessage("Turning a plane carrying recoverable strain is unsupported."));
          Stress increment;
          double phi=0.;
          for (unsigned int j=0; j<cell.indices.size(); ++j)
            {
              increment+=x[cell.indices[j]]*p.shapes[j].strain;
              phi+=x[cell.indices[j]]*p.shapes[j].phase;
            }
          const Stress predictor=B*(old.stress+2.*G*increment);
          const double u0=.5*(predictor[1][1]-predictor[0][0]),v0=predictor[0][1];
          const double u=cos2*u0-sin2*v0,v=sin2*u0+cos2*v0;
          phi=physical_phase(phi);
          const double c=material.maxwell().m*phi*(1.+phi),d=(1.-phi)*(1.-phi),F=c/(c+d);
          const double w=p.area*(1.-F)*(1.-F)*p.kernel;
          const auto local=onset_moments(p.area,p.kernel,F,u,v);
          for (unsigned int j=0; j<9; ++j) moments[9*k+j]+=local[j];
          if (w>0.) samples.push_back({k,w,u,v});
        }
    Utilities::MPI::sum(moments,comm,moments);
    // Center before squaring when the creep resistance nearly cancels the
    // predictor. The second pass uses the same global weighted means on every rank.
    for (const auto &p:samples)
      {
        const auto *m=&moments[9*p.k];
        const double du=p.u-m[6]/m[8],dv=p.v-m[7]/m[8];
        centered[3*p.k]+=p.weight*du*du;
        centered[3*p.k+1]+=p.weight*du*dv;
        centered[3*p.k+2]+=p.weight*dv*dv;
      }
    Utilities::MPI::sum(centered,comm,centered);
    bool changed=false;
    for (unsigned int k=0; k<angles.size(); ++k)
      if (eligible[k])
        {
          const auto *m=&moments[9*k];
          if (m[8]==0. || (m[3]+m[4]==0. && m[0]==0. && m[1]==0.)) continue;
          const auto resistance=[&](double free,double mobility)
          {
            return material.friction().solve_strip(k,free,mobility,h,sim.get_parameters().surface_pressure).resistance;
          };
          const auto score=[&](const double angle)
          {
            return candidate_onset(m,&centered[3*k],resistance,G*B*h*geometry.length(k,angle),angle).score;
          };
          const double limit=numbers::PI/4.,spacing=2.*limit/128.;
          unsigned int best=0;
          double maximum=-1.;
          for (unsigned int j=0; j<=128; ++j)
            {
              const double value=score(-limit+j*spacing);
              if (value>maximum)
                {
                  maximum=value;
                  best=j;
                }
            }
          AssertThrow(best>0 && best<128,ExcMessage("The preferred plane reaches the boundary of the forward sector."));
          double lo=-limit+(best-1)*spacing,hi=-limit+(best+1)*spacing;
          const double ratio=(std::sqrt(5.)-1.)/2.;
          double left=hi-ratio*(hi-lo),right=lo+ratio*(hi-lo),fl=score(left),fr=score(right);
          while (hi-lo>1.e-9)
            {
              if (fl<fr)
                {
                  lo=left;
                  left=right;
                  fl=fr;
                  right=lo+ratio*(hi-lo);
                  fr=score(right);
                }
              else
                {
                  hi=right;
                  right=left;
                  fr=fl;
                  left=hi-ratio*(hi-lo);
                  fl=score(left);
                }
            }
          const double selected=.5*(lo+hi);
          if (score(selected)-score(angles[k])>128.*std::numeric_limits<double>::epsilon()*(m[3]+m[4])
              && std::abs(selected-angles[k])>1.e-7)
            {
              angles[k]=selected;
              changed=true;
            }
        }
    return changed;
  }
}
