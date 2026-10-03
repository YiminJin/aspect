#include "../../bp3/plugin/geometry.h"
#include "../../bp3/plugin/bp3_model.h"
#include "../../bp3/plugin/runtime.h"
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_signals.h>
#include <aspect/reconstructed_fault/manager.h>
#include <aspect/plugins.h>
#include <aspect/material_model/phase_field_fault.h>
#include <fstream>
#include <iomanip>

namespace aspect { namespace Postprocess {
  template <int dim>
  class BP3GeometryCheck : public Interface<dim>, public SimulatorAccess<dim>
  {
    public:
    void initialize() override
    {
      this->get_signals().post_simulator_initialization.connect([this](const auto &){
        using BP3::make_geometry;
        const auto &g=BP3::geometry();
        const auto make=[&](const PrescribedInitialFault<2> &f){return make_geometry({f},g.origin,g.extent,.1,.99,BP3::geometry().weakening_length,true);};
        const auto reject=[&](const auto &fn){bool rejected=false;try{fn();}catch(const std::exception &){rejected=true;}AssertThrow(rejected,ExcMessage("Unsupported BP3 geometry was accepted."));};
        auto reversed=g.prescribed;
        std::reverse(reversed.vertices.begin(),reversed.vertices.end());
        std::reverse(reversed.core_phase_field_values.begin(),reversed.core_phase_field_values.end());
        const auto rev=make(reversed);
        AssertThrow(g.upper==rev.upper && g.lower==rev.lower && g.tangent==rev.tangent && g.normal==rev.normal && g.shear_sense==rev.shear_sense,
                    ExcMessage("Vertex order changed physical orientation."));
        auto broken=g.prescribed;broken.vertices.insert(broken.vertices.begin()+1,broken.vertices[0]);broken.core_phase_field_values.insert(broken.core_phase_field_values.begin()+1,g.peak_phase);
        reject([&]{make(broken);});
        broken=g.prescribed;broken.vertices.insert(broken.vertices.begin()+1,(broken.vertices[0]+broken.vertices[1])/2.);broken.vertices[1][0]+=100.;broken.core_phase_field_values.insert(broken.core_phase_field_values.begin()+1,g.peak_phase);
        reject([&]{make(broken);});
        broken=g.prescribed;broken.core_phase_field_values[1]=.7;
        reject([&]{make(broken);});
        broken=g.prescribed;broken.core_phase_field_values.assign(broken.vertices.size(),1.);
        reject([&]{make(broken);});
        reject([&]{make_geometry({g.prescribed,g.prescribed},g.origin,g.extent,.1,.99,0.,true);});
        reject([&]{make_geometry({g.prescribed},g.origin,g.extent,.1,.99,2*g.length,false);});
        broken=g.prescribed;for(auto &p:broken.vertices)p[0]+=g.origin[0]-g.upper[0];
        reject([&]{make(broken);});
        broken=g.prescribed;broken.vertices[0][1]+=10.;
        reject([&]{make(broken);});
        // Loading has a down-dip signed jump -Vp*t from negative to positive
        // normal side. All calls use the maintained profile/loading implementation.
        const auto center=(g.upper+g.lower)/2.;
        Point<dim> a,b;
        for(unsigned d=0;d<2;++d){a[d]=center[d]-10*g.length*g.normal[d];b[d]=center[d]+10*g.length*g.normal[d];}
        const auto ua=BP3Restore::loading(*this,a),ub=BP3Restore::loading(*this,b);
        for(unsigned d=0;d<2;++d)AssertThrow(std::abs(ub[d]-ua[d]+BP3::Vp*g.tangent[d])<1e-24,ExcMessage("Thrust loading jump changed."));
        AssertThrow(std::abs(g.tangent*g.normal)<1e-15 && std::abs(g.tangent.norm()-1.)<1e-15 && std::abs(g.normal.norm()-1.)<1e-15,ExcMessage("Invalid BP3 frame."));
        AssertThrow(std::abs(g.down_dip(g.lower[0],g.lower[1])-g.length)<1e-10*g.length,ExcMessage("Incorrect down-dip origin."));
        AssertThrow(g.minimum_cell_distance(center,Point<2>(10.,10.))==0.,ExcMessage("Cell crossing support was missed."));
        if(this->get_pcout().is_active())
          {
            std::ofstream out(this->get_output_directory()+"geometry.csv");out<<std::setprecision(17);
            out<<"upper_x,upper_y,lower_x,lower_y,length,tx,ty,nx,ny,peak,shear_sense\n";
            out<<g.upper[0]<<','<<g.upper[1]<<','<<g.lower[0]<<','<<g.lower[1]<<','<<g.length<<','<<g.tangent[0]<<','<<g.tangent[1]<<','<<g.normal[0]<<','<<g.normal[1]<<','<<g.peak_phase<<','<<g.shear_sense<<'\n';
          }
        // Complete every rank's checks and root output before the intentional
        // MPI abort can terminate a slower peer.
        std::cout<<"BP3 GEOMETRY CHECK COMPLETE rank "
                 <<Utilities::MPI::this_mpi_process(this->get_mpi_communicator())<<std::endl;
        MPI_Barrier(this->get_mpi_communicator());
        AssertThrow(false,ExcMessage("BP3 GEOMETRY PASS: intentional stop before mechanics."));
      });
    }
    std::pair<std::string,std::string> execute(TableHandler &) override{return {};}
  };
  ASPECT_REGISTER_POSTPROCESSOR(BP3GeometryCheck,"BP3 geometry check","Isolated maintained geometry and loading checks; intentional early stop.")
}}
