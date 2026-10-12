// Output-only ASPECT postprocessor. No simulator solution/history mutation.
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <fstream>
#include <array>
#include <iomanip>

namespace aspect { namespace Postprocess {
 template<int dim> class AcceptedSnapshot : public Interface<dim>, public SimulatorAccess<dim> {
  std::string input;
  class Field : public Function<dim> {
   public:
    std::vector<std::array<double,8>> rows;
    unsigned int component=1;
    double value(const Point<dim> &p, unsigned int=0) const override {
     auto upper=std::upper_bound(rows.begin(),rows.end(),p[1],[](double y,const auto &row){return y<row[0];});
     if(upper==rows.begin())return rows.front()[component];
     if(upper==rows.end())return rows.back()[component];
     const auto &a=*(upper-1),&b=*upper; const double s=(p[1]-a[0])/(b[0]-a[0]);
     return (1-s)*a[component]+s*b[component];
    }
  };
 public:
  static void declare_parameters(ParameterHandler &prm) {
   prm.enter_subsection("Postprocess");prm.enter_subsection("P2 accepted snapshot visualization");
   prm.declare_entry("Input directory","",Patterns::DirectoryName(),"Accepted prototype snapshots; rendering only, no history import.");
   prm.leave_subsection();prm.leave_subsection();
  }
  void parse_parameters(ParameterHandler &prm) override {
   prm.enter_subsection("Postprocess");prm.enter_subsection("P2 accepted snapshot visualization");input=prm.get("Input directory");
   prm.leave_subsection();prm.leave_subsection();
  }
  std::pair<std::string,std::string> execute(TableHandler &) override {
   AssertThrow(dim==2 && Utilities::MPI::n_mpi_processes(this->get_mpi_communicator())==1,
               ExcMessage("Snapshot visualization is a serial 2D output-only diagnostic."));
   AssertThrow(this->get_timestep_number()==0,ExcMessage("Render snapshots once at native initialization, without time advancement."));
   DoFHandler<dim> dofs(this->get_triangulation());FE_Q<dim> fe(1);dofs.distribute_dofs(fe);
   const std::array<std::string,7> names={{"phase","localization","shear_stress","cohesive_traction","cumulative_slip","velocity_x","velocity_y"}};
   const std::array<std::string,7> units={{"1","1/m","Pa","Pa","m","m/s","m/s"}};
   std::vector<std::pair<double,std::string>> records;
   for(unsigned int frame=0;frame<5;++frame) {
    Field field;std::ifstream in(input+"/snapshot-"+std::to_string(frame)+".txt");
    AssertThrow(bool(in),ExcMessage("Missing accepted snapshot input."));double time;in>>time;
    std::array<double,8> row;while(in>>row[0]>>row[1]>>row[2]>>row[3]>>row[4]>>row[5]>>row[6]>>row[7])field.rows.push_back(row);
    AssertThrow(field.rows.size()==513,ExcMessage("Expected 512 normal cells, with both endpoints."));
    for(unsigned int i=0;i<field.rows.size();++i) {
     AssertThrow(std::abs(field.rows[i][0]-double(i)/512)<1e-13,ExcMessage("Snapshot and native normal mesh mismatch."));
     AssertThrow(field.rows[i][1]>=0 && field.rows[i][1]<1 && field.rows[i][2]>=0,ExcMessage("Inadmissible accepted visualization field."));
    }
    DataOut<dim> data;data.attach_dof_handler(dofs);std::array<Vector<double>,7> values;
    for(unsigned int c=0;c<names.size();++c) {
     values[c].reinit(dofs.n_dofs());field.component=c+1;
     VectorTools::interpolate(this->get_mapping(),dofs,field,values[c]);data.add_data_vector(values[c],names[c]);
    }
    DataOutBase::VtkFlags flags;flags.time=time;flags.cycle=frame;
    flags.compression_level=DataOutBase::CompressionLevel::best_compression;
    for(unsigned int c=0;c<names.size();++c)flags.physical_units[names[c]]=units[c];
    data.set_flags(flags);data.build_patches(this->get_mapping(),1);
    const std::string filename="accepted-"+std::to_string(frame)+".vtu";
    std::ofstream out(this->get_output_directory()+filename);data.write_vtu(out);records.emplace_back(time,filename);
   }
   std::ofstream pvd(this->get_output_directory()+"accepted.pvd");DataOutBase::write_pvd_record(pvd,records);
   return {"Accepted snapshot visualization","5 compressed VTU frames (output-only replay, physical state unchanged)"};
  }
 };
 ASPECT_REGISTER_POSTPROCESSOR(AcceptedSnapshot,"P2 accepted snapshot visualization",
   "Render accepted P2 prototype snapshots through ASPECT/deal.II DataOut on the native mesh. No physical history import or update.")
}}
