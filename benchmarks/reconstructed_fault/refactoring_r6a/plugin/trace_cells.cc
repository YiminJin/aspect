// Fixture-only trace input; no mechanics, material evaluation or new production API.
#include <aspect/postprocess/interface.h>
#include <aspect/simulator_access.h>
#include <aspect/simulator_signals.h>
#include <fstream>

namespace aspect
{
  namespace Postprocess
  {
    template <int dim>
    class HistoryTraceCells : public Interface<dim>, public SimulatorAccess<dim>
    {
      public:
        static void declare_parameters(ParameterHandler &prm)
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("History trace cells");
          prm.declare_entry("Write selection", "true", Patterns::Bool());
          prm.leave_subsection();
          prm.leave_subsection();
        }
        void parse_parameters(ParameterHandler &prm) override
        {
          prm.enter_subsection("Postprocess");
          prm.enter_subsection("History trace cells");
          write_selection = prm.get_bool("Write selection");
          prm.leave_subsection();
          prm.leave_subsection();
        }
        void initialize() override
        {
          this->get_signals().pre_set_initial_state.connect([this](auto &tria)
          {
            if (!write_selection) return;
            std::ofstream out(this->get_output_directory()+"stress_trace_cells_rank"
              +std::to_string(Utilities::MPI::this_mpi_process(this->get_mpi_communicator()))+".txt");
            for (const auto &cell : tria.active_cell_iterators())
              if (cell->is_locally_owned()) out << cell->id() << '\n';
          });
        }
        std::pair<std::string,std::string> execute(TableHandler &) override { return {}; }
      private:
        bool write_selection = true;
    };
    ASPECT_REGISTER_POSTPROCESSOR(HistoryTraceCells, "history trace cells",
                                 "Write rank-local selected-cell input for the history trace regression.")
  }
}
