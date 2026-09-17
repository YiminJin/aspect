// Archived constitutive probe; never included by the production material model.
// Opt-in fixed-coordinate history expression for the bounded BP3 replay.
// No nodal projection: retain Theta0 and the accepted Q1 rate functions.
#include <fstream>
#include <map>
#include <stdexcept>
#include <vector>

namespace aspect
{
  namespace MaterialModel
  {
    namespace internal
    {
      class FaultThetaHistoryDiagnostic
      {
        public:
          template <typename Fault>
          void load(const char *path, const unsigned int step, const Fault &fault)
          {
            if (loaded_step==step) return;
            std::ifstream in(path);
            unsigned int n=0;
            if (!(in>>n) || n!=fault.n_vertices())
              throw std::runtime_error("Invalid functional Theta initial history.");
            initial.resize(n);
            for (unsigned int i=0;i<n;++i)
              {
                double x,y;
                if (!(in>>x>>y>>initial[i]) || x!=fault.vertex(i)[0] || y!=fault.vertex(i)[1] || !(initial[i]>0))
                  throw std::runtime_error("Functional Theta geometry/initial state mismatch.");
              }
            updates.clear();
            unsigned int k;
            while (in>>k)
              {
                Update u;u.rate.resize(n);
                if (k!=updates.size()+1 || !(in>>u.dt) || !(u.dt>0))
                  throw std::runtime_error("Noncontiguous functional Theta update history.");
                for (double &v:u.rate)
                  if (!(in>>v) || !(v>0)) throw std::runtime_error("Invalid accepted functional Theta rate.");
                updates.push_back(std::move(u));
              }
            if (!in.eof() || updates.size()!=(step==0 ? 0 : step-1))
              throw std::runtime_error("Mechanics must consume the immediately preceding functional Theta history.");
            cache.clear();loaded_step=step;
          }

          template <typename Aging>
          double value(const unsigned int segment, const double xi, const Aging &aging)
          {
            const auto key=std::make_pair(segment,xi);
            const auto found=cache.find(key);
            if (found!=cache.end()) return found->second;
            double theta=(1-xi)*initial.at(segment)+xi*initial.at(segment+1);
            // This exact composition of accepted updates defines an independent
            // continuous-coordinate history, including newly sampled coordinates.
            for (const auto &u:updates)
              {
                const double left=u.rate[segment],right=u.rate[segment+1];
                const double v=xi==0 ? left : xi==1 ? right : left+xi*(right-left);
                theta=aging(v,theta,u.dt);
              }
            cache.emplace(key,theta);
            return theta;
          }

        private:
          struct Update { double dt; std::vector<double> rate; };
          unsigned int loaded_step=static_cast<unsigned int>(-1);
          std::vector<double> initial;
          std::vector<Update> updates;
          std::map<std::pair<unsigned int,double>,double> cache;
      };
    }
  }
}
