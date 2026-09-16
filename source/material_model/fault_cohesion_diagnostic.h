// Bounded frozen-profile resistance diagnostic, not a cohesive history law.
#include <array>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace aspect
{
  namespace MaterialModel
  {
    namespace internal
    {
      class FaultCohesionDiagnostic
      {
        public:
          template <typename Fault>
          void load(const char *path,const Fault &fault)
          {
            if (!data.empty()) return;
            std::ifstream in(path);unsigned int n;
            if (!(in>>n) || n!=fault.n_vertices()) throw std::runtime_error("Invalid frozen cohesive snapshot.");
            data.resize(n);
            for (unsigned int i=0;i<n;++i)
              {
                double x,y;
                if (!(in>>x>>y>>data[i][0]>>data[i][1]>>data[i][2])
                    || x!=fault.vertex(i)[0] || y!=fault.vertex(i)[1]
                    || !(data[i][0]>=0) || !(data[i][1]>0) || !(data[i][2]>0))
                  throw std::runtime_error("Frozen cohesive geometry/state mismatch.");
              }
          }

          double value(const unsigned int segment,const double xi,const double beta0,const double kappa0) const
          {
            const auto a=data.at(segment),b=data.at(segment+1);
            const double C=(1-xi)*a[0]+xi*b[0],I=(1-xi)*a[2]+xi*b[2];
            const double V=xi==0 ? a[1] : xi==1 ? b[1] : a[1]+xi*(b[1]-a[1]);
            return (kappa0*V+beta0*I*C)/I;
          }

        private:
          std::vector<std::array<double,3>> data;
      };
    }
  }
}
