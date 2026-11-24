// Isolated 8 x 4 diagnostic. Files contain round-trippable binary64 values.
#include <fstream>
#include <iomanip>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static double strip_free, strip_mobility;

struct StripSample
{
  double area, kernel, fraction, u, v;
};

struct StripReplay
{
  std::string name;
  std::map<std::pair<std::string,unsigned int>,StripSample> supplied;
  std::vector<double> supplied_moments, angles;
  std::ofstream points, coefficients, moments, scores;
  const unsigned int rank;

  StripReplay(std::string mode, unsigned int rank) : name(mode), rank(rank)
  {
    std::ifstream requests("angles");
    for (double angle; requests >> angle;) angles.push_back(angle);
    if (angles.empty()) throw std::runtime_error("Missing angular probes");
    if (name=="native")
      {
        points.open("points-"+std::to_string(rank));
        coefficients.open("stage-"+std::to_string(rank));
        points << std::setprecision(17);
        coefficients << std::setprecision(17);
      }
    else if (name.find("point-")==0)
      {
        std::ifstream in("inputs/"+name);
        StripSample p;
        std::string cell;
        unsigned int q;
        while (in >> cell >> q >> p.area >> p.kernel >> p.fraction >> p.u >> p.v)
          if (!supplied.emplace(std::make_pair(cell,q),p).second)
            throw std::runtime_error("Duplicate supplied point");
        if (supplied.empty()) throw std::runtime_error("Empty point replay");
      }
    else
      {
        std::ifstream in("inputs/"+name);
        for (double v; in >> v;) supplied_moments.push_back(v);
        if (supplied_moments.size()!=12) throw std::runtime_error("Expected nine raw and three centered moments");
      }
    if (rank==0)
      {
        moments.open("moments-"+name);
        scores.open("scores-"+name);
        moments << std::setprecision(17);
        scores << std::setprecision(17);
      }
  }

  StripSample point(const std::string &cell, unsigned int q, StripSample p,
                    double phase, double xx, double yy, double xy)
  {
    if (name=="native")
      points << cell << ' ' << q << ' ' << p.area << ' ' << p.kernel << ' '
             << p.fraction << ' ' << p.u << ' ' << p.v << ' ' << phase << ' '
             << xx << ' ' << yy << ' ' << xy << '\n';
    return supplied.empty() ? p : supplied.at({cell,q});
  }

  void reduced(double *m,double *c)
  {
    for (unsigned int i=0;i<12;++i)
      {
        double &v=i<9 ? m[i] : c[i-9];
        if (!supplied_moments.empty()) v=supplied_moments[i];
        if (!rank) moments << (i ? " " : "") << v;
      }
    if (!rank) moments << '\n';
  }
};
