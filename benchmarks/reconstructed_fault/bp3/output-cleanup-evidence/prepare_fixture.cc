// Reduced functional fixture only: 2 km cells, ell=4 km, 2 km fault spacing.
// Same restored geometry/law and virtual Cartesian-Q1 endpoint completion.
// It is not a resolution study or a replacement production BP3 model.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <functional>
#include <iostream>
using namespace dealii;
int main(int argc,char **argv)
{
  if(argc!=3) return 2;
  const std::string source=argv[1],out=std::string(argv[2])+"/";
  std::filesystem::create_directories(out);
  const double sn=std::sqrt(3.)/2,h=2000.;
  unsigned int n; double m;
  // Scale Gc with ell to retain the original convex degradation law.
  std::ifstream input(source+"/profile.txt"); input>>n>>m;
  std::vector<double> radius(n),phi(n),primitive(n);
  std::ofstream table(out+"profile.txt"); table<<std::setprecision(17)<<n<<' '<<m<<'\n';
  for(unsigned int i=0;i<n;++i)
    {input>>radius[i]>>phi[i]>>primitive[i];radius[i]*=200.;primitive[i]*=200.;
     table<<radius[i]<<' '<<phi[i]<<' '<<primitive[i]<<'\n';}
  AssertThrow(input,ExcMessage("Cannot read original profile."));
  Triangulation<2> tria;
  GridGenerator::subdivided_hyper_rectangle(tria,{75,25},Point<2>(-60000,0),Point<2>(90000,50000));
  std::ofstream cells(out+"target_cells.txt");
  for(const auto &cell:tria.active_cell_iterators()) cells<<cell->id().to_string()<<'\n';
  std::ifstream fault(source+"/fault.txt");
  std::vector<Point<2>> corners,vertices;double x,y,p;
  while(fault>>x>>y>>p) corners.emplace_back(x,y);
  std::filesystem::copy_file(source+"/fault.txt",out+"fault.txt");
  for(unsigned int j=0;j+1<corners.size();++j)
    {
      const auto a=corners[j],b=corners[j+1];const double length=a.distance(b);
      const unsigned int count=std::ceil(length/h);
      for(unsigned int i=j==0 ? 0 : 1;i<=count;++i)
        {const double s=i==count ? length : i*(length/count);vertices.push_back(a+(s/length)*(b-a));}
    }
  auto profile=[&](double r) {
    if(r>=radius.back()) return 0.;
    const auto i=std::upper_bound(radius.begin(),radius.end(),r)-radius.begin()-1;
    return phi[i]+(phi[i+1]-phi[i])*(r-radius[i])/(radius[i+1]-radius[i]);};
  auto q1=[&](Point<2> p) {
    const double i=std::floor((p[0]+60000)/h),j=std::floor(p[1]/h);
    const double a=(p[0]+60000)/h-i,b=p[1]/h-j;double result=0.;
    for(unsigned int u=0;u<2;++u) for(unsigned int v=0;v<2;++v)
      result+=(u ? a:1-a)*(v ? b:1-b)*profile(std::abs(sn*(-60000+(i+u)*h)+.5*((j+v)*h-50000)));
    return result;};
  Tensor<1,2> normal;normal[0]=-sn;normal[1]=-.5;
  const QGauss<1> gauss(8),surface(3);
  auto integrate=[&](Point<2> origin,double lo,double hi) {
    if(lo>=hi) return 0.;
    std::vector<double> cuts{lo,hi};
    for(unsigned int d=0;d<2;++d)
      {
        const double offset=d==0 ? -60000:0;
        const double a=origin[d]+normal[d]*lo,b=origin[d]+normal[d]*hi;
        for(int k=std::floor((std::min(a,b)-offset)/h);k<=std::ceil((std::max(a,b)-offset)/h);++k)
          {double t=(offset+k*h-origin[d])/normal[d];if(lo<t && t<hi) cuts.push_back(t);}
      }
    std::sort(cuts.begin(),cuts.end());
    auto panel=[&](double a,double b) {double sum=0.;for(unsigned int q=0;q<gauss.size();++q)
      {const double f=q1(origin+(a+(b-a)*gauss.point(q)[0])*normal);
       sum+=(b-a)*gauss.weight(q)*m*f*(1+f)/((1-f)*(1-f));}return sum;};
    std::function<double(double,double,unsigned int)> adaptive=[&](double a,double b,unsigned int depth) {
      const double low=panel(a,b),mid=(a+b)/2,high=panel(a,mid)+panel(mid,b);
      if(std::abs(high-low)<1e-12*std::max(1.,std::abs(high))) return high;
      AssertThrow(depth<24,ExcMessage("Completion did not converge."));
      return adaptive(a,mid,depth+1)+adaptive(mid,b,depth+1);};
    double result=0.;for(unsigned int i=1;i<cuts.size();++i) result+=adaptive(cuts[i-1],cuts[i],0);
    return result;};
  std::ofstream completion(out+"completion.txt");
  completion<<std::setprecision(17)<<(vertices.size()-1)*24<<'\n';
  unsigned int id=0;const double extent=radius.back()+h*(sn+.5);
  for(unsigned int j=1;j<vertices.size();++j)
    for(unsigned int panel=0;panel<8;++panel) for(unsigned int q=0;q<surface.size();++q)
      {
        const double xi=(panel+surface.point(q)[0])/8;
        const auto origin=(1-xi)*vertices[j-1]+xi*vertices[j];
        const double top=(50000-origin[1])/normal[1],bottom=-origin[1]/normal[1];
        const double value=integrate(origin,-extent,std::min(top,extent))
                          +integrate(origin,std::max(bottom,-extent),extent);
        completion<<id++<<' '<<origin[0]<<' '<<origin[1]<<' '<<value<<'\n';
      }
  std::cout<<tria.n_active_cells()<<" cells, "<<vertices.size()<<" fault vertices\n";
}
