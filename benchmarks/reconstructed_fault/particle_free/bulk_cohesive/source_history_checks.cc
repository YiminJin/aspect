/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
#include "orientation.cc"

namespace aspect { namespace GeometryModel
{
  template<int dim> class BulkSourceDistorted:public Box<dim>
  {
    public:
      void create_coarse_mesh(parallel::distributed::Triangulation<dim> &grid)const override
      {
        AssertThrow(dim==2,ExcMessage("The source-history fixture is serial two dimensional."));
        Box<dim>::create_coarse_mesh(grid);
        if(std::getenv("ASPECT_BULK_SOURCE_AFFINE"))return;
        GridTools::transform([](const Point<dim> &p)
          {
            if(p[0]==0. || p[0]==1.5 || p[1]==0. || p[1]==1.)return p;
            Point<dim> result=p;
            const double s=std::sin(2*numbers::PI*p[0]/1.5)*std::sin(2*numbers::PI*p[1]);
            result[0]+=.006*s;result[1]+=.004*s;return result;
          },grid);
      }
  };
  ASPECT_REGISTER_GEOMETRY_MODEL(BulkSourceDistorted,"bulk source distortion",
    "Bounded fixed bilinear mesh for native FE/DG source-history qualification.")
}}
