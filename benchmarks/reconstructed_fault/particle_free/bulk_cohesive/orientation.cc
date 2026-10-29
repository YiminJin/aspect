/* Copyright (C) 2026 by the authors of ASPECT. GNU GPL v2 or later. */
// Isolated mesh-orientation fixture. No production algorithms are replaced.
#define ASPECT_BULK_ORIENTATION_FIXTURE
#include "native.cc"
#include <aspect/geometry_model/box.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/mapping_q1.h>

namespace aspect
{
  namespace GeometryModel
  {
    template<int dim> class BulkRotatedPatch:public Box<dim>
    {
      public:
        void create_coarse_mesh(parallel::distributed::Triangulation<dim> &grid)const override
        {
          AssertThrow(dim==2,ExcMessage("The bounded orientation fixture is two dimensional."));
          const bool identity=std::getenv("ASPECT_BULK_ORIENTATION_IDENTITY");
          if(identity){Box<dim>::create_coarse_mesh(grid);return;}
          GridIn<dim> input;input.attach_triangulation(grid);
          std::ifstream file("benchmarks/reconstructed_fault/particle_free/bulk_cohesive/orientation.inp");
          AssertThrow(file,ExcMessage("Run the orientation fixture from the worktree root."));
          input.read_ucd(file);
          for(const auto &cell:grid.active_cell_iterators())for(const auto f:cell->face_indices())
            if(cell->face(f)->at_boundary())
              {
                const auto p=cell->face(f)->center();
                if(p[0]==0.)cell->face(f)->set_boundary_id(0);
                else if(p[0]==1.5)cell->face(f)->set_boundary_id(1);
                else if(p[1]==0.)cell->face(f)->set_boundary_id(2);
                else if(p[1]==1.)cell->face(f)->set_boundary_id(3);
                else AssertThrow(false,ExcMessage("The orientation mesh has an unintended boundary."));
              }
        }
    };
    ASPECT_REGISTER_GEOMETRY_MODEL(BulkRotatedPatch,"bulk rotated patch",
      "Benchmark-only rigid 45 degree rectangular core and conforming fixed-boundary annulus.")
  }

  namespace Postprocess
  {
    template<int dim> class BulkOrientationAudit:public Interface<dim>,public SimulatorAccess<dim>
    {
      public:
        std::pair<std::string,std::string> execute(TableHandler&)override
        {
          if(this->get_timestep_number()!=0)return {"Orientation audit","fixed mesh"};
          AssertThrow(dynamic_cast<const MappingQ1<dim>*>(&this->get_mapping()),
                      ExcMessage("Rotated fixture requires native MappingQ1, never MappingCartesian."));
          const FE_Q<dim> element(2);
          const QGauss<dim> q3(3),q8(8);
          FEValues<dim> a(this->get_mapping(),element,q3,update_values|update_gradients|update_JxW_values);
          FEValues<dim> b(this->get_mapping(),element,q8,update_values|update_gradients|update_JxW_values|update_quadrature_points);
          std::ofstream out(this->get_output_directory()+"mesh-qualification.csv");
          out<<std::setprecision(17)<<"cell,x0,y0,x1,y1,x2,y2,x3,y3,affine_defect,min_det,max_det,area,affine_physical_error,source3_error,source8_error,coefficient3_8_difference\n";
          double area=0.;
          for(const auto &cell:this->get_triangulation().active_cell_iterators())
            {
              a.reinit(cell);b.reinit(cell);
              double min_det=std::numeric_limits<double>::max(),max_det=0.,cell_area=0.,physical_error=0.;
              for(unsigned int q=0;q<q8.size();++q)
                {
                  const double det=b.JxW(q)/q8.weight(q);
                  min_det=std::min(min_det,det);max_det=std::max(max_det,det);cell_area+=b.JxW(q);
                  // An affine physical scalar must have its exact gradient even
                  // in a bilinear transition cell. This checks the actual map.
                  Tensor<1,dim> gradient;
                  for(unsigned int i=0;i<9;++i)
                    {
                      const auto p=this->get_mapping().transform_unit_to_real_cell(cell,element.get_unit_support_points()[i]);
                      gradient+=(p[0]+2*p[1])*b.shape_grad(i,q);
                    }
                  physical_error=std::max(physical_error,std::hypot(gradient[0]-1.,gradient[1]-2.));
                }
              AssertThrow(min_det>0. && std::isfinite(max_det),ExcMessage("Inverted rotated transition cell."));
              // Independently project every native scalar Q2 basis gradient
              // into all nine Q2 modes; compare source rules without modifying
              // the native source3 rule or its pointwise publication guard.
              FullMatrix<double> m3(9,9),m8(9,9);
              for(unsigned int i=0;i<9;++i)for(unsigned int j=0;j<9;++j)
                {
                  for(unsigned int q=0;q<q3.size();++q)m3(i,j)+=a.JxW(q)*a.shape_value(i,q)*a.shape_value(j,q);
                  for(unsigned int q=0;q<q8.size();++q)m8(i,j)+=b.JxW(q)*b.shape_value(i,q)*b.shape_value(j,q);
                }
              m3.gauss_jordan();m8.gauss_jordan();
              double error3=0.,error8=0.,difference=0.;
              for(unsigned int k=0;k<9;++k)for(unsigned int d=0;d<2;++d)
                {
                  Vector<double> rhs3(9),rhs8(9),c3(9),c8(9);
                  for(unsigned int i=0;i<9;++i)
                    {
                      for(unsigned int q=0;q<q3.size();++q)rhs3[i]+=a.JxW(q)*a.shape_value(i,q)*a.shape_grad(k,q)[d];
                      for(unsigned int q=0;q<q8.size();++q)rhs8[i]+=b.JxW(q)*b.shape_value(i,q)*b.shape_grad(k,q)[d];
                    }
                  m3.vmult(c3,rhs3);m8.vmult(c8,rhs8);
                  for(unsigned int i=0;i<9;++i)difference=std::max(difference,std::abs(c3[i]-c8[i]));
                  for(unsigned int q=0;q<q8.size();++q)
                    {
                      double v3=0.,v8=0.;
                      for(unsigned int i=0;i<9;++i){v3+=c3[i]*b.shape_value(i,q);v8+=c8[i]*b.shape_value(i,q);}
                      const double value=b.shape_grad(k,q)[d];
                      error3=std::max(error3,std::abs(v3-value));error8=std::max(error8,std::abs(v8-value));
                    }
                }
              const auto defect=cell->vertex(0)+cell->vertex(3)-cell->vertex(1)-cell->vertex(2);
              out<<cell->id();for(unsigned int v=0;v<4;++v)out<<','<<cell->vertex(v)[0]<<','<<cell->vertex(v)[1];
              out<<','<<defect.norm()<<','<<min_det<<','<<max_det<<','<<cell_area<<','<<physical_error<<','<<error3<<','<<error8<<','<<difference<<'\n';
              area+=cell_area;
            }
          AssertThrow(std::abs(area-1.5)<1e-12,ExcMessage("Rotated mesh changed the physical domain."));
          return {"Orientation audit","native map, area and source representation recorded"};
        }
    };
    ASPECT_REGISTER_POSTPROCESSOR(BulkOrientationAudit,"bulk orientation audit",
      "Benchmark-only mesh, affine reproduction and independent source3/source8 comparison.")
  }
}
