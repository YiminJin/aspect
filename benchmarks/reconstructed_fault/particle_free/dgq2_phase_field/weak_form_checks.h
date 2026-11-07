/* Compare complete distributed operators with matched quadrature. This velocity
 * is representable in Q2 on affine and bilinear maps and has nonzero divergence.
 * Random/discontinuous phase traces exercise all four interface blocks. */
void check_material_weak_forms(const dealii::Mapping<2> &mapping,
                               const dealii::DoFHandler<2> &dofs,
                               const dealii::IndexSet &owned,
                               const dealii::IndexSet &relevant)
{
  using namespace aspect;
  using Form = Assemblers::ScalarAdvectionForm;
  PhaseField::DGOperator<2> conservative(mapping,dofs,0,0,owned,relevant,MPI_COMM_WORLD);
  PhaseField::DGOperator<2> advective(mapping,dofs,0,0,owned,relevant,MPI_COMM_WORLD);
  const auto velocity = [](const auto &,const auto &points,auto &v,auto &div)
  {
    for (unsigned int q = 0; q < points.size(); ++q)
      {
        const double x = points[q][0], y = points[q][1];
        v[q][0] = .7 + .13*x + .08*x*y;
        v[q][1] = -.21 + .07*y + .03*x*x;
        div[q] = .2 + .08*y;
      }
  };
  conservative.assemble_transport(velocity,Form::conservative_with_divergence);
  advective.assemble_transport(velocity,Form::advective);
  const auto &a = conservative.advection_matrix(), &b = advective.advection_matrix();
  double error = 0., scale = 0.;
  for (const auto row : owned)
    for (auto entry = a.begin(row); entry != a.end(row); ++entry)
      {
        error = std::max(error,std::abs(entry->value()-b.el(row,entry->column())));
        scale = std::max(scale,std::abs(entry->value()));
      }
  error = dealii::Utilities::MPI::max(error,MPI_COMM_WORLD);
  scale = dealii::Utilities::MPI::max(scale,MPI_COMM_WORLD);
  AssertThrow(error < 2e-12 * std::max(1.,scale),ExcMessage("Material weak forms disagree with matching quadrature."));
  aspect::LinearAlgebra::Vector state(owned,MPI_COMM_WORLD),left(state),right(state);
  for (const auto i : owned) state[i] = .35 + .1*std::sin(1.37*i);
  state.compress(VectorOperation::insert);
  const auto inflow = [](const Point<2> &x,double t) { return .3 + .01*x[1] + .02*t; };
  const auto l = conservative.transport(state,left,inflow,.7,.005,0.,1.,false);
  const auto r = advective.transport(state,right,inflow,.7,.005,0.,1.,false);
  left -= right;
  const double action_error = left.linfty_norm();
  AssertThrow(action_error < 2e-12,ExcMessage("Stage-time material advection disagrees between weak forms."));
  AssertThrow(std::abs(l.balance_error) < 2e-12 && std::abs(r.balance_error) < 2e-12,
              ExcMessage("Material balance lost divergence or boundary contribution."));
  if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::setprecision(17) << "{\"weak_form_matrix_error\":" << error
              << ",\"matrix_scale\":" << scale << ",\"stage_error\":" << action_error
              << ",\"conservative_balance\":" << l.balance_error
              << ",\"advective_balance\":" << r.balance_error << "}" << std::endl;
}
