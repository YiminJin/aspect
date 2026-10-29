// Independent extended-precision residual reference for diagnostic FD only.
// Neither the production point return nor a solved candidate is modified.
long double reference_resistance(const CohesiveInterface::PointState &old,
                                 const long double increment,
                                 const long double pressure,const long double dt) const
{
  const long double gc=law.Gc,peak=law.peak,c0=2*gc/(peak*peak),df=2*gc/peak;
  const long double e=(long double)old.jump+increment-old.reference;
  const long double a=std::max((long double)old.damage,std::min(std::abs(e)/df,1.L));
  const long double q=a==0 ? peak : (a==1 ? 0.L : (1-a)/(c0*a)*e);
  const long double velocity=increment/dt,speed=std::abs(velocity);
  const long double theta=((long double)old.theta+(long double)law.V0*dt/law.Dc)/(1+speed*dt/law.Dc);
  const long double z=speed/(2*(long double)law.V0)*std::exp(((long double)law.mu0+(long double)law.evolution*std::log(theta))/law.direct);
  const long double sign=velocity==0 ? 0.L : (velocity>0 ? 1.L : -1.L);
  return q+a*sign*(pressure*law.direct*std::asinh(z)+(long double)law.radiation*speed);
}

double derivative_reference(const std::vector<PointState> &old,
                            const std::vector<CohesiveInterface::PointState> &oldq,
                            const Trial &t,const double dt,const double rate_scale,
                            const double time) const
{
  AssertThrow(std::numeric_limits<long double>::digits>std::numeric_limits<double>::digits,
              ExcMessage("Derivative reference requires extended precision"));
  const double beta=1/(1+law.G/law.eta*dt);
  auto force=[&](const std::vector<long double> &inc)
  {
    std::vector<long double> f(ns),pn(ns);
    for(unsigned i=0;i<ns;++i)
      {f[i]=(long double)law.G/H*t.wall;pn[i]=law.p0;
       for(unsigned j=0;j<ns;++j)
         {const long double c=(long double)beta*((long double)old[j].relaxed_jump+inc[j]);
          f[i]-=(long double)K(i,j)*c;pn[i]+=(long double)N(i,j)*c;}}
    for(unsigned q=0;q<3*ns;++q)
      {long double dq=0,pq=0;for(unsigned i=0;i<ns;++i){dq+=(long double)Q(q,i)*inc[i];pq+=(long double)QP(q,i)*pn[i];}
       const auto r=reference_resistance(oldq[q],dq,pq,dt);
       for(unsigned i=0;i<ns;++i)f[i]-=(long double)qweights[q]*Q(q,i)*r/weights[i];}
    return f;
  };
  const auto suffix=std::to_string(std::lround(time*1e6));
  auto context=output("derivative-context-"+suffix+".csv");
  context<<"target_s,dt_s,wall_m,i,old_memory_m,increment_m,active\n";
  for(unsigned i=0;i<ns;++i)context<<time<<','<<dt<<','<<t.wall<<','<<i<<','<<old[i].relaxed_jump<<','<<t.increment[i]<<','<<t.active[i]<<'\n';
  auto report=output("derivative-extended.csv",true);
  if(report.tellp()==0)report<<"time_s,epsilon_m,relative_error,one_percent_biased_tangent_error,max_absolute_error_Pa_m,row,column\n";
  double final_error=0.;
  for(const double eps:{rate_scale*.001,rate_scale*.0005,rate_scale*.00025})
    {long double error=0,norm=0,biased=0,maximum=0;unsigned row=0,column=0;
     for(unsigned j=0;j<ns;++j)
       {if(mode=="front" && !t.active[j])continue;
        std::vector<long double> plus(t.increment.begin(),t.increment.end()),minus(plus);
        plus[j]+=(long double)eps;minus[j]-=(long double)eps;const auto fp=force(plus),fm=force(minus);
        for(unsigned i=0;i<ns;++i)
          {const long double d=(fp[i]-fm[i])/(2*(long double)eps),difference=d-t.jacobian(i,j);
           error+=difference*difference;norm+=d*d;const auto b=d-.99L*t.jacobian(i,j);biased+=b*b;
           if(std::abs(difference)>maximum){maximum=std::abs(difference);row=i;column=j;}}}
     final_error=std::sqrt(error/norm);report<<time<<','<<eps<<','<<final_error<<','<<(double)std::sqrt(biased/norm)<<','<<(double)maximum<<','<<row<<','<<column<<'\n';
     AssertThrow(std::sqrt(biased/norm)>1e-6,ExcMessage("Derivative reference failed biased-tangent negative control"));}
  return final_error;
}
