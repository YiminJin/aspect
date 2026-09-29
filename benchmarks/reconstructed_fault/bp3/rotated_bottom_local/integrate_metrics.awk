# Exact integrals of the exported piecewise-linear fault fields over fixed
# physical intervals. Input: local_fault_N.csv. No smoothing or extrapolation.
BEGIN {
  FS=OFS=","; L=1000/sqrt(3)*2;
  lo[0]=L-200; hi[0]=L; lo[1]=200; hi[1]=L-200; lo[2]=0; hi[2]=200;
  for(k=0;k<3;k++) vmin[k]=1e100;
}
FNR==1 {next}
{
  if(FNR>2) for(k=0;k<3;k++) {
    a=px<$1?px:$1; b=px>$1?px:$1;
    if(a<lo[k]) a=lo[k]; if(b>hi[k]) b=hi[k];
    if(b<=a) continue;
    fa=(a-px)/($1-px); fb=(b-px)/($1-px);
    da=pd+fa*($10-pd); db=pd+fb*($10-pd);
    ra=pr+fa*($11-pr); rb=pr+fb*($11-pr);
    va=pv+fa*($3-pv); vb=pv+fb*($3-pv);
    norm[k]+=(b-a)*(da*da+da*db+db*db)/3;
    rate[k]+=(b-a)*(ra*ra+ra*rb+rb*rb)/3;
    measure[k]+=b-a;
    if(sqrt(da*da)>max[k]) {max[k]=sqrt(da*da); where[k]=a;}
    if(sqrt(db*db)>max[k]) {max[k]=sqrt(db*db); where[k]=b;}
    if(va<vmin[k]) vmin[k]=va; if(vb<vmin[k]) vmin[k]=vb;
    if(va>vmax[k]) vmax[k]=va; if(vb>vmax[k]) vmax[k]=vb;
  }
  px=$1; pd=$10; pr=$11; pv=$3;
}
END {
  printf "%s,%d,%.17g", name,step,time;
  for(k=0;k<3;k++) {
    if(sqrt((measure[k]-(hi[k]-lo[k]))^2)>1e-7) exit 2;
    printf ",%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",sqrt(norm[k]/measure[k]),sqrt(rate[k]/measure[k]),max[k],where[k],vmin[k]/1e-9,vmax[k]/1e-9;
  }
  printf "\n";
}
