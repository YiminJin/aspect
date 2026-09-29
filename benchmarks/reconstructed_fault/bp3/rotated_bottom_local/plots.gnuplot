set datafile separator comma
if (!exists('pdf')) pdf=0
if (pdf) set terminal pdfcairo size 14in,12in font 'Sans,13'; else set terminal pngcairo size 1400,1200 font 'Sans,13'
set output (pdf ? 'analysis/final_endpoint.pdf' : 'analysis/final_endpoint.png')
set multiplot layout 3,2 title 'Local rotated-bottom test: t = 2e6 s (23.15 days); A full, B tangent only'
set grid
set key font ',10'
set xrange [0:200]
set xlabel 'Distance up dip from bottom (m)'
set ylabel 'Normal perturbation (kPa)'
L=2000/sqrt(3.)
plot 'analysis/A-final-fault.csv' u (L-$1):(($6-5e7)/1e3) w l lw 2 lc rgb '#c64b3c' t 'A raw', '' u (L-$1):(($7-5e7)/1e3) w l dt 2 lc rgb '#c64b3c' t 'A friction-used', \
     'analysis/B-final-fault.csv' u (L-$1):(($6-5e7)/1e3) w l lw 2 lc rgb '#247ba0' t 'B raw', '' u (L-$1):(($7-5e7)/1e3) w l dt 2 lc rgb '#247ba0' t 'B friction-used'
q=1e-9/(2e-6)*exp((.6+.015*log(1e3))/.025)
tau=5e7*.025*log(q+sqrt(q*q+1))+32038120320./(2*3464)*1e-9
set ylabel 'Shear minus steady prestress (kPa)'
plot 'analysis/A-final-fault.csv' u (L-$1):(($5-tau)/1e3) w l lw 2 lc rgb '#c64b3c' t 'A', 'analysis/B-final-fault.csv' u (L-$1):(($5-tau)/1e3) w l lw 2 lc rgb '#247ba0' t 'B'
set ylabel 'Raw-normal components (kPa)'
plot 'analysis/A-final-fault.csv' u (L-$1):($8/1e3) w l lw 2 lc rgb '#c64b3c' t 'A pressure', '' u (L-$1):($9/1e3) w l dt 2 lc rgb '#c64b3c' t 'A deviatoric', \
     'analysis/B-final-fault.csv' u (L-$1):($8/1e3) w l lw 2 lc rgb '#247ba0' t 'B pressure', '' u (L-$1):($9/1e3) w l dt 2 lc rgb '#247ba0' t 'B deviatoric'
set ylabel 'Reconstructed V / Vp'
plot 'analysis/A-final-fault.csv' u (L-$1):($3/1e-9) w l lw 2 lc rgb '#c64b3c' t 'A', 'analysis/B-final-fault.csv' u (L-$1):($3/1e-9) w l lw 2 lc rgb '#247ba0' t 'B'
set xrange [-150:150]
set xlabel 'Bottom x relative to fault (m)'
set ylabel 'Bottom u_t / Vp'
plot 'analysis/A-final-bottom.csv' u 1:($3/1e-9) w l lw 3 lc rgb '#c64b3c' t 'A', 'analysis/B-final-bottom.csv' u 1:($3/1e-9) w l dt 2 lw 2 lc rgb '#247ba0' t 'B'
set ylabel 'Bottom u_n / Vp'
plot 'analysis/A-final-bottom.csv' u 1:($4/1e-9) w l lw 2 lc rgb '#c64b3c' t 'A', 'analysis/B-final-bottom.csv' u 1:($4/1e-9) w l lw 2 lc rgb '#247ba0' t 'B'
unset multiplot
set output (pdf ? 'analysis/stress_history.pdf' : 'analysis/stress_history.png')
set multiplot layout 2,2 title 'Raw-normal increments from each initial equilibrium; exact Q1 interval integrals'
set xrange [0:24]
set xlabel 'Time (days)'
set ylabel 'Deep 200 m RMS increment (kPa)'
plot 'analysis/A-norms.csv' u ($3/86400):($4/1e3) w lp lc rgb '#c64b3c' t 'A dt', 'analysis/B-norms.csv' u ($3/86400):($4/1e3) w lp lc rgb '#247ba0' t 'B dt', \
     'analysis/A-half-norms.csv' u ($3/86400):($4/1e3) w l dt 2 lc rgb '#c64b3c' t 'A dt/2', 'analysis/B-half-norms.csv' u ($3/86400):($4/1e3) w l dt 2 lc rgb '#247ba0' t 'B dt/2'
set ylabel 'Deep maximum |increment| (kPa)'
plot 'analysis/A-norms.csv' u ($3/86400):($6/1e3) w lp lc rgb '#c64b3c' t 'A dt', 'analysis/B-norms.csv' u ($3/86400):($6/1e3) w lp lc rgb '#247ba0' t 'B dt', \
     'analysis/A-half-norms.csv' u ($3/86400):($6/1e3) w l dt 2 lc rgb '#c64b3c' t 'A dt/2', 'analysis/B-half-norms.csv' u ($3/86400):($6/1e3) w l dt 2 lc rgb '#247ba0' t 'B dt/2'
set ylabel 'Deep RMS step increment / dt (Pa/s)'
plot 'analysis/A-norms.csv' u ($3/86400):5 w lp lc rgb '#c64b3c' t 'A', 'analysis/B-norms.csv' u ($3/86400):5 w lp lc rgb '#247ba0' t 'B'
set ylabel 'Control-region RMS increment (kPa)'
plot 'analysis/A-norms.csv' u ($3/86400):($10/1e3) w lp lc rgb '#c64b3c' t 'A interior', '' u ($3/86400):($16/1e3) w l dt 2 lc rgb '#c64b3c' t 'A upper 200 m', \
     'analysis/B-norms.csv' u ($3/86400):($10/1e3) w lp lc rgb '#247ba0' t 'B interior', '' u ($3/86400):($16/1e3) w l dt 2 lc rgb '#247ba0' t 'B upper 200 m'
unset multiplot
