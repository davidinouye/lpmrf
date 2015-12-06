function makemex
%MAKE Compiles the mex code (only tested on 64-bit Linux with g++)
mex -largeArrayDims -Ieigen CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" COMPFLAGS="\$COMPFLAGS -openmp" CXXOPTIMFLAGS="\$CXXOPTIMFLAGS -fopenmp -std=c++11" -outdir ../+mrfs/+samplers -cxx ais_mex.cpp -cxx ais.cpp;
mex -largeArrayDims -Ieigen CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" COMPFLAGS="\$COMPFLAGS -openmp" CXXOPTIMFLAGS="\$CXXOPTIMFLAGS -fopenmp" -outdir ../+mrfs/+learners -cxx learnreg_mex.cpp;
mex -largeArrayDims -Ieigen CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" COMPFLAGS="\$COMPFLAGS -openmp -fopenmp" CXXOPTIMFLAGS="\$CXXOPTIMFLAGS -fopenmp" -outdir ../+mrfs/+learners -cxx learnmrf_mex.cpp;
mex -largeArrayDims -Ieigen CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" COMPFLAGS="\$COMPFLAGS -openmp -fopenmp" CXXOPTIMFLAGS="\$CXXOPTIMFLAGS -fopenmp" -outdir ../+mrfs/+learners/+meta -cxx learnalltopicmats_mex.cpp;

end
