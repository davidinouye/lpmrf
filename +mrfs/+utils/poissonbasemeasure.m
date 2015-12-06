function XtBaseMeasure = poissonbasemeasure( Xt )

XtGammaLn = mrfs.utils.splogfactorial( Xt ); % Compute base measure quickly for sparse matrix
XtBaseMeasure = -sum( XtGammaLn, 2 );

end