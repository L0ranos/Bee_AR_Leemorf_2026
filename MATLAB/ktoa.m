function   a = ktoa(k)
%KTOA    converts reflection coefficients to AR polynomial
%----
%   Usage:  A = ktoa(K)
%
%      where each column of A contains polynomial coefficients
%        and   "    "    of K contains PARCOR coefficients
%
%  see also ATOK

%---------------------------------------------------------------
% copyright 1994, by C.S. Burrus, J.H. McClellan, A.V. Oppenheim,
% T.W. Parks, R.W. Schafer, & H.W. Schussler.  For use with the book
% "Computer-Based Exercises for Signal Processing Using MATLAB"
% (Prentice-Hall, 1994).
%---------------------------------------------------------------

[M,N] = size(k);
if M==1              %<---- convert row to column
   k = k(:); [M,N] = size(k);
end
a = [ones(1,N); zeros(M,N)];   %-- first row of a is always 1
for i=1:M
  for j=1:N
     a(1:i+1,j) = a(1:i+1,j) + k(i,j)*flipud(conj(a(1:i+1,j)));
  end
end
