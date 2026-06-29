function [ro,fP,bP] = lee_morf_new(x,P,la)
% ===========================
% filtru adaptacyjny Lee-Morf'a lub Friedlandera 
% dziala rowniez dla wektorow zespolonych
% ===========================
% using - [ro,fP,bP] = lee_morf_new(x,P,la)
% ro    - Schur matrix with size T x P
% fP    - sygnal innowacyjny po ostatniej sekcji filtru (błąd w przód)
% bP    - sygnal błędu w tył po ostatniej sekcji filtru
% P     - rozmiar modelu; rząd filtru
% x     - N długość sygnału
% la    - lambda "współczynnik zapominania" \in(0:1]

% literatura
% [1] D.T.L. Lee, M. Morf, and B. Friedlander, “Recursive least squares ladder estimation algorithms,” IEEE Trans. Acoust., Speech, Signal Process. , vol. 29, pp. 627–641, 1981
% [2] J.Zarzycki, "Cyfrowa filtracja ortogonalna sygnałów losowych", Wydawnictwo Naukowo-Techniczne, Warszawa 1998
% [3] M. Niedźwiecki, M. Meller and D. Chojnacki, "Lattice filter based autoregressive spectrum estimation with joint model order and estimation bandwidth adaptation", 56th Annual Conference on Decision and Control CDC, 2017, Melbourne, Australia

x = x(:);
N = size(x,1);
fP = zeros(N,1);
bP = zeros(N,1);
ro = zeros(N+1,P);

if(nargin<3), la=0.999; end
la1 = 1-la;

% tablice współczynników
f = zeros(1,P+1);
b = zeros(2,P+1);
c = 1;

%% pętla po czasie
for n=1:N
    % pierwsza sekcja
    c = la*c + x(n)*x(n)';
    f(1,1) = x(n)/sqrt(c); b(1,1) = -f(1,1);
    % kolejne sekcje
    for p=1:P
        ff = sqrt(1-(f(1,p)*f(1,p)'));
        bb = sqrt(1-(b(2,p)*b(2,p)'));
        ro(n+1,p) = (ro(n,p) * ff * bb) + (f(1,p)*b(2,p)');
        roro = sqrt(1 - (ro(n+1,p)*ro(n+1,p)'));
        
        f(1,p+1) = (f(1,p) - ro(n+1,p)'*b(2,p))/(roro*bb);
        b(1,p+1) = (b(2,p) - ro(n+1,p)'*f(1,p))/(roro*ff);
    end
    % opóźnienie
    fP(n) = f(P+1); bP(n) = b(1,P+1);
    b(2,:) = b(1,:);
end
ro = ro(2:N+1,:);

if nargout==0,
    figure();plot(ro);legend('\rho_1','\rho_2','\rho_3','\rho_4','\rho_5','\rho_6','\rho_7','\rho_8');
    figure();plot(fP);
    figure();plot(bP);
end