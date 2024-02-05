clear;

[D,s] = cheb(-1,1,49);

omega = linspace(0,pi,50);

x = zeros(50,50);
y = zeros(50,50);

for i=1:50
    for j=1:50
        x(i,j) = s(i)*cos(omega(j));
        y(i,j) = s(i)*sin(omega(j));
    end
end
