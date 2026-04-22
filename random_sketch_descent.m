n = 100;
m = 50;
p = 200;
max_iter = 10000;
reg = 1;
rho = 1;

A = randn(m, n);
B = randn(p, n);
sol = randn(n, 1);
b = A * sol;
c = B * sol;

x = zeros(n, 1);

L = norm(B'*B);

hist = zeros(1, max_iter);
feasibility = zeros(1, max_iter);

for iter = 1:max_iter
    p = randi(n);
    S = randn(n, p);
    P_S = eye(p) - ( (A*S) \ (A*S) );
    [value, grad] = compute_huber(B*x - c, rho);
    grad = B' * grad;
    t = (P_S' * S' * L * S * P_S + eye(p) * reg) \ (P_S' * S' * grad);
    x = x - S * P_S * t;

    hist(iter) = value;
    feasibility(iter) = norm(A*x - b);
end

semilogy(hist)
hold on
semilogy(feasibility)
hold off
legend("Cost", "Feasibility)")