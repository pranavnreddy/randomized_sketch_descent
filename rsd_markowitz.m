function [suboptimality, feasibility] = rsd_markowitz(Sigma, mu, r, reg, max_iter, random_P, p, distribution)
n = size(Sigma, 1);

A = [mu; ones(1, n)];
b = [r;1];

x = A \ b;

suboptimality = zeros(1, max_iter);
feasibility = zeros(1, max_iter);

for iter = 1:max_iter
    if(any(random_P))
        p = randi(n);
    end
    if(distribution == 'g')
        % normal random
        S = randn(n, p);
    elseif(distribution == 'u')
        % uniform random
        S = rand(n, p);
    else
        % block sample
        ind = randperm(n, p);
        S = eye(n);
        S = S(:,ind);
    end
    P_S = eye(p) - ( (A*S) \ (A*S) );
    grad = 2 * Sigma * x;
    t = (P_S' * S' * Sigma * S * P_S + reg * eye(p)) \ (P_S' * S' * grad);
    x = x - S * P_S * t;

    suboptimality(iter) = x' * Sigma * x;
    feasibility(iter) = norm(A*x - b);
end

x = sdpvar(n,1);
cost = x' * Sigma * x;
constraints = [mu*x == r, sum(x) == 1];
options = sdpsettings('verbose', 0);
optimize(constraints, cost, options);

suboptimality = suboptimality - value(cost);
end