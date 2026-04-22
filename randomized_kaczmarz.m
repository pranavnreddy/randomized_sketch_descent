function [x, history] = randomized_kaczmarz(A, b, x0, maxIter, tol, reg)

[n, ~] = size(A);
x = x0;

row_norms_sq = sum(A.^2, 2);
p = row_norms_sq / sum(row_norms_sq);

history = zeros(maxIter, 1);

for iter = 1:maxIter
    ind = randsample(n, 1, true, p);

    ai = A(ind, :);
    bi = b(ind);

    residual = bi - ai * x;
    x = x + (residual / (row_norms_sq(ind) + reg)) * ai';

    r_full = A * x - b;
    history(iter) = norm(r_full);

    if history(iter) < tol
        history = history(1:iter);
        % fprintf('Converged in %d iterations.\n', k);
        return;
    end
end

% fprintf('Reached maximum iterations.\n');
end