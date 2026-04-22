function [x, hist] = block_kaczmarz(A, b, x0, blockSize, maxIter, tol, reg)
[n, ~] = size(A);
x = x0;

hist = zeros(1,maxIter);

for k = 1:maxIter
    idx = randperm(n, blockSize);
    A_S = A(idx, :);
    b_S = b(idx);

    y = (A_S * A_S' + reg * eye(blockSize)) \ (A_S * x - b_S);
    x = x - A_S'*y;

    r_full = A * x - b;
    hist(k) = norm(r_full);

    if hist(k) < tol
        hist = hist(1:k);
        fprintf('Converged in %d iterations.\n', k);
        return;
    end
end
end