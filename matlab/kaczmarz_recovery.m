n = 30;
m = 50;
max_iter = 10000;
reg = 2;

p = 100;
A = randn(n, m);
B = randn(p, m);
sol = randn(m, 1);

[U, Sigma, V] = svd(A);
Sigma(1:n+1:n*10) = 10 * Sigma(1:n+1:n*10);
A_worse = U * Sigma * V';

b = A * sol;
b_worse = A_worse * sol;
c = B * sol;

t = 10 .^ (-5:-2);

suboptimality = zeros(2, max_iter, length(t));
feasibility = zeros(2, max_iter, length(t));

row_norms_sq = sum(A.^2, 2);
p = row_norms_sq / sum(row_norms_sq);

parfor step = 1:length(t)
    x_worse = zeros(m,1);
    z_worse = zeros(m,1);
    
    x = zeros(m,1);
    z = zeros(m,1);

    for iter = 1:max_iter
        % ind = randsample(n, 1, true, p);
        % ai = A(ind, :);
        % bi = b(ind);
        % 
        % residual = bi - ai * x_worse;
        % z_worse = z_worse + t(step) * (residual / (row_norms_sq(ind) + reg)) * ai';
        % x_worse = B' * (B * z_worse - c);
        % 
        % suboptimality(1, iter, step) = norm(B * x_worse - c) / norm(c);
        % feasibility(1, iter, step) = norm(A_worse * x_worse - b_worse) / norm(b_worse);
        % 
        % if(any(isnan(z_worse)))
        %     disp('fail')
        %     break
        % end
    
        ind = randsample(n, 1, true, p);
        ai = A(ind, :);
        bi = b(ind);
    
        residual = bi - ai * x;
        z = z + t(step) * (residual / (row_norms_sq(ind) + reg)) * ai';
        x = B' * (B * z - c);
    
        suboptimality(2, iter, step) = norm(B * x - c) / norm(c);
        feasibility(2, iter, step) = norm(A * x - b) / norm(b);
    end
end

if(0)
    subplot(1, 2, 1)
    title("Relative Error")
    semilogy(suboptimality(1,:))
    hold on
    semilogy(suboptimality(2,:))
    hold off
    
    xlabel('Iterations')
    ylabel('Relative suboptimality')
    legend("Worse Conditioned Matrix", "Original Matrix")
    
    subplot(1, 2, 2)
    title("Relative Feasibility")
    semilogy(feasibility(1, :))
    hold on
    semilogy(feasibility(2, :))
    hold off

    xlabel('Iterations')
    ylabel('Relative feasibility')
    legend("Worse Conditioned Matrix", "Original Matrix")
end

hold on
for step = 1:length(t)
    plot(suboptimality(2, :, step), 'DisplayName',['$t = $' num2str(t(step))])
end
legend(gca,'show', 'Interpreter' ,'latex')
set(gca, 'YScale', 'log')
xlabel('Iterations')
ylabel('Suboptimality')