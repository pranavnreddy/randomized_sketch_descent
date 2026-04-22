n = 100;
max_iter = 5000;
p = 30;

reg = 10 .^ (-3:3);

num_trials = 100;

suboptimalities = zeros(num_trials, max_iter);
feasibilities = zeros(num_trials, max_iter);

m = zeros(length(reg), max_iter);
s = zeros(length(reg), max_iter);

for jj = 1:length(reg)
    reg_param = reg(jj);
    parfor ii = 1:num_trials
        mu = rand(1, n) + 1;
        r = rand(1) + 1;
        Sigma = randn(n);
        Sigma = Sigma' * Sigma;
        [suboptimalities(ii, :), feasibilities(ii, :)] = rsd_markowitz(Sigma, mu, r, reg_param, max_iter, 0, p, 'g');
    end
    m(jj, :) = mean(suboptimalities);
    s(jj, :) = std(suboptimalities);
end

hold on
for jj = 1:length(reg)
    errorbar(m(jj, 1:50:end), s(jj, 1:50:end), 'DisplayName',['$\rho = $' num2str(reg(jj))])
end
hold off
legend(gca,'show', 'Interpreter' ,'latex')
set(gca, 'YScale', 'log')
xlabel('Iterations (every 50)')
ylabel('Suboptimality')
title("Fixed $p = 30$", 'Interpreter','latex')