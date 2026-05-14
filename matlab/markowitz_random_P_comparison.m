n = 100;
max_iter = 5000;
PP = [10, 50];

reg = 1;

num_trials = 100;

suboptimalities = zeros(num_trials, max_iter);
feasibilities = zeros(num_trials, max_iter);

m = zeros(2, max_iter);
s = zeros(2, max_iter);

hold on

for p = PP
    parfor ii = 1:num_trials
        mu = rand(1, n) + 1;
        r = rand(1) + 1;
        Sigma = randn(n);
        Sigma = Sigma' * Sigma;
        [suboptimalities(ii, :), feasibilities(ii, :)] = rsd_markowitz(Sigma, mu, r, reg, max_iter, 0, p, 'g');
    end
    m(1, :) = mean(suboptimalities);
    s(1, :) = std(suboptimalities);
    errorbar(m(1, 1:50:end), s(1, 1:50:end), 'DisplayName', ['Gaussian, $p = $' num2str(p)])

    parfor ii = 1:num_trials
        mu = rand(1, n) + 1;
        r = rand(1) + 1;
        Sigma = randn(n);
        Sigma = Sigma' * Sigma;
        [suboptimalities(ii, :), feasibilities(ii, :)] = rsd_markowitz(Sigma, mu, r, reg, max_iter, 0, p, 'u');
    end
    m(1, :) = mean(suboptimalities);
    s(1, :) = std(suboptimalities);
    errorbar(m(1, 1:50:end), s(1, 1:50:end), 'DisplayName', ['Uniform, $p = $' num2str(p)])


    parfor ii = 1:num_trials
        mu = rand(1, n) + 1;
        r = rand(1) + 1;
        Sigma = randn(n);
        Sigma = Sigma' * Sigma;
        [suboptimalities(ii, :), feasibilities(ii, :)] = rsd_markowitz(Sigma, mu, r, reg, max_iter, 0, p, 'g');
    end
    m(1, :) = mean(suboptimalities);
    s(1, :) = std(suboptimalities);
    errorbar(m(1, 1:50:end), s(1, 1:50:end), 'DisplayName', ['Block, $p = $' num2str(p)])
end

parfor ii = 1:num_trials
    mu = rand(1, n) + 1;
    r = rand(1) + 1;
    Sigma = randn(n);
    Sigma = Sigma' * Sigma;
    [suboptimalities(ii, :), feasibilities(ii, :)] = rsd_markowitz(Sigma, mu, r, reg, max_iter, 1, 0, 'g');
end
m(2, :) = mean(suboptimalities);
s(2, :) = std(suboptimalities);
errorbar(m(2, 1:50:end), s(2, 1:50:end), 'DisplayName','Gaussian Random p')


parfor ii = 1:num_trials
    mu = rand(1, n) + 1;
    r = rand(1) + 1;
    Sigma = randn(n);
    Sigma = Sigma' * Sigma;
    [suboptimalities(ii, :), feasibilities(ii, :)] = rsd_markowitz(Sigma, mu, r, reg, max_iter, 1, 0, 'u');
end
m(2, :) = mean(suboptimalities);
s(2, :) = std(suboptimalities);
errorbar(m(2, 1:50:end), s(2, 1:50:end), 'DisplayName','Uniform Random p')



parfor ii = 1:num_trials
    mu = rand(1, n) + 1;
    r = rand(1) + 1;
    Sigma = randn(n);
    Sigma = Sigma' * Sigma;
    [suboptimalities(ii, :), feasibilities(ii, :)] = rsd_markowitz(Sigma, mu, r, reg, max_iter, 1, 0, 'b');
end
m(2, :) = mean(suboptimalities);
s(2, :) = std(suboptimalities);
errorbar(m(2, 1:50:end), s(2, 1:50:end), 'DisplayName','Block Random p')

hold off
legend(gca,'show', 'Interpreter' ,'latex')
set(gca, 'YScale', 'log')
xlabel('Iterations (every 50)')
ylabel('Suboptimality')
title("Comparison of Distributions", 'Interpreter','latex')