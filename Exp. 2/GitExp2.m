% =========================================================================
% Information Valuation under Complexity - Experiment 2 Analysis
% =========================================================================
%
% Description:
%   Tests whether decision complexity (operationalized as choice-RT in an
%   independent cohort) predicts the information valuation bias
%   (bid - normative information value) at the trial level. Cohort A
%   provides RT, expected value (EV), and outcome variability (STD) for
%   each trial; Cohort B provides bids and the bias measure. The script:
%       1. Loads and sanity-checks both cohorts (matched trial design).
%       2. Reports descriptive statistics for Cohort B bids.
%       3. Fits two linear mixed-effects models predicting bias from
%          complexity (LMM 1) and from complexity + EV + STD (LMM 2),
%          with random intercepts and slopes by subject. Satterthwaite
%          degrees of freedom are reported for paper-ready output.
%       4. Plots Figure 4: partial-correlation scatter of bias vs.
%          complexity, controlling for EV and STD, with a 95% CI ribbon
%          and binned means ± SE.
%
% Inputs:
%   summary_Part1.csv  - Cohort A trial-level means (RT, EV, STD,
%                              card values).
%   summary_Part2.csv  - Cohort B trial-level means (info-seeking
%                              bid, normative info value, card values).
%   bigT2.mat                - Subject-by-trial table for Cohort B with
%                              fields SubNum, TrialNumber, infoS,
%                              infoValue (used for LMMs).
%
% Outputs:
%   Console: descriptive stats, LMM fixed-effects tables (incl.
%            Satterthwaite df), partial correlation r and p.
%
% =========================================================================

clear; clc; close all;

%% ========================================================================
%% 1. LOAD + SANITY CHECK
%% ========================================================================
part1 = readtable('summary_Part1.csv');   % Cohort A
part2 = readtable('summary_Part2.csv');   % Cohort B

assert(isequal(part1.TrialNumber, part2.TrialNumber), ...
    'Trial numbers mismatch between parts');
for c = {'Card1','Card2','Card3','Card4','Card5','Card6'}
    assert(isequal(part1.(c{1}), part2.(c{1})), 'Card mismatch: %s', c{1});
end
nTrials = height(part1);
fprintf('Loaded %d trials for each cohort.\n', nTrials);

% Trial-level variables
RT      = part1.Mean_ChoiceRT;     % complexity proxy (s)
EV      = part1.Mean_EV;
STD     = part1.StdDev;
infoS   = part2.Mean_InfoSeeking;  % Cohort B mean bid per trial
infoVal = part2.Mean_InfoValue;    % normative info value per trial
bias_T  = infoS - infoVal;         % information valuation bias (per trial)

%% ========================================================================
%% 2. COHORT B DESCRIPTIVE (trial-level)
%% ========================================================================
fprintf('\n=== Cohort B descriptives ===\n');
fprintf('  Mean bid for information: %.2f ± %.2f SD (across trials)\n', ...
    mean(infoS), std(infoS));

%% ========================================================================
%% 3. LINEAR MIXED-EFFECTS MODELS (subject ª trial, Cohort B)
%% ========================================================================
bigT2 = readtable('dataExp2.csv');

% Attach Cohort-A complexity proxy and compute bias per subject-trial
bigT2.RT   = RT(bigT2.TrialNumber);
bigT2.bias = bigT2.infoS - bigT2.infoValue;

% --- Model 1: complexity alone ---
% Random intercept and random slope for RT by subject.
fprintf('\n=== LMM 1: bias ~ RT + (RT | SubNum) ===\n');
lme1 = fitlme(bigT2, 'bias ~ RT + (RT|SubNum)');
disp(lme1);

% Satterthwaite-corrected df (paper-reportable)
[~, ~, stats1] = fixedEffects(lme1, 'DFMethod', 'satterthwaite');
fprintf('-- LMM 1 fixed effects with Satterthwaite df --\n');
disp(stats1);

% --- Model 2: complexity + covariates (EV, uncertainty) ---
% One combined random-effects block = one random intercept + correlated
% random slopes for RT, EV, std. Avoids the redundant-intercepts issue
% that (RT|SubNum) + (EV|SubNum) + (std|SubNum) would produce.
fprintf('\n=== LMM 2: bias ~ RT + EV + std + (1 + RT + EV + std | SubNum) ===\n');
lme2 = fitlme(bigT2, ...
    'bias ~ RT + EV + std + (1 + RT + EV + std | SubNum)');
disp(lme2);

[~, ~, stats2] = fixedEffects(lme2, 'DFMethod', 'satterthwaite');
fprintf('-- LMM 2 fixed effects with Satterthwaite df --\n');
disp(stats2);

%% ========================================================================
%% 4. FIGURE 4 - partial-correlation scatter (bias vs complexity | EV, STD)
%% ========================================================================
% Plot constants
SLATE   = [ 41  52  65] / 255;
HI_BLUE = [ 59 110 143] / 255;
N_BINS  = 12;

% Residualize bias and RT with respect to EV + STD (trial-level)
X = [ones(nTrials, 1), EV, STD];
[~, ~, bias_res] = regress(bias_T, X);
[~, ~, RT_res]   = regress(RT,     X);

[r_val, p_val] = corr(RT_res, bias_res);
fprintf('\n=== Fig 4: partial correlation (RT, bias | EV, STD) ===\n');
fprintf('  r = %.3f, p = %.3g\n', r_val, p_val);

fig = figure('Color', 'w', 'Position', [100 100 550 450]);
ax  = axes(fig); hold(ax, 'on');
ax.Box = 'off'; ax.FontSize = 16;

% Raw scatter (faded)
scatter(RT_res, bias_res, 20, ...
    'MarkerEdgeColor', [0.1 0.1 0.1], 'MarkerEdgeAlpha', 0.15, ...
    'MarkerFaceColor', [0.7 0.7 0.7], 'MarkerFaceAlpha', 0.15);

% Linear fit + 95% CI ribbon
mdl_trial   = fitlm(RT_res, bias_res);
xvals       = linspace(min(RT_res), max(RT_res), 100)';
[yhat, yCI] = predict(mdl_trial, xvals);
fill([xvals; flipud(xvals)], [yCI(:,1); flipud(yCI(:,2))], HI_BLUE, ...
    'EdgeColor', 'none', 'FaceAlpha', 0.5);
plot(xvals, yhat, '-', 'Color', SLATE, 'LineWidth', 2);

% Binned means ± SE (drop last bin to avoid low-N edge)
edges        = linspace(min(RT_res), max(RT_res), N_BINS);
binCenters   = movmean(edges, 2, 'Endpoints', 'discard');
[~,~,binIdx] = histcounts(RT_res, edges);
binMean      = accumarray(binIdx, bias_res, [], @mean);
binSE        = accumarray(binIdx, bias_res, [], @(x) std(x)/sqrt(length(x)));
binMean(end) = NaN; binSE(end) = NaN;
valid = ~isnan(binMean);
errorbar(binCenters(valid), binMean(valid), binSE(valid), 'o', ...
    'Color', [0.2 0.2 0.2], 'MarkerFaceColor', 'w', ...
    'CapSize', 0, 'LineWidth', 1);

xlabel('Complexity (RT residuals)', 'FontSize', 16);
ylabel({'Information Valuation Bias', ...
        'Bid - information value (residuals)'}, 'FontSize', 16);
ylim([-0.6 0.6]);

% Significance marker
text(1, 0.4, '***', 'FontSize', 24, 'FontWeight', 'bold', ...
     'Color', [0.2 0.2 0.2], 'HorizontalAlignment', 'center');

% Over/Under evaluation arrows on the y-axis
ax.Position = [0.25 0.15 0.65 0.78];
annotation('arrow', [0.05 0.05], [0.66 0.86], 'Color', SLATE, 'HeadStyle', 'plain');
annotation('arrow', [0.05 0.05], [0.46 0.26], 'Color', SLATE, 'HeadStyle', 'plain');
axes('Position', [0 0 1 1], 'Visible', 'off');
text(0.015, 0.76, 'Overevaluation',  'Color', SLATE, 'FontName', 'Arial', ...
    'FontSize', 16, 'Rotation', 90, 'HorizontalAlignment', 'center');
text(0.015, 0.36, 'Underevaluation', 'Color', SLATE, 'FontName', 'Arial', ...
    'FontSize', 16, 'Rotation', 90, 'HorizontalAlignment', 'center');

fprintf('\n=== DONE ===\n');