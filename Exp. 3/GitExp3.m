% =========================================================================
% Information Valuation under Complexity - Experiment 3 Analysis
% =========================================================================
%
% Description:
%   Analyzes Experiment 3 data to test whether participants under-value
%   information when decisions are complex. For each subject, computes
%   average profit in a 2 (Complexity: Low / High) x 2 (Information /
%   No information) design under two reward definitions:
%       (i)  Reward = expected value of the full 6-card deck.
%       (ii) Reward = a single randomly selected card from the deck.
%  
% Inputs:
%   dataExp3.csv  - Trial-level data with columns including SubjectID,
%                   TrialNumber, Condition (3 = Information, 6 = No info),
%                   HighRT (0 = Low complexity, 1 = High), PlayChoice,
%                   ExpectedValue, and Card1...Card6.
%
% Outputs:
%   avgReward_EV    - nSubjects x 4 matrix (EV-based reward).
%   avgReward_rand  - nSubjects x 4 matrix (single-card reward).
%
% =========================================================================

close all; clear all; clc;

%% ========================================================================
%% 1. LOAD DATA
%% ========================================================================
bigT = readtable('dataExp3.csv');
subjects  = unique(bigT.SubjectID);
nSubjects = length(subjects);

%% ========================================================================
%% 2. BUILD avgReward_EV (reward = Expected Value)
%% Columns: 1 = Low-Info, 2 = Low-NoInfo, 3 = High-Info, 4 = High-NoInfo
%%   Low  = HighRT == 0   High = HighRT == 1
%%   Info = Condition 3   NoInfo = Condition 6
%% ========================================================================
bigT.Reward = bigT.PlayChoice .* bigT.ExpectedValue;

groupDef = {
    @(t) t.HighRT == 0 & t.Condition == 3
    @(t) t.HighRT == 0 & t.Condition == 6
    @(t) t.HighRT == 1 & t.Condition == 3
    @(t) t.HighRT == 1 & t.Condition == 6
};

avgReward_EV = nan(nSubjects, 4);
for s = 1:nSubjects
    subData = bigT(bigT.SubjectID == subjects(s), :);
    for g = 1:4
        m = groupDef{g}(subData);
        if any(m), avgReward_EV(s,g) = mean(subData.Reward(m)); end
    end
end

%% ========================================================================
%% 3. BUILD avgReward_rand (reward = random single card from the deck)
%% For each subject-trial: pick a random card value from the cond-6 row;
%%   - No-info profit = PlayChoice(cond6) * rewardVal
%%   - Info profit    = mean over cond-3 subsets that *contain* rewardVal of
%%                      PlayChoice * rewardVal
%% ========================================================================
rng(42);
allResults = [];  % [SubjectID, TrialNumber, InfoProfit, NoinfoProfit, HighRT]

for s = 1:nSubjects
    subData  = bigT(bigT.SubjectID == subjects(s), :);
    subCards = [subData.Card1, subData.Card2, subData.Card3, ...
                subData.Card4, subData.Card5, subData.Card6];
    trials = unique(subData.TrialNumber);

    for t = 1:length(trials)
        trialMask  = subData.TrialNumber == trials(t);
        trialData  = subData(trialMask, :);
        trialCards = subCards(trialMask, :);

        cond6idx = find(trialData.Condition == 6, 1);
        if isempty(cond6idx), continue; end

        fullArray    = trialCards(cond6idx, :);
        rewardVal    = fullArray(randi(6));
        noinfoProfit = trialData.PlayChoice(cond6idx) * rewardVal;
        highRT       = trialData.HighRT(cond6idx);

        cond3idx  = find(trialData.Condition == 3);
        validSubs = [];
        for c = 1:length(cond3idx)
            shown = trialCards(cond3idx(c), :);
            shown = shown(shown ~= 0);
            if ismember(rewardVal, shown)
                validSubs(end+1) = cond3idx(c); %#ok<SAGROW>
            end
        end

        if ~isempty(validSubs)
            infoProfit = mean(trialData.PlayChoice(validSubs) * rewardVal);
        else
            infoProfit = NaN;
        end

        allResults = [allResults; subjects(s), trials(t), infoProfit, noinfoProfit, highRT]; %#ok<AGROW>
    end
end

resT = array2table(allResults, 'VariableNames', ...
    {'SubjectID','TrialNumber','InfoProfit','NoinfoProfit','HighRT'});

avgReward_rand = nan(nSubjects, 4);
for s = 1:nSubjects
    rows = resT.SubjectID == subjects(s);

    idx = rows & resT.HighRT == 0 & ~isnan(resT.InfoProfit);
    if any(idx), avgReward_rand(s,1) = mean(resT.InfoProfit(idx));   end
    idx = rows & resT.HighRT == 0;
    if any(idx), avgReward_rand(s,2) = mean(resT.NoinfoProfit(idx)); end
    idx = rows & resT.HighRT == 1 & ~isnan(resT.InfoProfit);
    if any(idx), avgReward_rand(s,3) = mean(resT.InfoProfit(idx));   end
    idx = rows & resT.HighRT == 1;
    if any(idx), avgReward_rand(s,4) = mean(resT.NoinfoProfit(idx)); end
end

%% ========================================================================
%% 4. PLOT BOTH 2x2 FIGURES
%% ========================================================================
plot2x2(avgReward_EV,   'Reward = Expected Value',   'Information',          'No information');
plot2x2(avgReward_rand, 'Reward = Random card',      'Information', 'No information');

%% ========================================================================
%% 5. INFORMATION VALUATION BIAS  (EV-based)
%% bias = empirical VOI baseline - (Info profit - No info profit)
%% Baselines: 0.47 (Low complexity), 0.82 (High complexity)
%% ========================================================================
plot_bias(avgReward_EV, 0.47, 0.82);

%% ---- Print means ----
fprintf('\n=== EV reward (mean +/- SEM) ===\n');
print_means(avgReward_EV);
fprintf('\n=== Random-card reward (mean +/- SEM) ===\n');
print_means(avgReward_rand);

fprintf('\n=== DONE ===\n');


%% ========================================================================
%% LOCAL FUNCTIONS
%% ========================================================================
function plot2x2(data4col, titleStr, infoLbl, noinfoLbl)
% PLOT2X2  2x2 (Complexity x Information) profit figure.
%   data4col columns: 1=Low-Info, 2=Low-NoInfo, 3=High-Info, 4=High-NoInfo

SLATE     = [ 44  62  80]/255;
GRAY_LINE = [180 180 180]/255;
NAVY      = [ 60  75 110]/255;
MAUVE     = [170 150 155]/255;

dot_colors = {NAVY, MAUVE, NAVY, MAUVE};
markers    = {'o', '^', 'o', '^'};
xpos       = [1, 2, 4, 5];
jitter_amt = 0.12;
dot_sz     = 100;
meanSz     = 180;

ci95 = @(x) tinv(0.975, max(numel(x)-1,1)) * std(x,'omitnan') / max(sqrt(sum(~isnan(x))),1);

fig = figure('Color','w','Position',[100 100 640 520]);
ax  = axes(fig); hold(ax,'on');
ax.Box = 'off'; ax.TickDir = 'out';
ax.FontName = 'Arial'; ax.FontSize = 14;
ax.XColor = SLATE; ax.YColor = SLATE;
xlim([0 6]);

N = size(data4col, 1);
jittered_x = zeros(N, 4);
for i = 1:4
    jittered_x(:,i) = xpos(i) + (rand(N,1)-0.5)*2*jitter_amt;
end

% --- connecting lines within each complexity block ---
for s = 1:N
    if all(~isnan(data4col(s, [1 2])))
        line([jittered_x(s,1) jittered_x(s,2)], [data4col(s,1) data4col(s,2)], ...
             'Color', [GRAY_LINE 0.4], 'LineWidth', 0.5);
    end
    if all(~isnan(data4col(s, [3 4])))
        line([jittered_x(s,3) jittered_x(s,4)], [data4col(s,3) data4col(s,4)], ...
             'Color', [GRAY_LINE 0.4], 'LineWidth', 0.5);
    end
end

% --- scatter + mean + 95% CI ---
for i = 1:4
    v     = data4col(:, i);
    valid = ~isnan(v);
    scatter(jittered_x(valid,i), v(valid), dot_sz, markers{i}, ...
        'MarkerFaceColor', dot_colors{i}, 'MarkerEdgeColor','none', ...
        'MarkerFaceAlpha', 0.6);
    mu = mean(v,'omitnan'); ci = ci95(v(valid));
    line([xpos(i) xpos(i)], [mu-ci mu+ci], 'Color', SLATE, 'LineWidth', 2.5);
    plot(xpos(i), mu, markers{i}, 'MarkerSize', meanSz/10, ...
        'MarkerEdgeColor', SLATE, 'MarkerFaceColor', 'w', 'LineWidth', 2);
end

% --- labels ---
ax.YLabel.String = 'Profit (points)'; ax.YLabel.FontSize = 18;
ax.XTick = [];

yl = ax.YLim;
text(mean(xpos(1:2)), yl(1)-0.08*range(yl), 'Low',  ...
    'HorizontalAlignment','center','FontSize',18,'Color',SLATE);
text(mean(xpos(3:4)), yl(1)-0.08*range(yl), 'High', ...
    'HorizontalAlignment','center','FontSize',18,'Color',SLATE);
text(mean([xpos(1) xpos(4)]), yl(1)-0.18*range(yl), 'Complexity', ...
    'HorizontalAlignment','center','FontSize',18,'Color',SLATE);

% --- legend ---
h1 = plot(nan, nan, 'o', 'MarkerFaceColor', NAVY,  'MarkerEdgeColor','none', ...
    'MarkerSize', 13, 'LineStyle','none');
h2 = plot(nan, nan, '^', 'MarkerFaceColor', MAUVE, 'MarkerEdgeColor','none', ...
    'MarkerSize', 13, 'LineStyle','none');
legend([h1 h2], {infoLbl, noinfoLbl}, ...
    'Location','northeast','Box','off','FontSize',18,'Orientation','horizontal');

%title(titleStr, 'FontSize', 16, 'FontWeight', 'normal', 'Color', SLATE);
hold(ax,'off');
end

function print_means(data4col)
% PRINT_MEANS  Prints mean +/- SEM for each of the 4 cells.
labels = {'Low-Info ','Low-NoInfo','High-Info','High-NoInfo'};
for g = 1:4
    v = data4col(:, g);
    fprintf('  %s: %.3f +/- %.3f  (n=%d)\n', ...
        labels{g}, mean(v,'omitnan'), ...
        std(v,'omitnan')/sqrt(sum(~isnan(v))), sum(~isnan(v)));
end
end


function plot_bias(data4col, baseLow, baseHigh)
% PLOT_BIAS  Information valuation bias plot.
%   bias = baseline empirical VOI - (Info profit - No info profit)
%   Positive bias = overevaluation (bid > empirical VOI)
%   Negative bias = underevaluation (bid < empirical VOI)

SLATE = [ 44  62  80]/255;
BLUE  = [ 70  65  85]/255;

gain_low  = -(data4col(:,1) - data4col(:,2)) + baseLow;
gain_high = -(data4col(:,3) - data4col(:,4)) + baseHigh;

data_bias = {gain_low, gain_high};
xpos      = [1, 2];
jitter    = 0.12;
dot_sz    = 100;
meanSz    = 180;

ci95 = @(x) tinv(0.975, max(numel(x)-1,1)) * std(x,'omitnan') / max(sqrt(sum(~isnan(x))),1);

fig = figure('Color','w','Position',[100 100 640 520]);
ax  = axes(fig); hold(ax,'on');
ax.Box = 'off'; ax.TickDir = 'out';
ax.FontName = 'Arial'; ax.FontSize = 18;
ax.XColor = SLATE; ax.YColor = SLATE;

allvals = [gain_low(:); gain_high(:)];
pad = 0.08 * range(allvals); if pad == 0, pad = 0.1; end
yl = [min(allvals)-pad, max(allvals)+pad];
ylim(yl); xlim([0.4 2.6]);

line([0 3], [0 0], 'LineStyle','--','Color',SLATE,'LineWidth',1.5);

for i = 1:2
    v  = data_bias{i};
    xx = xpos(i) + (rand(size(v))-0.5)*2*jitter;
    scatter(xx, v, dot_sz, 'o', 'MarkerFaceColor', BLUE, ...
        'MarkerEdgeColor','none','MarkerFaceAlpha',0.7);
    mu = mean(v,'omitnan'); ci = ci95(v);
    line([xpos(i) xpos(i)], [mu-ci mu+ci], 'Color', SLATE, 'LineWidth', 2.5);
    plot(xpos(i), mu, 'o', 'MarkerSize', meanSz/10, ...
        'MarkerEdgeColor', SLATE, 'MarkerFaceColor', 'w', 'LineWidth', 2);
end

ax.YLabel.String = {'Information valuation bias', ...
                    '(Bid - Empirical value of information)'};
ax.YLabel.FontSize = 18;
ax.XTick = [];

text(xpos(1), yl(1)-0.06*range(yl), 'Low',  ...
    'HorizontalAlignment','center','FontSize',18,'Color',SLATE);
text(xpos(2), yl(1)-0.06*range(yl), 'High', ...
    'HorizontalAlignment','center','FontSize',18,'Color',SLATE);
text(mean(xpos), yl(1)-0.16*range(yl), 'Complexity', ...
    'HorizontalAlignment','center','FontSize',18,'Color',SLATE);

% Arrows + labels on y-axis
ax.Position = [0.28 0.15 0.65 0.78];
annotation(fig,'arrow',[0.05 0.05],[0.66 0.86],'Color',SLATE,'HeadStyle','plain');
annotation(fig,'arrow',[0.05 0.05],[0.46 0.26],'Color',SLATE,'HeadStyle','plain');
axes('Position',[0 0 1 1],'Visible','off');
text(0.015, 0.76, 'Overevaluation',  'Color', SLATE, 'FontName','Arial','FontSize',18, ...
    'Rotation',90,'HorizontalAlignment','center');
text(0.015, 0.36, 'Underevaluation', 'Color', SLATE, 'FontName','Arial','FontSize',18, ...
    'Rotation',90,'HorizontalAlignment','center');

hold(ax,'off');
end