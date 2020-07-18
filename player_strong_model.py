from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pylab as plt
import pystan

cols = [
    'tourney_id',  # Id of Tournament
    'tourney_name',  # Name of the Tournament
    'surface',  # Surface of the Court (Hard, Clay, Grass)
    'draw_size',  # Number of people in the tournament
    'tourney_level',  # Level of the tournament (A=ATP Tour, D=Davis Cup, G=Grand Slam, M=Masters)
    'tourney_date',  # Start date of tournament
    'match_num',  # Match number
    'winner_id',  # Id of winner
    'winner_seed',  # Seed of winner
    'winner_entry',  # How the winner entered the tournament
    'winner_name',  # Name of winner
    'winner_hand',  # Dominant hand of winner (L=Left, R=Right, U=Unknown?)
    'winner_ht',  # Height in cm of winner
    'winner_ioc',  # Country of winner
    'winner_age',  # Age of winner
    'winner_rank',  # Rank of winner
    'winner_rank_points',  # Rank points of winner
    'loser_id',
    'loser_seed',
    'loser_entry',
    'loser_name',
    'loser_hand',
    'loser_ht',
    'loser_ioc',
    'loser_age',
    'loser_rank',
    'loser_rank_points',
    'score',  # Score
    'best_of',  # Best of X number of sets
    'round',  # Round
    'minutes',  # Match length in minutes
    'w_ace',  # Number of aces for winner
    'w_df',  # Number of double faults for winner
    'w_svpt',  # Number of service points played by winner
    'w_1stIn',  # Number of first serves in for winner
    'w_1stWon',  # Number of first serve points won for winner
    'w_2ndWon',  # Number of second serve points won for winner
    'w_SvGms',  # Number of service games played by winner
    'w_bpSaved',  # Number of break points saved by winner
    'w_bpFaced',  # Number of break points faced by winner
    'l_ace',
    'l_df',
    'l_svpt',
    'l_1stIn',
    'l_1stWon',
    'l_2ndWon',
    'l_SvGms',
    'l_bpSaved',
    'l_bpFaced'
]

df_matches = pd.concat([
    pd.read_csv('./atp_matches_dataset/atp_matches_2000.csv', usecols=cols),
    pd.read_csv('./atp_matches_dataset/atp_matches_2001.csv', usecols=cols),
    pd.read_csv('./atp_matches_dataset/atp_matches_2002.csv', usecols=cols),
    pd.read_csv('./atp_matches_dataset/atp_matches_2003.csv', usecols=cols),
    pd.read_csv('./atp_matches_dataset/atp_matches_2004.csv', usecols=cols),
    pd.read_csv('./atp_matches_dataset/atp_matches_2005.csv', usecols=cols),
    pd.read_csv('./atp_matches_dataset/atp_matches_2006.csv', usecols=cols),
    pd.read_csv('./atp_matches_dataset/atp_matches_2007.csv', usecols=cols),
    pd.read_csv('./atp_matches_dataset/atp_matches_2008.csv', usecols=cols),
    pd.read_csv('./atp_matches_dataset/atp_matches_2009.csv', usecols=cols),
    pd.read_csv('./atp_matches_dataset/atp_matches_2010.csv', usecols=cols),
    pd.read_csv('./atp_matches_dataset/atp_matches_2011.csv', usecols=cols),
    pd.read_csv('./atp_matches_dataset/atp_matches_2012.csv', usecols=cols),
    pd.read_csv('./atp_matches_dataset/atp_matches_2013.csv', usecols=cols),
    pd.read_csv('./atp_matches_dataset/atp_matches_2014.csv', usecols=cols),
    pd.read_csv('./atp_matches_dataset/atp_matches_2015.csv', usecols=cols),
    pd.read_csv('./atp_matches_dataset/atp_matches_2016.csv', usecols=cols),
    pd.read_csv('./atp_matches_dataset/atp_matches_2017.csv', usecols=cols),
])
df_matches = df_matches.dropna(subset=['tourney_date'])
df_matches['year'] = df_matches['tourney_date'].apply(lambda x: int(str(x)[0:4]))
display(df_matches.head())
print(len(df_matches))


arr_target_player = np.array([
    'Roger Federer',
    'Rafael Nadal',
    'Novak Djokovic',
    'Andy Murray',
    'Stanislas Wawrinka',
    'Juan Martin Del Potro',
    'Milos Raonic',
    'Kei Nishikori',
    'Gael Monfils',
    'Tomas Berdych',
    'Jo Wilfried Tsonga',
    'David Ferrer',
    'Richard Gasquet',
    'Marin Cilic',
    'Grigor Dimitrov',
    'Dominic Thiem',
    'Nick Kyrgios',
    'Alexander Zverev'
])

print(arr_target_player)

df_tmp = df_matches[
    (df_matches['year'] >= 2015) &
    (df_matches['winner_name'].isin(arr_target_player)) &
    (df_matches['loser_name'].isin(arr_target_player))
]

arr_cnt = []
arr_rate = []

for player in arr_target_player:
    cnt_win = len(df_tmp[df_tmp['winner_name'] == player])
    cnt_lose = len(df_tmp[df_tmp['loser_name'] == player])

    arr_cnt.append(cnt_win + cnt_lose)
    arr_rate.append(cnt_win / (cnt_win + cnt_lose))

fig, axs = plt.subplots(ncols=2, figsize=(15, 5))

axs[0].bar(x=arr_target_player, height=arr_cnt, color='b', alpha=0.5)
axs[0].set(xlabel='player', ylabel='cnt')

for tick in axs[0].get_xticklabels():
    tick.set_rotation(75)

axs[1].bar(x=arr_target_player, height=arr_rate, color='r', alpha=0.5)
axs[1].set(xlabel='player', ylabel='rate')

for tick in axs[1].get_xticklabels():
    tick.set_rotation(75)

plt.show()

dic_target_player = {}

for player in arr_target_player:

    if player not in dic_target_player:
        dic_target_player[player] = len(dic_target_player) + 1

print(dic_target_player)

LW = []

for player_a in arr_target_player:
    for player_b in arr_target_player:

        df_tmp = df_matches[
            (df_matches['year'] >= 2015) &
            (df_matches['winner_name'] == player_a) &
            (df_matches['loser_name'] == player_b)
        ]

        for _ in range(len(df_tmp)):
            LW.append([dic_target_player[player_b], dic_target_player[player_a]])

        df_tmp = df_matches[
            (df_matches['year'] >= 2015) &
            (df_matches['winner_name'] == player_b) &
            (df_matches['loser_name'] == player_a)
        ]

        for _ in range(len(df_tmp)):
            LW.append([dic_target_player[player_a], dic_target_player[player_b]])

LW = np.array(LW, dtype=np.int32)
print(LW)

model = data {
    int N
    int G
    int < lower = 1, upper = N > LW[G, 2]
}
parameters {
    ordered[2] performance[G]
    vector < lower = 0 > [N] mu
    real < lower = 0 > s_mu
    vector < lower = 0 > [N] s_pf
}
model {
    for (g in 1: G)
    for (i in 1: 2)
    performance[g, i] ~ normal(mu[LW[g, i]], s_pf[LW[g, i]])

    mu ~ normal(0, s_mu)
    s_pf ~ gamma(10, 10)
}

fit1 = pystan.stan(model_code=model, data={'N': len(
    dic_target_player), 'G': len(LW), 'LW': LW}, iter=1000, chains=4)
la1 = fit1.extract()
print(fit1)

plt.figure(figsize=(15, 7))

colors = ['red', 'yellow', 'green', 'blue']

for i, player in enumerate(arr_target_player):
    for j in range(4):

        g = plt.violinplot(la1['mu'][j * 500:(j + 1) * 500, i], positions=[i], showmeans=False, showextrema=False,
                           showmedians=False)

        for pc in g['bodies']:
            pc.set_facecolor(colors[j])

plt.legend(['chain 1', 'chain 2', 'chain 3', 'chain 4'])

plt.xticks(list(range(len(arr_target_player))), arr_target_player)
plt.xticks(rotation=45)

plt.xlabel('player')
plt.ylabel('mu')
plt.show()

plt.figure(figsize=(15, 7))

colors = ['red', 'yellow', 'green', 'blue']

for i, player in enumerate(arr_target_player):
    for j in range(4):

        g = plt.violinplot(la1['s_pf'][j * 500:(j + 1) * 500, i], positions=[i], showmeans=False, showextrema=False,
                           showmedians=False)

        for pc in g['bodies']:
            pc.set_facecolor(colors[j])

plt.legend(['chain 1', 'chain 2', 'chain 3', 'chain 4'])

plt.xticks(list(range(len(arr_target_player))), arr_target_player)
plt.xticks(rotation=45)

plt.xlabel('player')
plt.ylabel('s_pf')
plt.show()

plt.figure(figsize=(15, 7))
cmap = matplotlib.cm.get_cmap('tab10')

for i, player in enumerate(arr_target_player):

    g = plt.violinplot(la1['mu'][:, i], positions=[i], showmeans=False,
                       showextrema=True, showmedians=True)
    c = cmap(i % 10)

    for pc in g['bodies']:
        pc.set_facecolor(c)

    g['cbars'].set_edgecolor(c)
    g['cmaxes'].set_edgecolor(c)
    g['cmedians'].set_edgecolor(c)
    g['cmins'].set_edgecolor(c)

plt.xticks(list(range(len(arr_target_player))), arr_target_player)
plt.xticks(rotation=45)

plt.xlabel('player')
plt.ylabel('latent strength')
plt.show()

# カラーマップの作成コード


def generate_cmap(colors):
    values = range(len(colors))

    vmax = np.ceil(np.max(values))
    color_list = []

    for v, c in zip(values, colors):
        color_list.append((v / vmax, c))

    return LinearSegmentedColormap.from_list('custom_cmap', color_list)


cm = generate_cmap(['white', 'violet'])

summary = np.zeros((len(arr_target_player), 4))

for i, player in enumerate(arr_target_player):
    samples = la1['mu'][:, i]

    median = np.median(samples, axis=0)
    std = np.std(samples, ddof=1)
    lower, upper = np.percentile(samples, q=[25.0, 75.0], axis=0)

    summary[i] = [median, std, lower, upper]

summary = pd.DataFrame(summary, index=arr_target_player, columns=['median', 'std', '25%', '75%'])

summary.style.background_gradient(cmap=cm, axis=0)


# 対象期間中の戦績を可視化してみた結果
cm = generate_cmap(['white', 'violet'])

df_tmp = df_matches[
    (df_matches['year'] >= 2015) &
    (df_matches['winner_name'].isin(arr_target_player)) &
    (df_matches['loser_name'].isin(arr_target_player))
]

summary = np.zeros((len(arr_target_player), len(arr_target_player)), dtype=np.int32)

for i in range(len(arr_target_player)):
    for j in range(i + 1, len(arr_target_player)):
        summary[i, j] = len(df_tmp[
            (df_tmp['winner_name'] == arr_target_player[i]) &
            (df_tmp['loser_name'] == arr_target_player[j])
        ])

        summary[j, i] = len(df_tmp[
            (df_tmp['loser_name'] == arr_target_player[i]) &
            (df_tmp['winner_name'] == arr_target_player[j])
        ])

fig, axs = plt.subplots(figsize=(9, 9))
im = axs.imshow(summary, cmap=cm, interpolation='nearest')

axs.grid(False)
axs.set_xticks(list(range(len(arr_target_player))))
axs.set_yticks(list(range(len(arr_target_player))))
axs.set_xticklabels(arr_target_player)
axs.set_yticklabels(arr_target_player)
axs.set_ylabel('winner')
axs.set_xlabel('loser')

for tick in axs.get_xticklabels():
    tick.set_rotation(75)

for i in range(len(arr_target_player)):
    for j in range(len(arr_target_player)):
        text = axs.text(j, i, summary[i, j], ha='center', va='center', color='black')

fig.tight_layout()
plt.show()


# 各選手の勝負ムラ \sigma_{pf} の事後分布
plt.figure(figsize=(15, 7))
cmap = matplotlib.cm.get_cmap('tab10')

for i, player in enumerate(arr_target_player):

    g = plt.violinplot(la1['s_pf'][:, i], positions=[i], showmeans=False,
                       showextrema=True, showmedians=True)
    c = cmap(i % 10)

    for pc in g['bodies']:
        pc.set_facecolor(c)

    g['cbars'].set_edgecolor(c)
    g['cmaxes'].set_edgecolor(c)
    g['cmedians'].set_edgecolor(c)
    g['cmins'].set_edgecolor(c)

plt.xticks(list(range(len(arr_target_player))), arr_target_player)
plt.xticks(rotation=45)

plt.xlabel('player')
plt.ylabel('uneven performance')
plt.show()


cm = generate_cmap(['white', 'violet'])

summary = np.zeros((len(arr_target_player), 4))

for i, player in enumerate(arr_target_player):
    samples = la1['s_pf'][:, i]

    median = np.median(samples, axis=0)
    std = np.std(samples, ddof=1)
    lower, upper = np.percentile(samples, q=[25.0, 75.0], axis=0)

    summary[i] = [median, std, lower, upper]

summary = pd.DataFrame(summary, index=arr_target_player, columns=['median', 'std', '25%', '75%'])

summary.style.background_gradient(cmap=cm, axis=0)
