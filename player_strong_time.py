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
# display(df_matches.head())
print(len(df_matches))

arr_target_year = np.array(list(range(2008, 2018)))
print(arr_target_year)

arr_target_player = np.array([
    'Roger Federer',
    'Rafael Nadal',
    'Novak Djokovic',
    'Andy Murray',
    'Stanislas Wawrinka',
    'Juan Martin Del Potro',
    # 'Milos Raonic',
    'Kei Nishikori',
    # 'Gael Monfils',
    'Tomas Berdych',
    # 'Jo Wilfried Tsonga',
    'David Ferrer',
    # 'Richard Gasquet',
    # 'Marin Cilic',
    # 'Grigor Dimitrov',
    # 'Dominic Thiem',
    # 'Nick Kyrgios',
    # 'Alexander Zverev'
])

# print(arr_target_player)

df_tmp = df_matches[
    (df_matches['winner_name'].isin(arr_target_player)) &
    (df_matches['loser_name'].isin(arr_target_player))
]
matrix_cnt = np.zeros((len(arr_target_year), len(arr_target_player)), dtype=np.float32)
matrix_rate = np.zeros((len(arr_target_year), len(arr_target_player)), dtype=np.float32)

for i, year in enumerate(arr_target_year):

    for j, player in enumerate(arr_target_player):
        cnt_win = len(df_tmp[(df_tmp['winner_name'] == player) & (df_tmp['year'] == year)])
        cnt_lose = len(df_tmp[(df_tmp['loser_name'] == player) & (df_tmp['year'] == year)])

        rate = 0 if (cnt_win + cnt_lose == 0) else cnt_win / (cnt_win + cnt_lose)

        matrix_cnt[i, j] = cnt_win + cnt_lose
        matrix_rate[i, j] = rate

for j, player in enumerate(arr_target_player):

    if j % 3 == 0:
        fig, axs = plt.subplots(ncols=3, figsize=(15, 3))

    axs[j % 3].plot(arr_target_year, matrix_cnt[:, j], marker='o', color='b', alpha=0.5)
    axs[j % 3].set(title=player, xlabel='year', ylabel='cnt', ylim=[0, 40])

plt.show()

for j, player in enumerate(arr_target_player):

    if j % 3 == 0:
        fig, axs = plt.subplots(ncols=3, figsize=(15, 3))

    axs[j % 3].plot(arr_target_year, matrix_rate[:, j], marker='o', color='r', alpha=0.5)
    axs[j % 3].set(title=player, xlabel='year', ylabel='rate', ylim=[0, 1])

plt.show()

dic_target_year = {}

for year in arr_target_year:

    if year not in dic_target_year:
        dic_target_year[year] = len(dic_target_year)+1

dic_target_player = {}

for player in arr_target_player:

    if player not in dic_target_player:
        dic_target_player[player] = len(dic_tar)

LW = []
GY = []

print('hello world')

for year in arr_target_year:
    for player_a in arr_target_player:
        for player_b in arr_target_player:

            df_tmp = df_matches[
                (df_matches['year'] == year) &
                (df_matches['winner_name'] == player_a) &
                (df_matches['loser_name'] == player_b)
            ]

            for _ in range(len(df_tmp)):

                LW.append([dic_target_player[player_b], dic_target_player[player_a]])
                GY.append(dic_target_year[year])

            df_tmp = df_matches[
                (df_matches['year'] == year) &
                (df_matches['winner_name'] == player_b) &
                (df_matches['loser_name'] == player_a)
            ]

            for _ in range(len(df_tmp)):

                LW.append([dic_target_player[player_a], dic_target_player[player_b]])
                GY.append(dic_target_year[year])

LW = np.array(LW, dtype=np.int32)
GY = np.array(GY, dtype=np.int32)


model = """
    data {
        int N;
        int G;
        int Y;
        int<lower=1> GY[G];
        int<lower=1, upper=N> LW[G, 2];
    }
    parameters {
        ordered[2] performance[G];
        matrix<lower=0>[N, Y] mu;
        matrix<lower=0>[N, Y] s_mu;
        matrix<lower=0>[N, Y] s_pf;
    }
    model {
        for (g in 1:G)
            for (i in 1:2)
                performance[g, i] ~ normal(mu[LW[g, i], GY[g]], s_pf[LW[g, i], GY[g]]);

        for (n in 1:N)
            mu[n, 1] ~ normal(0, s_mu[n, 1]);

        for (n in 1:N)
            for (y in 2:Y)
                mu[n, y] ~ normal(mu[n, y-1], s_mu[n, y]);

        for (n in 1:N)
            s_mu[n] ~ normal(0, 1);

        for (n in 1:N)
            s_pf[n] ~ gamma(10, 10);
    }
"""
data = {
    'N': len(dic_target_player),
    'G': len(LW),
    'Y': len(dic_target_year),
    'GY': GY,
    'LW': LW,
}
fit2 = pystan.stan(model_code=model, data=data, iter=5000, chains=4)

la2 = fit2.extract()
fit2

plt.figure(figsize=(15, 7))
cmap = matplotlib.cm.get_cmap('tab10')

for j, player in enumerate(arr_target_player):

    samples = la2['mu'][:, j, :]

    medians = np.median(samples, axis=0)
    lower, upper = np.percentile(samples, q=[25.0, 75.0], axis=0)

    c = cmap(j)

    plt.plot(arr_target_year, medians, marker='o', label=player, color=c)
    plt.fill_between(arr_target_year, lower, upper, alpha=0.2, color=c)

plt.xlabel('year')
plt.ylabel('latent strength')
plt.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
plt.show()


cmap = matplotlib.cm.get_cmap('tab10')

for j, player in enumerate(arr_target_player):

    if j % 3 == 0:
        fig, axs = plt.subplots(ncols=3, figsize=(15, 3))

    g = axs[j % 3].violinplot(la2['s_pf'][:, j, :], positions=arr_target_year,
                              showmeans=False, showextrema=False, showmedians=False)
    c = cmap(j % 10)

    for pc in g['bodies']:

        pc.set_facecolor(c)
        pc.set_alpha(0.7)

    axs[j % 3].set(title=player, xlabel='year', ylabel='uneven performance')

plt.show()

cmap = matplotlib.cm.get_cmap('tab10')

for j, player in enumerate(arr_target_player):

    if j % 3 == 0:

        fig, axs = plt.subplots(ncols=3, figsize=(15, 3))

    g = axs[j % 3].violinplot(la2['s_mu'][:, j, :], positions=arr_target_year,
                              showmeans=False, showextrema=False, showmedians=False)
    c = cmap(j % 10)

    for pc in g['bodies']:

        pc.set_facecolor(c)
        pc.set_alpha(0.7)

    axs[j % 3].set(title=player, xlabel='year', ylabel='change of strength')

plt.show()
