import pandas as pd
import vrp.decomp.helpers as helpers
from vrp.decomp.constants import *


def get_best_found(df) -> pd.DataFrame:
    x = df.loc[df.groupby(by=KEY_INSTANCE_NAME)[KEY_COST].idxmin(), [KEY_INSTANCE_NAME, KEY_NUM_SUBPROBS, KEY_COST]] # .reset_index(drop=True) # optionally reset the row index
    # a diff approach but same result
    y = df.sort_values(by=[KEY_COST_WAIT]).groupby(by=KEY_INSTANCE_NAME).head(1).sort_index().loc[:, [KEY_INSTANCE_NAME, KEY_NUM_SUBPROBS, KEY_COST_WAIT]]
    best_found = pd.merge(x, y, on=[KEY_INSTANCE_NAME], suffixes=['_cost', '_cost_wait'])
    return best_found


def percent_diff(col1, col2):
    percent = (col1 - col2) / col2 * 100
    return round(percent, 2)


def dump_comparison_data(exp_names):
    for exp_name in exp_names:
        file_name = exp_name + '.xlsx'
        dfs = dict(
            euc = pd.read_excel(file_name, sheet_name='euclidean'),
            tw = pd.read_excel(file_name, sheet_name='TW'),
            tw_gap = pd.read_excel(file_name, sheet_name='TW_Gap'),
            tw_pos_gap = pd.read_excel(file_name, sheet_name='TW_Pos_Gap'),

            euc_norm = pd.read_excel(file_name, sheet_name='euclidean_norm'),
            tw_norm = pd.read_excel(file_name, sheet_name='TW_norm'),
            tw_gap_norm = pd.read_excel(file_name, sheet_name='TW_Gap_norm'),
            tw_pos_gap_norm = pd.read_excel(file_name, sheet_name='TW_Pos_Gap_norm'),
        )

        dfs_best = {name: get_best_found(df) for name, df in dfs.items()}

        # comp = pd.merge(dfs_best['euc'], dfs_best['tw'], on=[KEY_INSTANCE_NAME])
        comp = pd.DataFrame()
        comp[KEY_INSTANCE_NAME] = dfs_best['euc'][KEY_INSTANCE_NAME]

        # comp['diff_'+KEY_COST] = dfs_best['tw_norm'][KEY_COST] - dfs_best['euc_norm'][KEY_COST]
        # comp['diff_'+KEY_COST_WAIT] = dfs_best['tw_norm'][KEY_COST_WAIT] - dfs_best['euc_norm'][KEY_COST_WAIT]
        # print(comp)

        for name, df in dfs_best.items():
            if name != 'euc':
                comp[f'{name}_{KEY_COST}'] = df[KEY_COST] - dfs_best['euc'][KEY_COST]

        for name, df in dfs_best.items():
            if name != 'euc':
                comp[f'{name}_{KEY_COST}_%'] = percent_diff(df[KEY_COST], dfs_best['euc'][KEY_COST])
                # comp[f'{key}_diff_{KEY_COST}_%'] = comp[f'{key}_diff_{KEY_COST}_%'].astype(str) + '%'

        # add a column gap b/t cost and cost_wait
        comp[''] = ''

        for name, df in dfs_best.items():
            if name != 'euc':
                comp[f'{name}_{KEY_COST_WAIT}'] = df[KEY_COST_WAIT] - dfs_best['euc'][KEY_COST_WAIT]

        for name, df in dfs_best.items():
            if name != 'euc':
                comp[f'{name}_{KEY_COST_WAIT}_%'] = percent_diff(df[KEY_COST_WAIT], dfs_best['euc'][KEY_COST_WAIT])

        # print(comp)
        helpers.write_to_excel(comp, file_name='comparison', sheet_name=exp_name, overlay=False)


if __name__ == '__main__':
    exp_names = [
        'k_medoids_C1',
        'k_medoids_C2',
        'k_medoids_R1',
        'k_medoids_R2',
        'k_medoids_RC1',
        'k_medoids_RC2',
    ]

    dump_comparison_data(exp_names)
