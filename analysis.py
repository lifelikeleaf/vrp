import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl as xl
from openpyxl.styles import PatternFill
from openpyxl.formatting.rule import CellIsRule
from openpyxl.utils.cell import get_column_letter
import vrp.decomp.helpers as helpers
from vrp.decomp.constants import *

num_subprobs = np.array([])

def get_best_found(df, name) -> pd.DataFrame:
    x = df.loc[
        # 1. group by instance name
        # 2. get the index of the min cost within each group
        # 3. select only the rows that correspond to the min cost within each group
        df.groupby(by=KEY_INSTANCE_NAME)[KEY_COST].idxmin(), # rows
        [KEY_INSTANCE_NAME, KEY_NUM_SUBPROBS, KEY_COST] # cols
    ] # .reset_index(drop=True) # optionally reset the row index

    # a diff approach but same result
    y = df.sort_values(by=[KEY_COST_WAIT]) \
        .groupby(by=KEY_INSTANCE_NAME).head(1).sort_index() \
        .loc[:, [KEY_INSTANCE_NAME, KEY_NUM_SUBPROBS, KEY_COST_WAIT]]

    best_found = pd.merge(x, y, on=[KEY_INSTANCE_NAME], suffixes=['_cost', '_cost_wait'])

    print(f'\n{name}')
    print(best_found)
    global num_subprobs
    num_subprobs = np.append(num_subprobs, best_found['num_subprobs_cost'].values)

    return best_found


def get_avg(df):
    pass


def percent_diff(col1, col2):
    percent = (col1 - col2) / col2
    return percent


def dump_comparison_data(exp_names, dir_name, sub_dir, output_name, print_best_found_only=False):
    for exp_name in exp_names:
        input_file_name = os.path.join(dir_name, f'{dir_name}_{exp_name}.xlsx')
        dfs = dict(
            euc = pd.read_excel(input_file_name, sheet_name='euclidean'),
        )


        '''MODIFY: sheet names and df column names'''

        # versions = ['v2_5_vectorized', 'v2_6_vectorized']
        versions = ['v2_1', 'v2_2', 'v2_3']
        for v in versions:
            dfs[f'ol_{v}'] = pd.read_excel(input_file_name, sheet_name=f'OL_{v}')
            dfs[f'gap_{v}'] = pd.read_excel(input_file_name, sheet_name=f'Gap_{v}')
            dfs[f'both_{v}'] = pd.read_excel(input_file_name, sheet_name=f'Both_{v}')

        '''END MODIFY'''


        dfs_best = {name: get_best_found(df, name) for name, df in dfs.items()}

        if print_best_found_only:
            # only print the DFs from get_best_found
            # do not output to excel
            continue

        comp = pd.DataFrame()
        comp[KEY_INSTANCE_NAME] = dfs_best['euc'][KEY_INSTANCE_NAME]

        for name, df in dfs_best.items():
            if name != 'euc':
                comp[f'{name}_{KEY_COST}'] = df[KEY_COST] - dfs_best['euc'][KEY_COST]

        for name, df in dfs_best.items():
            if name != 'euc':
                comp[f'{name}_{KEY_COST}_%'] = percent_diff(df[KEY_COST], dfs_best['euc'][KEY_COST])

        # add a column gap b/t cost and cost_wait
        comp[''] = ''

        for name, df in dfs_best.items():
            if name != 'euc':
                comp[f'{name}_{KEY_COST_WAIT}'] = df[KEY_COST_WAIT] - dfs_best['euc'][KEY_COST_WAIT]

        for name, df in dfs_best.items():
            if name != 'euc':
                comp[f'{name}_{KEY_COST_WAIT}_%'] = percent_diff(df[KEY_COST_WAIT], dfs_best['euc'][KEY_COST_WAIT])

        dir_path = os.path.join(dir_name, sub_dir)
        helpers.make_dirs(dir_path)
        out = os.path.join(dir_path, output_name)
        helpers.write_to_excel(comp, file_name=out, sheet_name=exp_name, overlay=False)


def conditional_formatting(dir_name, sub_dir, file_name):
    file_to_load = os.path.join(dir_name, sub_dir, f'{file_name}.xlsx')
    wb = xl.load_workbook(file_to_load)

    for sheet in wb:
        # conditional color formatting:
        # - highlight cell value < 0 green
        # - highlight cell value b/t 0 and 0.01 yellow (i.e. b/t 0% and 1%)
        green_fill = PatternFill(bgColor='00FF00', fill_type='solid')
        yellow_fill = PatternFill(bgColor='FFFF00', fill_type='solid')
        less_than_rule = CellIsRule(operator='lessThan', formula=['0'], fill=green_fill)
        between_rule = CellIsRule(operator='between', formula=['0', '0.01'], fill=yellow_fill)
        start_row = sheet.min_row + 1
        start_col = get_column_letter(sheet.min_column)
        end_row = sheet.max_row
        end_col = get_column_letter(sheet.max_column)
        # e.g. range_string = 'B2:F4'
        range_string = f'{start_col}{start_row}:{end_col}{end_row}'
        sheet.conditional_formatting.add(range_string, less_than_rule)
        sheet.conditional_formatting.add(range_string, between_rule)

        headers = sheet[1] # row 1
        for cell in headers:
            if cell.value is not None and '%' in cell.value:
                percent_col = sheet[cell.column_letter]
                for i in range(1, len(percent_col)): # skip the header row
                    cell = percent_col[i]
                    cell.number_format = '0.00%'

        # add additional rows for counting comparison results per column
        # - how many have diffs < 0? (i.e. performed strictly better)
        # - how many have diffs b/t 0 and 1? (i.e. performed almost the same)
        # - how many have diffs <= 1? (i.e. performed almost the same or better)
        cur_max_row = sheet.max_row
        cols_gen = sheet.columns
        # next(cols_gen) # skip the first column which contains instance names
        for col in cols_gen:
            cell = col[0]
            if cell.value is not None:
                col_letter = cell.column_letter
                cell_loc_1 = f'{col_letter}{cur_max_row + 2}'
                cell_loc_2 = f'{col_letter}{cur_max_row + 3}'
                cell_loc_3 = f'{col_letter}{cur_max_row + 4}'
                formula_range = f'{col_letter}{start_row}:{col_letter}{cur_max_row}'
                if col_letter == 'A':
                    sheet[cell_loc_1] = 'x<0'
                    sheet[cell_loc_2] = '0<=x<=1'
                    sheet[cell_loc_3] = 'x<=1'
                else:
                    sheet[cell_loc_1] = f'=COUNTIF({formula_range}, "<0")'
                    sheet[cell_loc_2] = f'=COUNTIFS({formula_range}, ">=0", {formula_range}, "<=1")'
                    sheet[cell_loc_3] = f'=COUNTIF({formula_range}, "<=1")'

    dir_path = os.path.join(dir_name, sub_dir)
    helpers.make_dirs(dir_path)
    out = os.path.join(dir_path, f'{file_name}_formatted.xlsx')
    wb.save(out)


if __name__ == '__main__':
    '''MODIFY: experiment names and dir name; must match params in experiments.py'''

    exp_names = [
        'k_medoids_C1',
        # 'k_medoids_C2',
        'k_medoids_R1',
        # 'k_medoids_R2',
        # 'k_medoids_RC1',
        'k_medoids_RC2',
        # 'k_medoids_focus_C1',
        # 'k_medoids_focus_C2',
        # 'k_medoids_focus_R1',
        # 'k_medoids_focus_R2',
        # 'k_medoids_focus_RC1',
        # 'k_medoids_focus_RC2',
    ]

    dir_name = 'E6'
    print_best_found_only = True

    '''END MODIFY'''

    sub_dir = file_name = f'{dir_name}_comparison'

    dump_comparison_data(exp_names, dir_name, sub_dir, file_name, print_best_found_only=print_best_found_only)
    if print_best_found_only:
        print(num_subprobs)
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.hist(num_subprobs)
        plt.show()

    # conditional_formatting(dir_name, sub_dir, file_name)

