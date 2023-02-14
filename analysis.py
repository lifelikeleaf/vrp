import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import openpyxl as xl
from openpyxl.styles import PatternFill
from openpyxl.formatting.rule import CellIsRule, FormulaRule
from openpyxl.utils.cell import get_column_letter
import vrp.decomp.helpers as helpers
from vrp.decomp.constants import *

num_subprobs_best_found = np.array([], dtype=int)

def get_best_found(df, name) -> pd.DataFrame:
    cost = df.loc[
        # 1. group by instance name
        # 2. get the index of the min cost within each group
        # 3. select only the rows that correspond to the min cost within each group
        df.groupby(by=KEY_INSTANCE_NAME)[KEY_COST].idxmin(), # rows
        [KEY_INSTANCE_NAME, KEY_NUM_SUBPROBS, KEY_COST] # cols
    ].sort_index()
    # optionally reset the row index; when merged, index is automatically reset
    #.reset_index(drop=True)

    # a diff approach but same result
    cost_wait = df.sort_values(by=[KEY_COST_WAIT]) \
        .groupby(by=KEY_INSTANCE_NAME).head(1).sort_index() \
        .loc[:, [KEY_INSTANCE_NAME, KEY_NUM_SUBPROBS, KEY_COST_WAIT]]

    best_found = pd.merge(cost, cost_wait, on=[KEY_INSTANCE_NAME], suffixes=[f'_{KEY_COST}', f'_{KEY_COST_WAIT}'])

    # print(f'\n{name}')
    # print(best_found)
    global num_subprobs_best_found
    num_subprobs_best_found = np.append(num_subprobs_best_found, best_found[f'{KEY_NUM_SUBPROBS}_{KEY_COST}'].values)

    return best_found


def get_avg(df, name):
    # this key function should be vectorized. It should expect a Series
    # and return a Series with the same shape as the input.
    # Thus it must use pandas.Series vectorized string functions for
    # extracting substrings and stripping underscores '_'.
    # Sort instance names by the ending digits, such that,
    # e.g. R1_10_3 would appear before R1_10_10.
    # Default python string sort would cause R1_10_10 to appear after R1_10_1
    # but before R1_10_2, like this: R1_10_1, R1_10_10, R1_10_2, R1_10_3, etc.
    # NOTE: this quick hack only works for HG instance names
    # i.e. C1_10_1, R1_10_10, RC1_10_9
    sort = lambda x: x.str[6:].str.strip('_').astype(int)
    # must use as_index=False, so the group by key (KEY_INSTANCE_NAME)
    # isn't used as index, otherwise it screws up column assignment
    # in comparision code, where comp['col_name'] = df[KEY_COST] - dfs['euc'][KEY_COST]
    # results in comp having NaN as the result of the column arithmitic
    avg = df.groupby(by=KEY_INSTANCE_NAME, as_index=False).mean() \
        .sort_values(by=KEY_INSTANCE_NAME, key=sort)[[KEY_INSTANCE_NAME, KEY_COST, KEY_COST_WAIT]] \
        .reset_index(drop=True)

    # print(f'\n{name}')
    # print(avg)

    return avg


def percent_diff(col1, col2):
    percent = (col1 - col2) / col2
    return percent


def dump_comparison_data(exp_names, dir_name, sub_dir, output_name, dump_best=False, dump_avg=False):
    for exp_name in exp_names:
        input_file_name = os.path.join(dir_name, f'{dir_name}_{exp_name}.xlsx')
        dfs = dict(
            euc = pd.read_excel(input_file_name, sheet_name='euclidean'),
        )


        '''MODIFY: sheet names and df column names'''

        # versions = ['v1', 'v2_1', 'v2_2', 'v2_3', 'v2_4', 'v2_5', 'v2_6']
        # versions = ['v2_7', 'v2_8', 'v2_9', 'v2_10']
        versions = ['v2_5', 'v2_8', 'v2_9']
        for v in versions:
            dfs[f'ol_{v}'] = pd.read_excel(input_file_name, sheet_name=f'OL_{v}')
            dfs[f'gap_{v}'] = pd.read_excel(input_file_name, sheet_name=f'Gap_{v}')

            ''' `Both` almost never worked well'''
            # dfs[f'both_{v}'] = pd.read_excel(input_file_name, sheet_name=f'Both_{v}')

        '''END MODIFY'''


        basis = pd.read_excel(input_file_name, sheet_name='Basis')

        if dump_best:
            dfs_best = {name: get_best_found(df, name) for name, df in dfs.items()}
            comp_best = pd.DataFrame()
            comp_best[KEY_INSTANCE_NAME] = basis[KEY_INSTANCE_NAME]
            comp_best['euc vs no decomp'] = dfs_best['euc'][KEY_COST] - basis[f'{KEY_COST}_NO_decomp']
            # add an empty column
            comp_best[''] = ''
            output_name_best = f'best_{output_name}'
            dump_comp(dfs_best, comp_best, dir_name, sub_dir, output_name_best, exp_name)

        if dump_avg:
            dfs_avg = {name: get_avg(df, name) for name, df in dfs.items()}
            comp_avg = pd.DataFrame()
            comp_avg[KEY_INSTANCE_NAME] = basis[KEY_INSTANCE_NAME]
            output_name_avg = f'avg_{output_name}'
            dump_comp(dfs_avg, comp_avg, dir_name, sub_dir, output_name_avg, exp_name)


def dump_comp(dfs, comp, dir_name, sub_dir, output_name, sheet_name):
    for name, df in dfs.items():
        if name != 'euc':
            comp[f'{name}_{KEY_COST}'] = df[KEY_COST] - dfs['euc'][KEY_COST]

    # cells containing the string 'N/A' will be set to None below by formatting
    comp['N/A'] = ''

    for name, df in dfs.items():
        if name != 'euc':
            comp[f'{name}_{KEY_COST}_%'] = percent_diff(df[KEY_COST], dfs['euc'][KEY_COST])

    # add an empty column b/t cost and cost_wait
    comp[' '] = ''

    for name, df in dfs.items():
        if name != 'euc':
            comp[f'{name}_{KEY_COST_WAIT}'] = df[KEY_COST_WAIT] - dfs['euc'][KEY_COST_WAIT]

    comp['N/A2'] = ''

    for name, df in dfs.items():
        if name != 'euc':
            comp[f'{name}_{KEY_COST_WAIT}_%'] = percent_diff(df[KEY_COST_WAIT], dfs['euc'][KEY_COST_WAIT])

    dir_path = os.path.join(dir_name, sub_dir)
    helpers.make_dirs(dir_path)
    out = os.path.join(dir_path, output_name)
    helpers.df_to_excel(comp, file_name=out, sheet_name=sheet_name, overlay=False)


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
        # between_rule = CellIsRule(operator='between', formula=['0', '0.01'], fill=yellow_fill)
        '''Do not apply formula to blank cells, i.e. cell.value == None
        The formula uses a relative reference (A2), which will be expanded
        to the range over which the format is applied; this is a peculiar feature of Excel'''
        between_rule = FormulaRule(formula=['AND( NOT( ISBLANK(A2) ), A2 >= 0, A2 <= 0.01 )'], fill=yellow_fill)
        start_row = sheet.min_row + 1 # skip the headers
        start_col = get_column_letter(sheet.min_column)
        end_row = sheet.max_row
        end_col = get_column_letter(sheet.max_column)
        # e.g. range_string = 'B2:F4'
        range_string = f'{start_col}{start_row}:{end_col}{end_row}'
        sheet.conditional_formatting.add(range_string, less_than_rule)
        sheet.conditional_formatting.add(range_string, between_rule)

        headers = sheet[1] # row 1
        for header_cell in headers:
            if header_cell.value is not None and '%' in header_cell.value:
                percent_col = sheet[header_cell.column_letter]
                for i in range(1, len(percent_col)): # skip the header row
                    cell = percent_col[i]
                    cell.number_format = '0.00%'

            if header_cell.value is None or header_cell.value.strip() == '' or 'N/A' in header_cell.value:
                col = sheet[header_cell.column_letter]
                for i in range(len(col)):
                    cell = col[i]
                    cell.value = None

        # add additional rows for counting comparison results per column
        # - how many have diffs < 0? (i.e. performed strictly better)
        # - how many have diffs b/t 0 and 0.01? (i.e. performed almost the same)
        # - how many have diffs <= 0.01? (i.e. performed almost the same or better)
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
                if col_letter == 'A': # first column
                    sheet[cell_loc_1] = 'x<0'
                    sheet[cell_loc_2] = '0<=x<=0.01'
                    sheet[cell_loc_3] = 'x<=0.01'
                else:
                    sheet[cell_loc_1] = f'=COUNTIF({formula_range}, "<0")'
                    sheet[cell_loc_2] = f'=COUNTIFS({formula_range}, ">=0", {formula_range}, "<=0.01")'
                    sheet[cell_loc_3] = f'=COUNTIF({formula_range}, "<=0.01")'

    dir_path = os.path.join(dir_name, sub_dir)
    helpers.make_dirs(dir_path)
    out = os.path.join(dir_path, f'{file_name}_formatted.xlsx')
    wb.save(out)


if __name__ == '__main__':
    '''MODIFY: experiment names and dir name; must match params in experiments.py'''

    exp_names = [
        'k_medoids_C1',
        'k_medoids_C2',
        'k_medoids_R1',
        'k_medoids_R2',
        'k_medoids_RC1',
        'k_medoids_RC2',
        # 'k_medoids_focus_C1',
        # 'k_medoids_focus_C2',
        # 'k_medoids_focus_R1',
        # 'k_medoids_focus_R2',
        # 'k_medoids_focus_RC1',
        # 'k_medoids_focus_RC2',
    ]

    dir_name = 'E13'
    print_num_subprobs = True # dump_best must be True for this to be meaningful
    dump_best = True
    dump_avg = True

    '''END MODIFY'''

    sub_dir = file_name = f'{dir_name}_comparison'

    dump_comparison_data(exp_names, dir_name, sub_dir, file_name, dump_best=dump_best, dump_avg=dump_avg)
    if print_num_subprobs:
        print(num_subprobs_best_found.tolist())
        for i in range(num_subprobs_best_found.min(), num_subprobs_best_found.max() + 1, 1):
            '''
            num_clusters=2: 70
            num_clusters=3: 100
            num_clusters=4: 161
            num_clusters=5: 45
            num_clusters=6: 23
            num_clusters=7: 9
            num_clusters=8: 7
            num_clusters=9: 4
            num_clusters=10: 1
            '''
            print(f'num_clusters={i}: {len(num_subprobs_best_found[num_subprobs_best_found == i])}')
        fig, ax = plt.subplots()
        ax.hist(num_subprobs_best_found, bins=9, linewidth=0.5, edgecolor="white", align='mid', rwidth=0.8)
        plt.show()

    if dump_best:
        conditional_formatting(dir_name, sub_dir, f'best_{file_name}')
    if dump_avg:
        conditional_formatting(dir_name, sub_dir, f'avg_{file_name}')

