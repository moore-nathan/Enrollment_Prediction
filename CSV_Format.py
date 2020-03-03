import pandas as pd
import numpy as np

def csv_format(df, cat_names, cont_names, YN):
    # cat_names = ["State", "Country", "Ethnic Code", "Denomination", "Intended Major 1"]  # category names
    # cont_names = ['SAT_COMP', 'ACT_COMP', 'Expected Financial Contribution', 'STANDING', 'HS GPA']
    # YN = ['Enrolled', 'Home State of PA', 'Home County of Cambria', 'Gender', 'Admitted', 'Housing Type',
    #       'Legacy', 'Roster Athlete']

    def panda_strip(df, cat_names):
        # test = df
        # Strips whitespace from all categorical data entries
        for i in cat_names:
            stripped = df[i].str.strip()
            df = df.drop(i, axis="columns")
            df = df.join(stripped)
        return df

    # Stripping the whitespace from all categorical entries
    df = panda_strip(df, cat_names)

    # Stripping the whitespace from all Y/N categories (not actually just Y/n but those with less than 3 types)
    # YN = ['Enrolled', 'Home State of PA', 'Home County of Cambria', 'Gender', 'Admitted', 'Housing Type',
    #       'Legacy', 'Roster Athlete']
    df = panda_strip(df, YN)

    act_to_sat = {36: 1590,
                  35: 1540,
                  34: 1500,
                  33: 1460,
                  32: 1430,
                  31: 1400,
                  30: 1370,
                  29: 1340,
                  28: 1310,
                  27: 1280,
                  26: 1240,
                  25: 1210,
                  24: 1180,
                  23: 1140,
                  22: 1110,
                  21: 1080,
                  20: 1040,
                  19: 1010,
                  18: 970,
                  17: 930,
                  16: 890,
                  15: 850,
                  14: 800,
                  13: 760,
                  12: 710,
                  11: 670,
                  10: 630,
                  9: 590, }

    # reformatting the EFC category from currency to numbers
    # df[df.columns[1:]].replace('[\$,]', '', regex=True).astype(float)
    df.loc[:, 'Expected Financial Contribution'] = \
        df.loc[:, 'Expected Financial Contribution'].replace('[\$,]', '', regex=True).astype(float)

    # cont_names = ['SAT_COMP', 'ACT_COMP', 'Expected Financial Contribution', 'STANDING', 'HS GPA']

    # this is how you have to replace things in a column (can be used for multiple columns at one time
    # df.loc[:, 'STANDING'] = df.loc[:, 'STANDING'].replace(0,np.nan)
    df.loc[:, cont_names] = df.loc[:, cont_names].replace(0, np.nan)

    # Take any instances where SAT is empty and if there is an ACT then fill empty cell with converted ACT
    n = 0
    for i in df['SAT_COMP']:
        if pd.isna(i):
            if pd.notna(df.loc[n, 'ACT_COMP']):
                df.loc[n, 'SAT_COMP'] = act_to_sat[df.loc[n, 'ACT_COMP']]
        n += 1

    # filling all NaN with the mean in their column
    df.loc[:, cont_names] = df.loc[:, cont_names].fillna(df.mean())

    df = df.drop('ACT_COMP', axis='columns')

    # Drop the rows where admitted is not Y then drop the column
    df = df[df.Admitted != 'N']
    df = df.drop('Admitted', axis='columns')

    YN.remove('Admitted')  # removing drop column from list

    # Changing the YN category data to 0's and 1's
    # All blanks will be assumed to be the null 0 option (must mention where that seems weird in paper)
    df.loc[:, YN] = df.loc[:, YN].replace('', 0)
    df.loc[:, YN] = df.loc[:, YN].replace('N', 0)
    df.loc[:, YN] = df.loc[:, YN].replace('Y', 1)
    # Male/Female
    df.loc[:, 'Gender'] = df.loc[:, 'Gender'].replace('M', 1)
    df.loc[:, 'Gender'] = df.loc[:, 'Gender'].replace('F', 0)
    # Housing type had categories of R (resident), C (commuter), and L (???)
    df.loc[:, 'Housing Type'] = df.loc[:, 'Housing Type'].replace('R', 1)
    df.loc[:, 'Housing Type'] = df.loc[:, 'Housing Type'].replace('C', 0)
    df.loc[:, 'Housing Type'] = df.loc[:, 'Housing Type'].replace('L', 2) #replaced this with 2 instead of -1 for testing
    df.loc[:, 'Legacy'] = df.loc[:, 'Legacy'].fillna(0)  # Legacy had Nan instead of blanks

    # Replace empty cells with 'NONE'
    df.replace('', 'NONE', inplace=True)

    # Had Nan in Country for some reason
    df.loc[:, 'Country'] = df.loc[:, 'Country'].fillna('NONE')

    # normalized_df=(df-df.mean())/df.std()

    # Normalizing the EFC column
    df.loc[:, 'Expected Financial Contribution'] = \
        (df.loc[:, 'Expected Financial Contribution'] - df.loc[:, 'Expected Financial Contribution'].mean()) / df.loc[:, 'Expected Financial Contribution'].std()
    # print(df['Expected Financial Contribution'].value_counts())

    return df
