import pandas as pd
import pycountry_convert as pc
import pycountry

def pre_process(df):
    def state_to_region():
        New_England = ('CT', 'ME', 'MA', 'NH', 'RI', 'VT')
        Mid_Atlantic = ('NJ', 'NY', 'PA')
        East_North_Central = ('IL', 'IN', 'MI', 'OH', 'WI')
        West_North_Central = ('IA', 'KS', 'MN', 'MO', 'NE', 'ND','SD')
        South_Atlantic = ('DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'DC', 'WV')
        East_South_Central = ('AL', 'KY', 'MS', 'TN')
        West_South_Central = ('AR', 'LA', 'OK', 'TX')
        Mountain = ('AZ', 'CO', 'ID', 'MT', 'NV', "NM", 'UT', 'WY')
        Pacific = ('AK', 'CA', 'HI', 'OR', 'WA')

        df.loc[:, 'State'] = df.loc[:, 'State'].replace(dict.fromkeys(New_England, 'New_England'))
        df.loc[:, 'State'] = df.loc[:, 'State'].replace(dict.fromkeys(Mid_Atlantic, 'Mid_Atlantic'))
        df.loc[:, 'State'] = df.loc[:, 'State'].replace(dict.fromkeys(East_North_Central, 'West_North_Central'))
        df.loc[:, 'State'] = df.loc[:, 'State'].replace(dict.fromkeys(West_North_Central, 'West_North_Central'))
        df.loc[:, 'State'] = df.loc[:, 'State'].replace(dict.fromkeys(South_Atlantic, 'South_Atlantic' ))
        df.loc[:, 'State'] = df.loc[:, 'State'].replace(dict.fromkeys(East_South_Central, 'East_South_Central' ))
        df.loc[:, 'State'] = df.loc[:, 'State'].replace(dict.fromkeys(West_South_Central, 'West_South_Central'))
        df.loc[:, 'State'] = df.loc[:, 'State'].replace(dict.fromkeys(Mountain, 'Mountain'))
        df.loc[:, 'State'] = df.loc[:, 'State'].replace(dict.fromkeys(Pacific, 'Pacific'))
        # Ask if rest of inputs in State are Canadian?


    def europe_code():
        # found code that made this inferior, but still interesting so function was kept
        e_url = 'https://pkgstore.datahub.io/opendatafortaxjustice/listofeucountries/listofeucountries_csv/data/5ab24e62d2ad8f06b59a0e7ffd7cb556/listofeucountries_csv.csv'
        e = pd.read_csv(e_url)
        # print(e.head())
        europe = e.x.tolist()
        # print(europe)
        list = [1,3,4]
        europe = [x.upper() for x in europe]
        europe.remove("FRANCE")
        df.loc[:, 'Country'] = df.loc[:, 'Country'].replace(dict.fromkeys(europe, 'EUROPE'))


    def country_to_region():
        continents = {
            'NA': 'NORTH_AMERICA',
            'SA': 'SOUTH_AMERICA',
            'AS': 'ASIA',
            'OC': 'AUSTRALIA',
            'AF': 'AFRICA',
            'EU': 'EUROPE'
        }
        countries = list(pycountry.countries)
        countries = [x.name for x in countries]
        countries = [x for x in countries if x not in ['United States', 'France', 'Canada']]
        # country_codes = [pc.country_name_to_country_alpha2(x, cn_name_format="default") for x in countries]
        cont = []
        for country in countries:
            try:
                cont_code = pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country))
                cont.append(continents[cont_code])
            except:
                cont.append('NONE')
        # con = [continents[pc.country_alpha2_to_continent_code(pc.country_name_to_country_alpha2(country))] for country in countries]
        countries = [x.upper() for x in countries]
        continents = dict(zip(countries, cont))

        df.loc[:, 'Country'] = df.loc[:, 'Country'].replace(continents)
        # some names are off because they include full title but this works

    country_to_region()
    state_to_region()
    return df
