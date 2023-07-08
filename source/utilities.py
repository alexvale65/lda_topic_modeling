from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
import geopy
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.pyplot import text

geopy.geocoders.options.default_user_agent = "text-mining"


def style_negative(v, props=''):
    return props if v > 0 else None


def get_country_affiliation(x):
    if isinstance(x, str):
        list_countries = [item.split(',')[-1] for item in x.split(';')]
        countries = list(set([country.strip().title() for country in list_countries]))
        rm = []
        for c in countries:
            code, continent = get_continent(c)
            if code == 'Unknown' or continent == 'Unknown':
                print(f'removing <<{c}>> from:\n{countries}')
                print(x)
                print(50 * '-')
                rm.append(c)
        cln_countries = list(set(countries) - set(rm))
        if len(cln_countries) > 0:
            return cln_countries
        print('No country affiliantion available')
        print(50 * '*')
    return


def get_continent(col):
    try:
        cn_a2_code = country_name_to_country_alpha2(col)
    except:
        cn_a2_code = 'Unknown'
    try:
        cn_continent = country_alpha2_to_continent_code(cn_a2_code)
    except:
        cn_continent = 'Unknown'
    return (cn_a2_code, cn_continent)


# #function to get longitude and latitude data from country name
def geolocate(country):
    geolocator = geopy.geocoders.Nominatim()
    try:
        # Geolocate the center of the country
        loc = geolocator.geocode(country[0])
        # And return latitude and longitude
        return (loc.latitude, loc.longitude)
    except:
        # Return missing value
        return np.nan


def select_author_references(x):
    authors_references = []
    if type(x) == str:
        list_full_references = x.strip().lower().split(';')
        for item in list_full_references:
            authors = item.split('., ')
            if len(authors) > 0:
                authors_references += authors[:-1]
        return list(set([author for author in authors_references if len(author) < 30]))


def stack_matrix(vector, k=3):
    df_dummies = pd.get_dummies(pd.DataFrame(vector), prefix='', prefix_sep='').sum(level=0, axis=1)
    coocurrence_df = df_dummies.T.dot(df_dummies)
    coocurrence_df.values[np.tril(np.ones(coocurrence_df.shape)).astype(np.bool)] = 0
    df_stack = coocurrence_df.stack()
    df_stack = df_stack[df_stack >= k].rename_axis(('source', 'target')).reset_index(name='weight')
    df_stack['source'] = df_stack['source'].map(lambda x: x.strip().lower())
    df_stack['target'] = df_stack['target'].map(lambda x: x.strip().lower())
    return df_stack.groupby(['source','target']).sum().reset_index()


def plot_network(network_frame, quantile_threshold=0.99, threshold_weight=2, figsize=(20, 8)):
    q4_connections = network_frame.drop_duplicates(subset=['source'])['num_connections'].quantile(quantile_threshold)
    q4_authors = network_frame[network_frame['num_connections'] > q4_connections]. \
                                                                        drop_duplicates(subset=['source']). \
                                                                        set_index('source')['num_connections'].to_dict()

    fig = plt.figure(1, figsize=figsize, dpi=60)
    net = nx.from_pandas_edgelist(network_frame[(network_frame['source'].isin(q4_authors.keys())) & \
                                                 (network_frame['weight'] > threshold_weight)]
                                  , edge_attr=True
                                  , create_using=nx.DiGraph())
    degrees = dict(net.degree())
    pos = nx.kamada_kawai_layout(net)
    nx.draw(net
            , pos=pos
            , arrows=False
            , node_size=[degrees[i] * 600 for i in degrees]
            )
    for node, (x, y) in pos.items():
        text(x, y, node, fontsize=degrees[node], ha='center', va='center')


def stack_coocurrence_matrix(data_df, field_name):
    values_list = []
    frames = []
    clusters = data_df['Dominant_Topic'].unique()
    for cluster_name in clusters:
        for idx, row in data_df[data_df['Dominant_Topic']==cluster_name].dropna(subset=[field_name]).iterrows():
            _list = list(set([x.strip().lower() for x in row[field_name].split(';')]))
            values_list.append(_list)
        df_stack = stack_matrix(vector=values_list)
        df_stack['cluster'] = cluster_name
        frames.append(df_stack)
    return pd.concat(frames, axis=0)


def getQuantilePosition(data, field):
    LS = data[field].quantile(0.75) + 1.5 * (data[field].quantile(0.75) - data[field].quantile(0.25))
    df_quantile = data[data[field] <= LS].copy()
    q1 = df_quantile[field].quantile(0.25)
    q2 = df_quantile[field].quantile(0.5)
    q3 = df_quantile[field].quantile(0.75)

    data.loc[data[field] <= q1, f'cat{field}'] = 1
    data.loc[(data[field] > q1) & (data[field] <= q2), f'cat{field}'] = 2
    data.loc[(data[field] > q2) & (data[field] <= q3), f'cat{field}'] = 3
    data.loc[(data[field] > q3) & (data[field] <= LS), f'cat{field}'] = 4
    data.loc[data[field] > LS, f'cat{field}'] = 5
    return data, q1, q2, q3, LS


