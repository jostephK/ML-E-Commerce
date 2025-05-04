import pandas as pd
import argparse

country_codes = {
    "Australia": 0,
    "Austria": 1,
    "Bahrain": 2,
    "Belgium":	3,
    "Brazil":	4,
    "Canada":	5,
    "Channel Islands":	6,
    "Cyprus":	7,
    "Czech Republic":	8,
    "Denmark":	9,
    "EIRE":	10,
    "European Community":	11,
    "Finland":	12,
    "France":	13,
    "Germany":	14,
    "Greece":	15,
    "Hong Kong":	16,
    "Iceland":	17,
    "Israel":	18,
    "Italy":	19,
    "Japan":	20,
    "Lebanon":	21,
    "Lithuania":	22,
    "Malta":	23,
    "Netherlands":	24,
    "Norway":	25,
    "Poland":	26,
    "Portugal":	27,
    "RSA":	28,
    "Saudi Arabia":	29,
    "Singapore":	30,
    "Spain":	31,
    "Sweden":	32,
    "Switzerland":	33,
    "United Arab Emirates":	34,
    "United Kingdom":	35,
    "Unspecified":	36,
    "USA":	37
}

def preprocess_csv(filepath, deletecolumns):
    # reads the csv data and converts it to pandas dataframe
    data = pd.read_csv(filepath, encoding='ISO-8859-1')

    # drop any of the columns that are a part of the deletecolumns
    data.drop(columns=deletecolumns, inplace=True, errors='ignore')

    # drop any rows where the quantity column is negative
    data = data[data['Quantity'] >= 0].reset_index(drop=True)
    data = data[data['UnitPrice'] >= 0].reset_index(drop=True)
    # delete outliers
    data = data[data['Quantity'] < 3000].reset_index(drop=True)

    # replace all of the Country names with the associated country_codes
    data['Country'] = data['Country'].map(country_codes)

    return data
