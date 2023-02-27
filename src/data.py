import folktables
from folktables import ACSDataSource
import pandas as pd
import numpy as np



def load_folktables_income(sensitive_attributes="sex"):
    if sensitive_attributes == "sex":
        features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
        group = "SEX"
    elif sensitive_attributes == "race":
        features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
        group = "RAC1P"
    else:
        features = ['AGEP','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P']
        group = "SEX"


    def adult_filter(data):
        """Mimic the filters in place for Adult data.

        Adult documentation notes: Extraction was done by Barry Becker from
        the 1994 Census database. A set of reasonably clean records was extracted
        using the following conditions:
        ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
        """
        df = data
        df = df[df['AGEP'] > 16]
        df = df[df['PINCP'] > 100]
        df = df[df['WKHP'] > 0]
        df = df[df['PWGTP'] >= 1]
        return df


    ACSIncome = folktables.BasicProblem(
        features=features,
        target='PINCP',
        target_transform=lambda x: x > 50000,
        group=group,
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )


    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person',
                                root_dir="/data/han/data/fairness/folktables/")
    acs_data = data_source.get_data(states=["CA"], download=True)
    features, label, group = ACSIncome.df_to_numpy(acs_data)

    X = pd.DataFrame(features, columns=ACSIncome.features)
    y = pd.Series(label)
    s = pd.Series(group, name=ACSIncome.group).to_frame()
    if sensitive_attributes == "sex":
        s["SEX"] = s["SEX"] - 1
    elif sensitive_attributes == "race":
        s["RAC1P"] = (s["RAC1P"] == 1).astype(np.int)

    X = pd.get_dummies(X, columns=ACSIncome.features)

    return X, y, s


