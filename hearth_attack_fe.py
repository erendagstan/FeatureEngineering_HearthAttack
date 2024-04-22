import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


def load():
    data = pd.read_csv('feature_engineering/hearth_attack/heart_attack_prediction_dataset.csv')
    return data


df = load()
df.head()


def quick_look(dataframe, num=5):
    print("################ Shape ################")
    print(dataframe.shape)
    print("################ Types ################")
    print(dataframe.dtypes)
    print("################ Head ################")
    print(dataframe.head(num))
    print("################ Tail ################")
    print(dataframe.tail(num))
    print("################ NA ################")
    print(dataframe.isnull().sum())
    print("################ Quantiles ################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


quick_look(df)

# Aykırı Degerler

# boxplot
sns.boxplot(x=df['BMI'])
plt.show()
plt.close()
plt.boxplot(x=df['Cholesterol'])
plt.show()

# IQR

quartile1 = df['BMI'].quantile(.25)
quartile3 = df['BMI'].quantile(.75)
iqr = quartile3 - quartile1
low = quartile1 - 1.5 * iqr
up = quartile3 + 1.5 * iqr

df[(df['BMI'] < low) | (df['BMI'] > up)]
df[(df['BMI'] < low) | (df['BMI'] > up)].index


def outlier_thresholds(dataframe, col, q1=0.01, q3=0.99):
    quartile1 = dataframe[col].quantile(q1)
    quartile3 = dataframe[col].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * iqr
    up_limit = quartile3 + 1.5 * iqr
    return low_limit, up_limit


def check_outlier(dataframe, col):
    low, up = outlier_thresholds(dataframe, col)
    if (dataframe[(dataframe[col] < low) | (dataframe[col] > up)]).any(axis=None):
        return True
    else:
        return False


check_outlier(dataframe=df, col='Sleep Hours Per Day')
check_outlier(dataframe=df, col='Income')
check_outlier(dataframe=df, col='Triglycerides')

cat_cols = [col for col in df.columns if df[col].dtype == 'O']
cat_but_car = [col for col in df.columns if (df[col].dtype == 'O') & (df[col].nunique() > 20)]
num_cols = [col for col in df.columns if df[col].dtypes in ['int32', 'int64', 'float64']]
num_but_cat = [col for col in df.columns if (col in num_cols) & (df[col].nunique() < 5)]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]
num_cols = [col for col in num_cols if col not in num_but_cat]

df.head()
df['Blood Pressure'].nunique()


# num_cols, cat_cols, cat_but_car
def grab_col_names(dataframe, cat_th=5, car_th=20):
    """

    Parameters
    ----------
    dataframe
    cat_th
    car_th

    Returns
    -------
    num_cols : numeric columns in dataframe
    cat_cols : categoric columns in dataframe
    cat_but_car : look like categoric but cardinal columns in dataframe
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == 'O']
    cat_but_car = [col for col in dataframe.columns if
                   (dataframe[col].dtype == 'O') & (dataframe[col].nunique() > car_th)]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ['int32', 'int64', 'float64']]
    num_but_cat = [col for col in dataframe.columns if (col in num_cols) & (dataframe[col].nunique() < cat_th)]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(dataframe=df)

# except target var
cat_cols = [col for col in cat_cols if 'Heart Attack Risk' not in col]

# check outliers for numeric cols
for var in num_cols:
    print("Variable -> " + str(var) + " : " + str(check_outlier(df, var)))


# if outliers exist use this one
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


for var in num_cols:
    grab_outliers(dataframe=df, col_name=var)  # empty

df[num_cols].describe([0.1, 0.2, 0.3, 0.4, .5, .6, .7, .8, .9, .99]).T

#  Target variable analysis

for var in cat_cols:
    print(df.groupby(var).agg({'Heart Attack Risk': ['mean', 'count']}))
    print('##########################################')

for var in num_cols:
    print(df.groupby('Heart Attack Risk').agg({var: ['mean', 'count']}))
    print('##########################################')


# cat summary
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for var in cat_cols:
    cat_summary(df, var)

# Missing value analysis
df.isnull().sum()
quick_look(df)

# Correlation analysis

# cat_but_num
cat_but_num = [col for col in cat_cols if df[col].dtype != 'O']
columns = cat_but_num + num_cols

correlation_matrix2 = df[num_cols].corr()
print(correlation_matrix2)

# using plt
correlation_matrix = df[columns].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

df.describe().T

############ Feature Engineering ############

# Heart Attack Risk
df.head()

# Binning
# AGE_QCUT
df['AGE_QCUT'] = pd.qcut(x=df['Age'], q=4)

# NEW_AGE_CAT
df['Age'].describe().T
df.loc[df['Age'] < 30, 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 30) & (df['Age'] <= 45), 'NEW_AGE_CAT'] = 'middle'
df.loc[(df['Age'] > 45) & (df['Age'] <= 65), 'NEW_AGE_CAT'] = 'senior_middle'
df.loc[(df['Age'] > 65) & (df['Age'] <= 80), 'NEW_AGE_CAT'] = 'old'
df.loc[(df['Age'] > 80), 'NEW_AGE_CAT'] = 'senior_old'

# NEW_SEX_CAT
df.loc[(df['Sex'] == 'Male') & (df['Age'] < 30), 'NEW_SEX_CAT'] = 'young_male'
df.loc[(df['Sex'] == 'Male') & (df['Age'] >= 30) & (df['Age'] <= 45), 'NEW_SEX_CAT'] = 'middle_male'
df.loc[(df['Sex'] == 'Male') & (df['Age'] > 45) & (df['Age'] <= 65), 'NEW_SEX_CAT'] = 'senior_middle_male'
df.loc[(df['Sex'] == 'Male') & (df['Age'] > 65) & (df['Age'] <= 80), 'NEW_SEX_CAT'] = 'old_male'
df.loc[(df['Sex'] == 'Male') & (df['Age'] > 80), 'NEW_SEX_CAT'] = 'senior_old_male'

df.loc[(df['Sex'] == 'Female') & (df['Age'] < 30), 'NEW_SEX_CAT'] = 'young_female'
df.loc[(df['Sex'] == 'Female') & (df['Age'] >= 30) & (df['Age'] <= 45), 'NEW_SEX_CAT'] = 'middle_female'
df.loc[(df['Sex'] == 'Female') & (df['Age'] > 45) & (df['Age'] <= 65), 'NEW_SEX_CAT'] = 'senior_middle_female'
df.loc[(df['Sex'] == 'Female') & (df['Age'] > 65) & (df['Age'] <= 80), 'NEW_SEX_CAT'] = 'old_female'
df.loc[(df['Sex'] == 'Female') & (df['Age'] > 80), 'NEW_SEX_CAT'] = 'senior_old_female'

# CHO_DIV_HR
df['CHO_DIV_HR'] = round(df['Cholesterol'] / df['Heart Rate'])
df.groupby('Heart Attack Risk').agg({'CHO_DIV_HR': 'mean'})

# EXER_CHO
df['EXER_CHO'] = df['Cholesterol'] * df['Exercise Hours Per Week']
df.groupby('Heart Attack Risk').agg({'EXER_CHO': 'mean'})

# AGE_MULTI_HR
df['AGE_MULTI_HR'] = df['Age'] * df['Heart Rate']
df.groupby('Heart Attack Risk').agg({'AGE_MULTI_HR': 'mean'})

# STRESS_MULTI_AGE
df.head()
df['STRESS_MULTI_AGE'] = df['Age'] * df['Stress Level']
df.groupby('Heart Attack Risk').agg({'STRESS_MULTI_AGE': 'mean'})

# INCOME_DIV_STRESS
df['INCOME_DIV_STRESS'] = df['Income'] / df['Stress Level']
df.groupby('Heart Attack Risk').agg({'INCOME_DIV_STRESS': 'mean'})

# MULTI_AGE_CHO_DIV_HR
df['MULTI_AGE_CHO_DIV_HR'] = (df['Age'] * df['Cholesterol']) / df['Heart Rate']
df.groupby('Heart Attack Risk').agg({'MULTI_AGE_CHO_DIV_HR': 'mean'})

# TRI_DIV_BMI
df['TRI_DIV_BMI'] = df['Triglycerides'] / df['BMI']
df.groupby('Heart Attack Risk').agg({'TRI_DIV_BMI': 'mean'})

# TRI_DIV_BMI_MULTI_AGE
df['TRI_DIV_BMI_MULTI_AGE'] = df['TRI_DIV_BMI'] * df['Age']
df.groupby('Heart Attack Risk').agg({'TRI_DIV_BMI_MULTI_AGE': 'mean'})

# BMI_MULTI_HR
df['BMI_MULTI_HR'] = df['BMI'] * df['Heart Rate']
df.groupby('Heart Attack Risk').agg({'BMI_MULTI_HR': 'mean'})

# $ BLOOD PRESSURE NEW
df[['BP_Systolic', 'BP_Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
df[['BP_Systolic', 'BP_Diastolic']] = df[['BP_Systolic', 'BP_Diastolic']].astype(int)
df.drop('Blood Pressure', axis=1, inplace=True)
df.head()
df['Blood Pressure'] = df['BP_Systolic'] / df['BP_Diastolic']

# BP_MULT_HR
df['BP_MULT_HR'] = df['Blood Pressure'] * df['Heart Rate']
df.groupby('Heart Attack Risk').agg({'BP_MULT_HR': 'mean'})

# SAMPLE1
df['AG_CH/HR__TRI/BMI'] = df['MULTI_AGE_CHO_DIV_HR'] * df['TRI_DIV_BMI']
df.groupby('Heart Attack Risk').agg({'AG_CH/HR__TRI/BMI': 'mean'})

# NUM_DIET
df['Diet'].unique()
df.loc[df['Diet'] == 'Unhealthy', 'NUM_DIET'] = 1
df.loc[df['Diet'] == 'Average', 'NUM_DIET'] = 2
df.loc[df['Diet'] == 'Healthy', 'NUM_DIET'] = 3

# SAMPLE2
df['AG_CH/ND_EH'] = (df['Age'] * df['Cholesterol']) / (df['NUM_DIET'] * df['Exercise Hours Per Week'])
df.groupby('Heart Attack Risk').agg({'AG_CH/ND_EH': 'mean'})

df.head()
df.dtypes

# Label Encoder : diet
# ordinal kat : Stress Level, Physical Activity Days Per Week, Sleep Hours Per Day

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "int32", "float64"]
               and df[col].nunique() == 2]

df.dtypes


def label_encoder(dataframe, binary_col):
    label_encoder = LabelEncoder()
    dataframe[binary_col] = label_encoder.fit_transform(dataframe[binary_col])
    return dataframe


for var in binary_cols:
    df = label_encoder(df, var)
df.head()

# One Hot Encoding
ohe_cols = [col for col in df.columns if 30 >= df[col].nunique() > 2]
ohe_cols = [col for col in ohe_cols if
            col not in ['Diet', 'Stress Level', 'Physical Activity Days Per Week', 'Sleep Hours Per Day']]
df.head(10)

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols = [col for col in cat_cols if 'Heart Attack Risk' not in col]

df.dtypes


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


# Diet is categoric but ordinal
label_encoder(df, 'Diet')

df = one_hot_encoder(df, ohe_cols)
df.head()

df = one_hot_encoder(df, cat_cols, True)

df = one_hot_encoder(df, cat_cols, True)
df.head()

binary_cols = [col for col in df.columns if df[col].dtype not in ["int64", "int32", "float64"]
               and df[col].nunique() <= 10]

for var in binary_cols:
    label_encoder(df, var)


# Rare Encoding
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols = [col for col in cat_cols if 'Heart Attack Risk' not in col]
rare_analyser(df, "Heart Attack Risk", cat_cols)

df.head()
country_columns = [col for col in df.columns if "Country_" in col]
rare_analyser(df, "Heart Attack Risk", country_columns)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

for var in df.columns:
    print(df[var].unique())

for var in df.columns:
    print(df[var].unique())

df.columns
df = rare_encoder(df, 0.02)
df.head()
#############################################
# Feature Scaling (Özellik Ölçeklendirme)
#############################################

# StandardScaler:
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols = [col for col in cat_cols if 'Heart Attack Risk' not in col]
df.head()
ss = StandardScaler()
for var in num_cols:
    df[var] = ss.fit_transform(df[[var]])

df.head()
num_cols
##########
# Model
##########
df.head()
y = df["Heart Attack Risk"]
X = df.drop(["Patient ID", "Heart Attack Risk"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

df.head()
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)


# Yeni ürettiğimiz değişkenler ne alemde?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)
