# Preprocessing module
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder

def complete_vars(df):
    """
    percentage of columns with information
    """
    df_ = df.copy()
    N_cols = df_.shape[1]
    return 1 - df_.apply(lambda x: pd.isnull(x)).sum(axis=1)/N_cols




def ratios(df_,):
    # http://www.fce.unal.edu.co/media/files/UAMF/Anlisis_de_riesgo_de_crdito_e_indicadores_financieros.pdf
    df = df_.copy()
  # Nota capital de trabajo es activo corriente - pasivo corriente..
    # -------------------------------------------*(Debt equity ratio)
    df['DER'] = df['Total pasivos'] / df['Patrimonio total']
   ### Second ratios added....
   #Razón corriente or ----------------------------*(Current ratio)
    df['CR'] = df['Activos corrientes totales'] / df['Pasivos corrientes totales']
  # Marge de ganancia bruta (MGB)---------------------*(Gross Profit Margin)
  # (Gross Profit Margin )  = Gross Profit / Operating Revenues  
  # Ganancia bruta / ingresos de actividades ordianrias
    df['GPM'] = df['Ganancia bruta'] / df['Ingresos de actividades ordinarias']
  # Remember that ganancia bruta is:
    df['Ingresos de actividades ordinarias'] - df['Costo de ventas']
  ## Margen de ganancia Neta (MGN) ------------------- *(Net Profit Margin)
  #df['Ganancia (pérdida) antes de impuestos'] - df['Ingreso (gasto) por impuestos'] #esto es el resultado del periodo
  #df[ 'Ganancia (pérdida)'] # que en la base resultado del periodo es esta variable
  # (Net Profit Margin) = Profit(Loss) / Operating Revenues  
    df['NPM'] = df[ 'Ganancia (pérdida)'] /  df['Ingresos de actividades ordinarias']
  # Rendimiento del patrimonio ---- Return on Equity (ROE) 
  # (ROE) = Profit(Loss) / Total equity
    df['ROE'] = df['Ganancia (pérdida)']/ df['Patrimonio total']
  # Rendimiento del Activo (ROA)
  # (ROA) = Profit(Loss) / Total Assets
    df['ROA'] = df['Ganancia (pérdida)']/ df['Total de activos']
  ## NIVEL DE ENDEUDAMIENTO (NE)-----------*(Indebtedness ratio)
  # (Indebtedness Ratio) = Total Liabilities / Total Assets
    df['IR'] = df['Total pasivos'] / df['Total de activos']
  ## Concenttración de Pasivos a corto plazo (PCP)  ----------- *(ratio of Short-Term Liabilities)
  # Total current labilitiies / Total liabilitites
    df['RSL'] = df['Pasivos corrientes totales'] / df['Total pasivos']
  ## Endeudamiento Financiero (EF)
  ## Crear primero otros pasivos financieros
   # df['Otros pasivos financieros'] = df['Otros pasivos financieros corrientes'] + df['Otros activos no financieros corrientes']
    #df['EF'] = df['Otros pasivos financieros'] /  df['Ingresos de actividades ordinarias']
  ## Impacto de la carga financiera (IF)
    #df['IF'] = df['Costos financieros']/ df['Ingresos de actividades ordinarias']
  ## ALTMAN
    # AX1 = total current assets  - Total current liabilities / Total Asssets
    df['Ax1'] = (df['Activos corrientes totales'] - df['Pasivos corrientes totales'])/df['Total de activos']
    # Ax3 = Accumulated Profits / Total Assets
    df['Ax2'] = (df['Ganancias acumuladas'] / df['Total de activos'])
    return df


def breakdown_vars(df):
    """
    This function allow us categorize accodign to numerical or not
    """
    categorial = []
    nonormal = []
    normal = []
    for t in df.columns:
        if df[t].dtypes=="object" or df[t].dtypes.name=='category':
            categorial.append(t)
        if df[t].dtypes=="int64" or df[t].dtypes=="float64":
                n,p = stats.shapiro(df[t])
                if p<0.05:
                    nonormal.append(t)
                else: 
                    normal.append(t)
    return categorial, nonormal, normal



def dummies_ohe(df_,cats):
    """
    Returns a dataframe with dummies,and dropped the categorical in original
    the cats arguments receive the cats to transform.
    """
    df = df_.copy()
    df.reset_index(drop=True, inplace=True)
    ohe = OneHotEncoder(drop='first',handle_unknown='ignore', sparse_output=False)
    dummies = pd.DataFrame(ohe.fit_transform(df[cats]))
    dummies.columns = ohe.get_feature_names_out()  #Names ohe.get_feature_names_out()-> all dummies
    df.drop(columns=cats, inplace=True)
    df = pd.concat([df,dummies], axis=1)
    return df



def std_z(nums, df_, event=None):
    """
    standardizing nums(numerical) variables
    """
    df = df_.copy()
    binaries = is_binary(df, nums)
    for col in nums:
        if col not in binaries:
            df[col] = (df[col] - df[col].mean())/df[col].std()
    return df


def Xy(df_,target):
    """
    Split the data in X,y to ML implementations
    """
    df = df_.copy()
    X = df.loc[ : , df.columns != target]
    y = df[target]
    return X,y


def is_binary(df_, nums):
    df = df_.copy()
    variables = []
    for var in nums:
        flag = True
        unique = df_[var].unique()
        for value in unique:
            if value not in [0, 1, np.nan, 0.0, 1.0]:
                flag = False
        if flag == True:
            variables.append(var)
    return variables