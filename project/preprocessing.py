# Preprocessing module
import pandas as pd
import re
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


def binary(candidate):
        return bool(re.match(r"^[01]$", candidate))  # 1.0 ?? 

def is_binary(df, var):
    # if the percent of binaries is higher than threshold then 
    # classify as binary
    # .count() count only no-null values.
    percent = df[var].apply(lambda x: binary(str(x))).sum() / df[var].count()
    if percent == 1:
        return True
    else:
        return False
        
    
def binary_df(df):
    binaries = []
    for var in  df.columns:
        if is_binary(df, var):
            binaries.append(var)
    return binaries


def breakdown_vars(df, off_binary=False):
    """
    This function allow us categorize accodign to numerical or not
    """
    binaries = binary_df(df)
    categorial = []
    nonormal = []
    normal = []
    for t in df.columns:
        if off_binary == False:
          if (df[t].dtypes.name=="object" or df[t].dtypes.name=='category') and  t not in binaries:
            categorial.append(t)
        else:
           if (df[t].dtypes.name=="object" or df[t].dtypes.name=='category'):
            categorial.append(t)
        if (df[t].dtypes=="int64" or df[t].dtypes=="float64") and t not in binaries:
                n,p = stats.shapiro(df[t])
                if p<0.05:
                    nonormal.append(t)
                else:
                    normal.append(t)
    if off_binary == False:
      return categorial, binaries, nonormal, normal
    else:
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
    binaries = binary_df(df)
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



def standardize_X_test(X_train, X_test):
    X_test_ = X_test.copy()
    cats, binaries, nonormal, normal  = breakdown_vars(X_train)
    locations_scales = {}
    for var in normal + nonormal:
        locations_scales[var] = [X_train[var].mean(), X_train[var].std()]
    for var in locations_scales:
        print(var)
        X_test_[var] = (X_test_[var] - locations_scales[var][0])/locations_scales[var][1]
    return X_test_
