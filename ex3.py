# Import required libraries
import pandas as pd
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt

#Loading datas
df = pd.read_csv("/content/VariableExplicative1.csv", ";")
df.columns

#pre process data , taking only the EMU ones
ls_target = ['prices_EMU Equity Mid-Cap','prices_EMU Equity Mix-Cap','prices_EMU Equity Small-Cap','prices_EMU Equity Small-Cap Growth','prices_EMU Equity Large-Mid-Cap Growth','prices_EMU Equity Large-Mid-Cap Quality','prices_EMU Equity Large-Mid-Cap Multi Factors','prices_EMU Equity Large-Mid-Cap Value','prices_EMU Equity Large-Mid-Cap Low volatility','prices_EMU Equity Large-Mid-Cap Income'
]
df_target = df[ls_target]
df_target.columns
df_target.dropna(inplace=True)
df_target.head()

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
chi_square_value,p_value=calculate_bartlett_sphericity(df_target)
chi_square_value, p_value
 #(35481.98420769405, 0.0)
 #p value is 0 so the test is significant

from factor_analyzer.factor_analyzer import calculate_kmo
kmo_all,kmo_model=calculate_kmo(df_target)
kmo_model
#the overall KMO is 0.75 which is really good

# Create factor analysis object and perform factor analysis

fa = FactorAnalyzer()
fa.fit(df_target)

# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev
#[5.02713192, 1.82720529, 1.42613782, 0.55516132, 0.48810998,0.28005469, 0.14905855, 0.10471176, 0.07905187, 0.06337678]
#only 3 factors eigen values are greater than one. It means that we need to choose only 3 factors

#we do the factor analysis with 3 factors
fa.set_params(n_factors=3, rotation='varimax')
fa.fit(df_target)
fa.loadings_
fa.get_factor_variance()
total of 75% of the variance explained by the 3 variables
