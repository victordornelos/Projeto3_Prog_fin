import yfinance as yf
import pandas as pd
import numpy as np
from tabulate import tabulate

## Carteira brasileira

acoes_br = ['BBDC4.SA','CMIG4.SA','PETR4.SA','GOAU4.SA','MRFG3.SA']

# Baixar os dados

df_br = yf.download(acoes_br,auto_adjust=True,start='2014-01-01',end='2024-12-31')['Close']



# Salvando em CSV

df_br.to_csv('carteira_brasileira.csv')



# Importando os dados

df_br = pd.read_csv('carteira_brasileira.csv', index_col='Date', parse_dates=True)

pesos_br = np.array([0.26, 0.21, 0.20, 0.19,0.14])







# Retornos logarítmicos diários

retornos_diarios = np.log(df_br / df_br.shift(1)).dropna()



# Indicadores individuais

retorno_anual = retornos_diarios.mean() * 252

retorno_acumulado = (df_br.iloc[-1] / df_br.iloc[0]) - 1

desvio_padrao_anual = retornos_diarios.std() * np.sqrt(252)

coef_variacao = desvio_padrao_anual / retorno_anual





# Tabela por ativo

tabela_ativos = pd.DataFrame({

'Retorno Anual (%)': retorno_anual * 100,

'Retorno Acumulado (%)': retorno_acumulado * 100,

'Desvio Padrão Anual (%)': desvio_padrao_anual * 100,

'Coef. de Variação': coef_variacao

})





# Retorno do portfólio

retorno_portfolio_diario = retornos_diarios @ pesos_br

retorno_anual_ptf = retorno_portfolio_diario.mean() * 252

retorno_acumulado_ptf = np.exp(retorno_portfolio_diario.cumsum())[-1] - 1

desvio_padrao_ptf = retorno_portfolio_diario.std() * np.sqrt(252)

coef_variacao_ptf = desvio_padrao_ptf / retorno_anual_ptf





# Adiciona à tabela

tabela_ativos.loc['Carteira'] = [

retorno_anual_ptf * 100,

retorno_acumulado_ptf * 100,

desvio_padrao_ptf * 100,

coef_variacao_ptf

]



# Arredondar para melhor visualização

tabela_ativos = tabela_ativos.round(2)



# Taxa livre de risco (ajuste conforme o país e período)

r_f = 0.0945 # CDI Médio de 2014 a 2024



# Sharpe dos ativos

sharpe_ativos = (retorno_anual - r_f) / desvio_padrao_anual



# Sharpe da carteira

sharpe_portfolio = (retorno_anual_ptf - r_f) / desvio_padrao_ptf



# Adiciona à tabela

tabela_ativos['Sharpe Ratio'] = sharpe_ativos

tabela_ativos.loc['Carteira', 'Sharpe Ratio'] = sharpe_portfolio



# Exibe com tabulate

from tabulate import tabulate

print("\n📊 Indicadores com Sharpe Ratio:\n")

print(tabulate(tabela_ativos.round(2), headers='keys', tablefmt='fancy_grid'))
