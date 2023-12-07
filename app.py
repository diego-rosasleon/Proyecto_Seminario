

# BIBLIOTECAS ##################################################################
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly as plt
import plotly.graph_objects as go
import exchange_calendars as xcals
from datetime import date

import yfinance as yf
import pandas as pd
import plotly as plt
from datetime import date
import numpy as np
from scipy.stats import norm, skew, kurtosis
from numpy.linalg import multi_dot
import warnings
import random
from scipy.stats import kurtosis
import plotly.express as px
warnings.filterwarnings('ignore')

# FUNCIONES ##################################################################

def download_data(tickers, start_date='2010-01-01', end_date=date.today().strftime('%Y-%m-%d')):
    data = yf.download(tickers, start=start_date, end=end_date)

    return data['Close']

def calcular_fechas(hoy: pd.Timestamp):
    # Obtén el calendario de la bolsa de México
    xmex = xcals.get_calendar("XMEX")

    # Si el día de la semana es lunes (0 en el sistema Python weekday()), retrocede 3 días
    if hoy.weekday() == 0:
        prev_business_day = hoy - pd.Timedelta(days=3)
    # De lo contrario, solo retrocede un día
    else:
        prev_business_day = hoy - pd.Timedelta(days=1)

    # Si el día calculado no es un día hábil, busca el día hábil más reciente
    if not xmex.is_session(prev_business_day):
        prev_business_day = xmex.previous_close(prev_business_day).to_pydatetime()

    ayer = prev_business_day

    # Crear un diccionario para almacenar los resultados
    resultado = {}

    primer_dia = pd.Timestamp("2021-01-01")

    # Calcula los días hábiles entre el primer día del mes y hoy
    dias_habiles = len(xmex.sessions_in_range(primer_dia, hoy))+1

    return dias_habiles


# MAIN ##################################################################

tickers = ['IVV', 'EWW', 'TLT', "USO", 'EMB', "MXN=X"]

activos=download_data(tickers)
activos = activos.dropna()

df_activos = activos.copy()
df_activos['IVV'] = df_activos['IVV'] * df_activos["MXN=X"]
df_activos['EWW'] = df_activos['EWW'] * df_activos["MXN=X"]
df_activos['TLT'] = df_activos['TLT'] * df_activos["MXN=X"]
df_activos['USO'] = df_activos['USO'] * df_activos["MXN=X"]
df_activos['EMB'] = df_activos['EMB'] * df_activos["MXN=X"]
df_activos=df_activos.drop("MXN=X",axis=1)

returns = df_activos.pct_change().dropna()




# Opciones de navegación
st.sidebar.title("Navegación")
option = st.sidebar.radio("Seleccione una página", ["Activos", "Portafolios"])


if option == "Activos":
    st.title("Resumen y Estadisticas del activo")

    activo = st.sidebar.selectbox(
        "Elige un activo",
        ('IVV', 'EWW', 'TLT', "USO", 'EMB')
    )
    df_activo = df_activos[activo]

    # Descripción del activo
    st.header(activo)
    st.subheader('Descripción')

    if activo == 'IVV':
      st.write('''Este activo corresponde al ramo de renta variable desarrollada. Dicho ETF busca replicar el comportamiento del índice S&P 500, en dicho ETF lo que buscamos es tener exposición al mercado de renta variable de Estados Unidos.
      Dentro de este ETF el sector que predomina es el de las Tecnologías de la Información con un 28.63% por contener empresas como lo son APPLE o NVIDIA, seguido por el sector financiero con un 13% al poseer posiciones en empresas como JPMORGAN o VISA. El precio proporcionado se encuentra en USD. ''')
      st.write('''
      Al estar replicando el índice por excelencia del mercado estadounidense, tendremos una beta del portafolio de 1.
      Mientras que nuestra desviación se ubicaría en 17.8% (Tomando en cuenta datos desde 3 años atrás y hasta el día de hoy). Debido a la gran diversidad de empresas contenidas en este ETF no es posible asignarle alguna clasificación entre growth o value por estar expuestos a ambos tipos de empresas.
      Los costos de administración de dicho ETF son de 0.03% lo que se considera dentro de los valores comunes del mercado.''')
      st.write('''
      Grado de inversión: A''')
      st.write('''
      inferimos que tiene un estilo tipo Growth porque tiene empresas con mayor capitalización, apostandole a su crecimiento''')


    elif activo == 'EWW':
      st.write('''Con este ETF lo que buscamos es tener una exposición al mercado de renta variable
      mexicano. El valor que nos proporciona ya se encuentra en dólares. Dentro de dicho
      instrumento se encuentran empresas en diversos ámbitos como lo son FEMSA, WALMEX
      o BIMBO en el caso del sector de productos básicos; de igual forma podremos estar
      expuestos a fibras y al sector financiero.''')
      st.write(''' Al tener una amplia selección de empresas, la Beta obtenida es de 1.02 que nos indica
      que el valor de este instrumento se mueve un poco más que el mercado por un 2%.
      Mientras que su desviación estándar se sitúa en 24.86%. Dentro de este ETF, las
      empresas que representan una mayor proporción son las que se consideran como del tipo
      Value. Es importante recalcar que los gastos de adquirir este ETF son mayores que el
      promedio del mercado pues tiene una comisión del 0.5%. ''')
      st.write('''
      Grado de inversión: LTL''')

    elif activo == 'TLT':
      st.write('''Este ETF busca replicar los resultados de inversión de un índice compuesto por bonos del
      tesoro estadounidenses con vencimientos residuales entre 10 y 20 años. Obviamente
      nuestra divisa base para dicho ETF son dólares.''')
      st.write('''
      Dado que son bonos con un vencimiento sumamente largo, nuestra beta será de apenas
      0.47. Nuestra desviación estándar es de 14.66% tomando datos hasta el día 31 de oct del
      y desde 3 años anteriores a dicha fecha. Los costos por adquirir este ETF son de
      0.15 %. Esta inversión se considera conservadora al invertir en activos que se consideran
      "seguros" o al menos, son menos riesgosos que los que entran en la clasificación de renta
      variable.''')

    elif activo == 'USO':
      st.write('''Este ETF busca reflejar el rendimiento diario de los contratos de futuros del petróleo crudo
      West Texas Intermediate (WTI). Es uno de los ETFs más populares para obtener
      exposición directa al precio del petróleo. Dicho ETF se encuentra en la divisa de origen
      que son los dólares. ''')
      st.write('''
      Este instrumento presenta una beta de 1.4, además de poseer una desviación estándar
      de 29.88% (Ambos datos fueron calculados con una ventana de tiempo de 3 años).
      Mientras que los gastos para adquirir este ETF se sitúan en 0.8605%. Este tipo de activos
      son sumamente recurridos en épocas de tensiones políticas o durante las guerras entre
      países.''')

    elif activo == 'EMB':
      st.write('''Este ETF en particular replica el comportamiento de los bonos gubernamentales emitidos
      por países de mercados emergentes. Está tasado en dólares, con la ayuda de este
      instrumento podemos estar expuestos a los mercaos turcos, brasileño, filipino, mexicano
      entre otros.''')
      st.write('''
      Este instrumento financiero posee una desviación estándar de 10.83% en una ventana de
      tiempo de 3 años contados hasta el 31 de octubre del 2023. Respecto a la beta, esta le
      corresponde un valor de 0.5. La comisión por comercializar con este ETF es de 0.39%.''')




    # Crear la figura
    fig = go.Figure()

    # Agregar los datos del activo a la figura
    fig.add_trace(go.Scatter(x=df_activo.index, y=df_activo.values, mode='lines'))

    # Establecer títulos y etiquetas
    fig.update_layout(title='Precio de cierre historico del activo',
                    xaxis_title='Fecha',
                    yaxis_title='Precio de Cierre (en $)')

    st.plotly_chart(fig)

    st.text("Pon la fecha a la que quieres los rendimientos:")

    hoy = st.date_input('Introduce la fecha')
    hoy = pd.Timestamp(hoy)
    dias = calcular_fechas(hoy)

    st.write("El numero de dia es:",dias)

    st.write("A continuación mostramos un dataframe de nuestro activo.")
    df_activos[activo]

    st.subheader("Rendimientos históricos del activo")

    returns[activo]

    st.subheader("Métricas de riesgo")

    mean = np.mean(returns)

    st.write('La media de retornos de', activo, ' es de:',mean[activo])

    sesgo=skew(returns[activo])

    st.write('El sesgo del activo financiero ', activo,' es: ',sesgo)

    curtosis = kurtosis(returns[activo])
    # Calcular el exceso de curtosis restando 3 (la curtosis de la distribución normal estándar es 3)
    exceso_curtosis = curtosis - 3
    st.write('El exceso de curtosis del ETF denominado', activo,' es de',exceso_curtosis)

    stdev = np.std(returns)
    st.write('La desviación estándar de retornos de', activo, ' es de:',stdev[activo])
    st.write('Este último valor nos exlica que tan alejados en promedio están nuestros datos de la media dada con anterioridad.')

    st.write('Después, tomando estos valores como base procederemos a calcular uno de las métrcias más comunes dentro de los portafolios de inversión que es el Var.En este caso Paramétrico considerando una distribución Normal.')
    VaRPar_95=norm.ppf(1-0.95,mean[activo],stdev[activo])
    st.write("El VaR Paramétrico al 95% de", activo, ' es: ', round(VaRPar_95*100,4))

    st.write('Procedemos a calcular el VaR Histórico, este solamente considera los valores de retornos que ha tenido a lo largo del tiempo nuestros ETF´s.')
    hVaR_95 = returns[activo].quantile(0.05)
    st.write("El VaR Histórico al 95% de ",activo,' es: ',  round(hVaR_95*100,4))

    st.write('Continuamos con el VaR Montecarlo este simiula de manera aleatoria cierta cantidad de rendimientos, después dados esos rendimientos procederemos a obtener el percenteil que deseamos, este caso seria el 5 para que tengamos una confianza del 95%')
    # Number of simulations
    n_sims = 100000
    # Simulate returns and sort
    sim_returns = np.random.normal(mean[activo], stdev[activo], n_sims)
    MCVaR_95 = np.percentile(sim_returns, 5)
    st.write("El VaR Montecarlo al 95% de ",activo, round(MCVaR_95*100,4))

