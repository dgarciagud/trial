from datetime import date, timedelta
from duka.app.app import app as duka_app
from duka.core.utils import TimeFrame

# Rango: últimos 12 meses (evitamos hoy por si aún no hay barra cerrada)
end = date.today() - timedelta(days=1)
start = end - timedelta(days=365)

# Descarga D1 para EURUSD y USDJPY, con cabecera, en carpeta ./data
duka_app(
    symbols=['EURUSD', 'USDJPY'],
    start=start,
    end=end,
    threads=8,                  # ajusta según tu conexión/CPU
    timeframe=TimeFrame.D1,     # vela diaria
    folder='data',
    header=True                 # para que el CSV tenga cabecera
)
print(f"Listo: {start} → {end}")
