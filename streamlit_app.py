import pandas as pd
import numpy as np
import pandas_datareader as web
import datetime as dt
 
 
start = dt.datetime(2020,1,1)
end = dt.datetime.now()
 
df = web.DataReader('XRP-USD', 'yahoo', start, end)
df
