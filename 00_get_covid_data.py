#%%
import pandas as pd
from tabula import read_pdf
import requests

#%%
num = 49
url = 'D:\\Data\\covid_42.pdf'
doc = read_pdf(url)
doc[1]

#%%
print(doc)

# %%
webs = range(1,50)

for w in webs: 
    name = 'covid_'+str(w)+'.pdf'
    doc = requests.get('https://www.mscbs.gob.es/profesionales/saludPublica/ccayes/alertasActual/nCov-China/documentos/Actualizacion_'+str(w)+'_COVID-19.pdf', verify=False)
    with open(r"D:\\Data\\" + name, 'wb') as f: 
        f.write(doc.content)

# %%
webs = range(1,50)

for d in docs: 
    name = 'covid_'+str(w)+'.pdf'
    doc = requests.get('https://www.mscbs.gob.es/profesionales/saludPublica/ccayes/alertasActual/nCov-China/documentos/Actualizacion_'+str(w)+'_COVID-19.pdf', verify=False)
    with open(r"D:\\Data\\" + name, 'wb') as f: 
        f.write(doc.content)