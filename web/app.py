from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Configurar la carpeta de archivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configurar Jinja2Templates
templates = Jinja2Templates(directory="templates")

# Rutas para cada sección
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    
@app.get("/readme", response_class=HTMLResponse)
async def read_readme(request: Request):
    return templates.TemplateResponse("readme.html", {"request": request})

@app.get("/explanation", response_class=HTMLResponse)
async def read_explanation(request: Request):
    return templates.TemplateResponse("explanation.html", {"request": request})

@app.get("/performance", response_class=HTMLResponse)
async def read_performance(request: Request):
    return templates.TemplateResponse("performance.html", {"request": request})

@app.get("/prediction", response_class=HTMLResponse)
async def read_prediction(request: Request):
    return templates.TemplateResponse("prediction.html", {"request": request})
@app.get("/solución", response_class=HTMLResponse)
async def read_prediction(request: Request):
    return templates.TemplateResponse("solución.html", {"request": request})
