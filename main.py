from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from llm_service import perguntar_iara, processar_upload_csv
import pandas as pd
import io
from typing import Optional

app = FastAPI(title="IARA - API", description="Assistente Financeira Inteligente")

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Armazenar CSV em memória (simples para este caso)
# Em produção, use um banco de dados ou cache
dados_csv = None
nome_usuario = None

class ChatRequest(BaseModel):
    pergunta: str
    nome: Optional[str] = None

class ChatResponse(BaseModel):
    resposta: str

class UploadResponse(BaseModel):
    mensagem: str
    linhas: int
    colunas: list[str]
    preview: list[dict]

@app.get("/")
def home():
    return {
        "status": "online",
        "api": "IARA",
        "versao": "2.0 - Modo CSV"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload-csv", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    """
    Endpoint para upload do CSV do usuário
    """
    global dados_csv
    
    try:
        # Validar extensão
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Arquivo deve ser CSV")
        
        # Ler o CSV
        contents = await file.read()
        
        # Tentar diferentes codificações
        try:
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        except UnicodeDecodeError:
            df = pd.read_csv(io.StringIO(contents.decode('latin-1')))
        
        # Validar colunas mínimas
        colunas_necessarias = ['data', 'descricao', 'categoria', 'valor']
        colunas_presentes = [col for col in colunas_necessarias if col in df.columns]
        
        if len(colunas_presentes) < 2:
            raise HTTPException(
                status_code=400, 
                detail="CSV deve conter pelo menos 2 das colunas: data, descricao, categoria, valor"
            )
        
        # Salvar na memória
        dados_csv = df
        
        # Preparar preview
        preview = df.head(5).fillna('').to_dict(orient='records')
        
        return UploadResponse(
            mensagem="CSV carregado com sucesso!",
            linhas=len(df),
            colunas=list(df.columns),
            preview=preview
        )
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Arquivo CSV está vazio")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar CSV: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Endpoint para conversar com a IARA usando o CSV enviado
    """
    global dados_csv, nome_usuario
    
    try:
        # Atualizar nome se fornecido
        if req.nome:
            nome_usuario = req.nome
        
        # Chamar IARA com o CSV
        resposta = perguntar_iara(
            pergunta=req.pergunta,
            nome=nome_usuario,
            csv_df=dados_csv
        )
        
        return ChatResponse(resposta=resposta)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/csv-info")
def csv_info():
    """
    Retorna informações sobre o CSV carregado
    """
    global dados_csv
    
    if dados_csv is None:
        return {"carregado": False}
    
    return {
        "carregado": True,
        "linhas": len(dados_csv),
        "colunas": list(dados_csv.columns),
        "tipos": dados_csv.dtypes.astype(str).to_dict()
    }

@app.post("/reset")
def reset():
    """
    Reseta os dados (CSV e nome)
    """
    global dados_csv, nome_usuario
    dados_csv = None
    nome_usuario = None
    return {"mensagem": "Dados resetados com sucesso"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)