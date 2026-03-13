import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import time
import re

load_dotenv()

# Constantes
HF_TOKEN = os.getenv("HF_TOKEN")

# Lista de modelos para tentar em ordem (fallback automático)
MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta"
]

API_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ============================================
# FUNÇÃO PARA REMOVER ASTERISCOS
# ============================================

def remover_asteriscos(texto: str) -> str:
    """
    Remove asteriscos do texto e formata corretamente
    """
    if not texto:
        return texto
    
    # Remove ** (negrito markdown) mas mantém o conteúdo
    texto = re.sub(r'\*\*(.*?)\*\*', r'\1', texto)
    
    # Remove * (itálico markdown) mas mantém o conteúdo
    texto = re.sub(r'\*(.*?)\*', r'\1', texto)
    
    # Remove asteriscos solitários
    texto = texto.replace('*', '')
    
    return texto


def formatar_resposta(texto: str, primeira_interacao: bool = False) -> str:
    """
    Formata a resposta final da IARA
    """
    # Remover asteriscos
    texto = remover_asteriscos(texto)
    
    # Garantir que não haja markdown
    texto = texto.replace('#', '').replace('_', '')
    
    # Se for primeira interação, garantir apresentação
    if primeira_interacao and "IARA" not in texto and "Olá" not in texto:
        texto = f"Olá! Sou a IARA. {texto}"
    
    return texto.strip()


# ============================================
# FUNÇÕES PARA ANALISAR CSV
# ============================================

def analisar_csv(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analisa o DataFrame do usuário e extrai informações relevantes
    """
    if df is None or df.empty:
        return {"erro": "CSV vazio ou não disponível"}
    
    analise = {
        "total_linhas": len(df),
        "colunas": list(df.columns),
        "tipos": df.dtypes.astype(str).to_dict(),
        "estatisticas": {},
        "resumo": df.describe().to_dict() if len(df) > 0 else {},
        "primeiras_linhas": df.head(10).fillna('').to_dict(orient='records')
    }
    
    # Análise de valores se existir coluna de valor
    if 'valor' in df.columns:
        analise['estatisticas']['valor'] = {
            'total': float(df['valor'].sum()),
            'media': float(df['valor'].mean()),
            'minimo': float(df['valor'].min()),
            'maximo': float(df['valor'].max())
        }
    
    # Agrupamento por categoria se disponível
    if 'categoria' in df.columns and 'valor' in df.columns:
        analise['estatisticas']['por_categoria'] = df.groupby('categoria')['valor'].sum().to_dict()
    
    # Análise temporal se existir coluna de data
    if 'data' in df.columns:
        try:
            df['data'] = pd.to_datetime(df['data'])
            analise['estatisticas']['por_mes'] = df.groupby(df['data'].dt.to_period('M'))['valor'].sum().astype(float).to_dict()
        except:
            pass
    
    return analise


def criar_contexto_csv(df: pd.DataFrame, nome: str = None) -> str:
    """
    Cria um contexto em texto com os dados do CSV
    """
    if df is None or df.empty:
        return "Nenhum dado CSV disponível."
    
    contexto = []
    
    if nome:
        contexto.append(f"Usuário: {nome}")
    
    contexto.append(f"Total de transações: {len(df)}")
    contexto.append(f"Colunas disponíveis: {', '.join(df.columns)}")
    
    # Estatísticas básicas
    if 'valor' in df.columns:
        total = df['valor'].sum()
        media = df['valor'].mean()
        contexto.append(f"Total gasto: R$ {total:,.2f}".replace(',', 'v').replace('.', ',').replace('v', '.'))
        contexto.append(f"Média por transação: R$ {media:,.2f}".replace(',', 'v').replace('.', ',').replace('v', '.'))
    
    # Top categorias
    if 'categoria' in df.columns and 'valor' in df.columns:
        top_categorias = df.groupby('categoria')['valor'].sum().nlargest(3)
        contexto.append("\nPrincipais categorias de gasto:")
        for cat, val in top_categorias.items():
            contexto.append(f"  • {cat}: R$ {val:,.2f}".replace(',', 'v').replace('.', ',').replace('v', '.'))
    
    # Últimas transações
    if len(df) > 0:
        contexto.append("\nÚltimas transações:")
        for i, row in df.head(5).iterrows():
            linha = []
            if 'data' in row:
                linha.append(str(row['data']))
            if 'descricao' in row:
                linha.append(row['descricao'])
            if 'categoria' in row:
                linha.append(f"[{row['categoria']}]")
            if 'valor' in row:
                linha.append(f"R$ {row['valor']:,.2f}".replace(',', 'v').replace('.', ',').replace('v', '.'))
            
            contexto.append(f"  • {' '.join(linha)}")
    
    return "\n".join(contexto)


def testar_modelo(modelo: str, contexto: str, pergunta: str, primeira_vez: bool = False) -> Optional[str]:
    """Testa um modelo específico e retorna resposta ou None"""
    
    print(f"🔄 Testando modelo: {modelo}")
    
    # Prompt do sistema melhorado para evitar asteriscos
    system_prompt = """Você é IARA, uma assistente financeira pessoal e amigável.

REGRAS IMPORTANTES:
1. NÃO use asteriscos (*) ou markdown no texto
2. Use apenas texto simples e emojis quando apropriado
3. Responda com base APENAS nos dados do CSV fornecidos
4. Seja clara, direta e educada
5. Explique os números de forma simples
6. Se não souber algo, diga que não tem informação
7. Só se apresente na primeira interação"""

    user_prompt = f"""CONTEXTO (dados do CSV do usuário):
{contexto}

PERGUNTA DO USUÁRIO: {pergunta}

Responda de forma útil e personalizada (sem usar asteriscos):"""

    payload = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 600,
        "temperature": 0.7
    }
    
    try:
        print(f"📡 Enviando requisição...")
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                resposta = data["choices"][0]["message"]["content"].strip()
                # Remover qualquer asterisco que ainda possa vir
                resposta = remover_asteriscos(resposta)
                print(f"✅ Resposta recebida de {modelo}")
                return resposta
            else:
                print(f"⚠️ Resposta sem 'choices': {data}")
                return None
        
        print(f"⚠️ Modelo {modelo} falhou: {response.status_code}")
        print(f"📄 Resposta: {response.text[:200]}")
        return None
        
    except Exception as e:
        print(f"⚠️ Erro com modelo {modelo}: {e}")
        return None


# ============================================
# FUNÇÃO PRINCIPAL - APENAS COM CSV
# ============================================

def perguntar_iara(pergunta: str, nome: str = None, csv_df: pd.DataFrame = None) -> str:
    """
    Função principal que recebe:
    - pergunta: texto do usuário
    - nome: nome do usuário (opcional)
    - csv_df: DataFrame com os dados do CSV enviado
    
    Retorna resposta da IARA baseada APENAS no CSV
    """
    try:
        print("\n" + "="*60)
        print(f"🤔 Pergunta: {pergunta}")
        if nome:
            print(f"👤 Usuário: {nome}")
        if csv_df is not None:
            print(f"📊 CSV carregado: {len(csv_df)} linhas")
        print("="*60)
        
        # Verificar token
        if not HF_TOKEN:
            return "❌ Erro: Token da Hugging Face não configurado. Verifique seu arquivo .env"
        
        # Verificar se tem CSV
        if csv_df is None or csv_df.empty:
            return """Olá! 👋

Antes de começar, preciso que você envie seu arquivo CSV com as transações financeiras.

O CSV deve conter colunas como:
• data
• descrição
• categoria
• valor

Use o botão 📎 para enviar o arquivo."""
        
        # Criar contexto apenas com o CSV
        contexto = criar_contexto_csv(csv_df, nome)
        print(f"📄 Contexto criado com base no CSV")
        
        # Verificar se é primeira interação (para apresentação)
        primeira_vez = (nome is not None and len(pergunta) < 20)
        
        # Tentar modelos em ordem
        for i, modelo in enumerate(MODELS):
            print(f"\n🔄 Tentativa {i+1}/{len(MODELS)}")
            
            resposta = testar_modelo(modelo, contexto, pergunta, primeira_vez)
            
            if resposta:
                print(f"✅ Sucesso com modelo: {modelo}")
                # Garantir que a resposta não tem asteriscos
                resposta = remover_asteriscos(resposta)
                return resposta
            
            time.sleep(1)
        
        return """
Desculpe, estou temporariamente indisponível. Por favor, tente novamente em alguns minutos.
"""
        
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        return f"Erro interno: {str(e)}"


# ============================================
# FUNÇÃO PARA PROCESSAR UPLOAD (chamada pelo main)
# ============================================

def processar_upload_csv(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Processa o CSV e retorna uma análise para o frontend
    """
    return analisar_csv(df)


# Teste rápido se executado diretamente
if __name__ == "__main__":
    print("🧪 Testando IARA modo CSV...")
    # Criar DataFrame de exemplo
    df_teste = pd.DataFrame({
        'data': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'descricao': ['Supermercado', 'Uber', 'Restaurante'],
        'categoria': ['alimentação', 'transporte', 'alimentação'],
        'valor': [350.00, 45.00, 120.00]
    })
    
    resposta = perguntar_iara(
        "Quanto gastei em alimentação?",
        nome="Teste",
        csv_df=df_teste
    )
    print(f"\n💬 {resposta}")