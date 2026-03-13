
# 📄 README.md — Projeto IARA


# 🤖 IARA - Inteligência Artificial para Análise de Renda e Atividades
<img width="1490" height="668" alt="image" src="https://github.com/user-attachments/assets/dde0b652-9af5-4862-991a-5a4b24106e02" />


IARA é uma assistente financeira inteligente capaz de analisar transações financeiras a partir de arquivos CSV e responder perguntas sobre gastos, categorias e hábitos financeiros através de um chat interativo.

O projeto combina **Inteligência Artificial, análise de dados e uma interface de chat moderna**.

---

# 🚀 Funcionalidades

✔ Chat interativo com IA  
✔ Upload de arquivo CSV de transações financeiras  
✔ Análise automática de gastos  
✔ Respostas inteligentes sobre finanças pessoais  
✔ Interface moderna estilo chat  
✔ Identificação do usuário pelo nome  

Exemplos de perguntas que o usuário pode fazer:

- Quanto gastei por categoria?
- Qual foi meu maior gasto?
- Como posso economizar?
- Qual foi meu gasto total no mês?

---

# 🧠 Como funciona

Fluxo da aplicação:

1. Usuário acessa o chat
2. A IA se apresenta e pergunta o nome do usuário
3. O usuário envia um arquivo CSV de transações
4. O backend processa os dados
5. O usuário pode fazer perguntas sobre seus gastos

---

# 📊 Formato do CSV esperado

O arquivo deve conter colunas como:

```

data,descricao,categoria,valor
2024-01-01,Supermercado,Alimentação,-150
2024-01-02,Salário,Renda,3500
2024-01-03,Uber,Transporte,-25

```

---

# 🏗 Arquitetura do Projeto

```

IARA-AGENT
│
├── iara-backend
│   ├── main.py
│   ├── llm_service.py
│   └── requirements.txt
│
└── iara-frontend
├── src
│   ├── components
│   │   ├── Chat.tsx
│   │   ├── Input.tsx
│   │   └── Message.tsx
│   │
│   ├── services
│   │   └── api.ts
│   │
│   ├── types
│   │   └── message.ts
│   │
│   └── assets
│        └── logoIara.png

````

---

# ⚙️ Tecnologias utilizadas

## Frontend

- React
- TypeScript
- Vite
- CSS

## Backend

- Python
- FastAPI
- Uvicorn
- Pandas
- Requests

## Inteligência Artificial

- Integração com API de LLM (IA generativa)


# 🔧 Backend

link repositório: https://github.com/raissabispo/iara-backend

Entre na pasta:

```bash
cd iara-backend
```

Crie o ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente:

Windows

```bash
.venv\Scripts\activate
```

Linux / Mac

```bash
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o servidor:

```bash
python -m uvicorn main:app --reload
```

Servidor rodará em:

```
http://localhost:8000
```

---

# 💻 Frontend

link do repositório: https://github.com/raissabispo/iara-frontend
Entre na pasta:

```bash
cd iara-frontend
```

Instale dependências:

```bash
npm install
```

Execute:

```bash
npm run dev
```

Aplicação rodará em:

```
http://localhost:5173
```

---

# 🔗 API Endpoints

### Chat com IA

```
POST /chat
```

Body:

```
{
  "pergunta": "Quanto gastei em alimentação?"
}
```

---

### Upload de CSV

```
POST /upload-csv
```

Envia arquivo CSV com transações financeiras.

---

# 🎯 Objetivo do projeto

O objetivo da IARA é demonstrar como **Inteligência Artificial pode ser usada para análise financeira pessoal**, tornando dados financeiros mais acessíveis e compreensíveis para qualquer usuário.

---

# 🔮 Melhorias futuras

* 📊 geração automática de gráficos
* 📈 dashboard financeiro
* 📂 arrastar CSV para upload (drag & drop)
* 🧠 memória de conversas
* ☁ deploy em nuvem (Render / Vercel)

---

# 👩‍💻 Autor

Projeto desenvolvido por **Raissa Vitória**.




