# YOLOv8 Trainer Web Interface

Este projeto fornece uma interface web para treinamento, upload e teste de modelos YOLOv8.

## Features

- Upload e treinamento de modelos YOLOv8 customizados
- Upload de modelos YOLOv8 pré-treinados
- Teste de detecção de objetos através do upload de imagens
- Visualização de resultados de detecção com caixas delimitadoras e pontuações de confiança

## Estrutura do Projeto

```
├── backend/            # API Flask backend
│   ├── app.py          # Aplicação Flask principal  
│   └── requirements.txt # Dependências Python
├── frontend/           # Frontend React
│   ├── public/         # Assets estáticos
│   └── src/            # Código fonte React
├── models/             # Modelos armazenados (ignorados pelo Git)
├── uploads/            # Imagens enviadas (ignoradas pelo Git)
└── train_data/         # Datasets de treinamento (ignorados pelo Git)
```

## Controle de Versão

Este projeto utiliza Git para controle de versão, com as seguintes configurações:

- Arquivos ignorados pelo Git (configurados no `.gitignore`):
  - Diretório `models/` (contém arquivos de modelo pesados)
  - Diretório `train_data/` (contém datasets de treinamento)
  - Diretório `uploads/` (contém imagens enviadas pelos usuários)
  - Arquivos com extensão `.pt` (arquivos de modelo PyTorch)
  - Diretório `node_modules/` (dependências do Node.js)
  - Arquivos temporários e de cache

## Instruções de Configuração

### Pré-requisitos

- Python 3.8+ 
- Node.js 16+ 
- npm ou yarn

### Configuração do Backend (Flask)

1. Abra um terminal e navegue até o diretório backend:

```
cd backend
```

2. Crie um ambiente virtual (recomendado):

```
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Instale as dependências necessárias:

```
pip install -r requirements.txt
```

4. Inicie o servidor Flask:

```
python app.py
```

O servidor backend será iniciado em http://localhost:5000

### Configuração do Frontend (React)

1. Abra um novo terminal e navegue até o diretório frontend:

```
cd frontend
```

2. Instale as dependências:

```
npm install
```

3. Inicie o servidor de desenvolvimento:

```
npm run dev
```

O servidor de desenvolvimento frontend será iniciado em http://localhost:3000

## Uso

1. Abra seu navegador e acesse http://localhost:3000
2. Use o menu de navegação para acessar diferentes recursos:
   - **Treinar Modelo**: Faça upload de um dataset e treine um modelo YOLOv8 personalizado
   - **Testar Modelo**: Selecione um modelo e faça upload de imagens para detecção de objetos
   - **Upload de Modelo**: Faça upload de arquivos de modelo YOLOv8 pré-treinados (.pt)

## Formato dos Dados de Treinamento

Os dados de treinamento devem estar no formato YOLO, tipicamente um arquivo ZIP contendo:

- Diretório `images/` com imagens de treinamento
- Diretório `labels/` com arquivos de rótulos correspondentes
- Arquivo de configuração `data.yaml` definindo as classes

## Notas

- O treinamento real de modelos é intensivo em recursos e pode exigir uma GPU.
- Para uso em produção, considere implementar processamento de tarefas em segundo plano para tarefas de treinamento.
- Arquivos de modelo grandes podem exigir ajustes nos limites de tamanho de arquivo do servidor. 