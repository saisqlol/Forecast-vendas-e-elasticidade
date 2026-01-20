# Dockerfile

# 1. Imagem Base:
FROM python:3.12-slim

# 2. Variáveis de Ambiente: Evita que o Python gere arquivos .pyc e armazene logs em buffer.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. Diretório de Trabalho: Define o diretório onde o código será executado dentro do contêiner.
WORKDIR /app

# 4. Instalar Dependências:

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar o Código do Projeto:
# Copia as pastas com suas funções e os arquivos de configuração para o contêiner.
COPY Functions/ ./Functions/
COPY config.py .
COPY main.py .

# 6. Copiar Arquivos de Entrada (se necessário):

COPY "Lista Produtos"/ ./Lista Produtos/
# COPY Forecast/ ./Forecast/

# 7. Comando de Execução:

CMD ["python", "main.py"]
