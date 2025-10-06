# Dockerfile

# 1. Imagem Base: Comece com uma imagem Python oficial e enxuta.
FROM python:3.12-slim

# 2. Variáveis de Ambiente: Evita que o Python gere arquivos .pyc e armazene logs em buffer.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 3. Diretório de Trabalho: Define o diretório onde o código será executado dentro do contêiner.
WORKDIR /app

# 4. Instalar Dependências:
# Copia apenas o arquivo de requirements primeiro para aproveitar o cache do Docker.
# Se este arquivo não mudar, o Docker não reinstalará tudo a cada build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar o Código do Projeto:
# Copia as pastas com suas funções e os arquivos de configuração para o contêiner.
COPY Functions/ ./Functions/
COPY config.py .
COPY main.py .

# 6. Copiar Arquivos de Entrada (se necessário):
# Se a sua lista de SKUs ou outros arquivos de input estiverem no repositório, copie-os também.
# Lembre-se de ajustar os caminhos no config.py para refletir a estrutura dentro do contêiner.
COPY "Lista Produtos"/ ./Lista Produtos/
# COPY Forecast/ ./Forecast/

# 7. Comando de Execução:
# Define o comando que será executado quando o contêiner iniciar.
CMD ["python", "main.py"]
