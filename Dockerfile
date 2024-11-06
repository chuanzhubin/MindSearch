FROM continuumio/miniconda3

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

ARG BING_API_KEY
ENV BING_API_KEY=${BING_API_KEY}

# 设置环境变量
ENV PATH=/opt/conda/bin:$PATH

# 克隆git仓库
#RUN git clone -b lagent_v1.0 https://github.com/braisedpork1964/MindSearch.git /app
# 复制本地 mindsearch 目录到镜像的 /app 目录
COPY . /app

WORKDIR /app

# 创建并激活 fastapi 环境，并安装依赖包
RUN conda create --name fastapi python=3.10 -y && \
    conda run -n fastapi pip install -r requirements.txt && \
    conda clean --all -f -y

# 暴露 FastAPI 默认端口
EXPOSE 8000

# 启动 FastAPI 服务
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "fastapi"]
#python -m mindsearch.app --lang cn --model_format internlm_silicon --search_engine Elasticsearch
#python -m mindsearch.app --lang cn --model_format internlm_silicon --search_engine DuckDuckGoSearch
CMD ["python3", "-m", "mindsearch.app", "--lang", "cn", "--model_format", "internlm_silicon", "--search_engine", "DuckDuckGoSearch", "--host", "0.0.0.0", "--port", "8002"]