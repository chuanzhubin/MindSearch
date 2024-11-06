# 使用官方的 Node.js 18 镜像作为基础镜像
FROM node:18



# 设置环境变量
#ENV PATH=/opt/conda/bin:$PATH

# 克隆git仓库
RUN git clone -b lagent_v1.0 https://github.com/braisedpork1964/MindSearch.git /git/MindSearch

WORKDIR /git/MindSearch/frontend/React

# 安装项目依赖
RUN npm install

# 暴露 FastAPI 默认端口
EXPOSE 8080

# 启动应用
CMD ["npm", "start"]