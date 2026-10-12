# m3s / M3Sculptor

力学超材料**论文检索增强生成（RAG）**助手，使用 LangGraph StateGraph 编排论文中的检索、结构生成与物理评估流程。normal 模式采用 64³ 体素、八子块镜像和侵蚀/膨胀候选遍历。

两轮流程：

1. 用户输入研究主题 → 知识图谱（Neo4j + Milvus）检索论文 → 逐篇分析相关性 → 合适则生成分析报告，不合适则自动换下一篇。
2. 用户给出性质目标与容差 → 分析该论文的子图结构是否满足 → 满足则返回图片供选择，不满足则重新推荐论文。

## 运行

```bash
conda activate m3s
pip install -e .
bash src/m3s/START.sh --port 8766 --mode test
```

界面和工作流在同一个进程内执行，没有独立后端。详见 `src/m3s/app/README.md`。

## 基础设施

```bash
sudo bash check.sh   # 启动 neo4j + mongod + milvus
```
