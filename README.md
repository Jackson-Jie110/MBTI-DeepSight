# MBTI DeepSight · 人格解码实验室

> 从答题到 AI 侧写，一次完成可分享、可运营、可观测的 MBTI 在线测评闭环。

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?logo=fastapi&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.x-D71F00?logo=sqlalchemy&logoColor=white)
![Jinja2](https://img.shields.io/badge/Jinja2-Template-B41717?logo=jinja&logoColor=white)
![HTMX](https://img.shields.io/badge/HTMX-1.9.x-3366CC?logo=htmx&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-runtime-06B6D4?logo=tailwindcss&logoColor=white)
![Vercel](https://img.shields.io/badge/Deploy-Vercel-000000?logo=vercel&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 项目介绍 (Introduction)

`MBTI DeepSight` 是一个基于 FastAPI + Jinja2 的在线 MBTI 测试系统，覆盖「开始测试 → 渐进答题 → 结果生成 → AI 深度解读 → 用户反馈 → 后台运营」完整链路。

它解决了常见 MBTI 小工具的三个痛点：

- **测试不稳定**：按 EI/SN/TF/JP 四维均衡抽题，并支持边界自动加测（最多 +10 题）。
- **结果解释浅**：结果页与分析页均支持 AI 生成内容，含流式输出与容错处理。
- **运营不可视**：提供后台反馈看板、错误日志看板、Excel 导出与清理能力，便于持续优化。

> 项目当前为 FastAPI 服务端渲染架构（非 Flask），本地默认 SQLite，线上可切换 PostgreSQL。

## ✨ 核心功能 (Key Features)

1. **分层测评流程 + 边界加测机制**  
   支持 20/40/60 题模式，5 级量表，一题一页；当维度分值接近边界时自动触发 Tie-breaker 加测，降低误判。

2. **双 AI 能力：结果流式侧写 + 分析卡片生成**  
   - 结果页：`/result/ai_content/{share_token}` 流式输出 Markdown，前端逐步渲染。  
   - 分析页：`/analysis/content_card` 生成结构化内容并渲染卡片。  
   同时引入 `json_repair`，提升模型 JSON 输出解析鲁棒性。

3. **异常可观测与前端容灾体验**  
   后端将 AI 异常落库到 `error_logs`，前端统一错误弹窗提示并支持重试，后台可查看原始响应与错误详情。

4. **后台运营能力完整**  
   反馈看板与报错看板独立分页（5条/页）、单条删除、批量清空、删除免打扰、Excel 导出（`.xlsx`）等功能已集成。

## 🖼️ 界面展示 (Screenshots)

> 请将截图放到 `docs/screenshots/`（或你习惯的目录），并替换下方链接。

- 首页（模式选择）  
  ![首页](docs/screenshots/home.png)
- 测试页（一题一页 + 5级量表）  
  ![测试页](docs/screenshots/test.png)
- 结果页（类型报告 + AI 流式解读）  
  ![结果页](docs/screenshots/result1.png)
  ![结果页](docs/screenshots/result2.png)
- 后台管理（反馈/报错双看板）  
  ![后台管理](docs/screenshots/admin-dashboard.png)

## 🛠️ 技术栈 (Tech Stack)

### 后端

- **FastAPI**：路由与接口编排（`app/main.py`, `app/routes/*.py`）
- **SQLAlchemy 2.x**：ORM 模型与数据访问（`app/models.py`）
- **Jinja2**：SSR 模板渲染（`app/templates/`）
- **Uvicorn**：ASGI 运行时

### 前端

- **HTMX**：增强式导航与局部交互（`hx-boost`、ext preload/sse）
- **TailwindCSS runtime + 自定义 CSS**：页面样式体系
- **Vanilla JS**：流式渲染、反馈提交、后台交互逻辑
- **Marked.js**：Markdown 前端渲染
- **Chart.js**：分析页雷达图

### AI 与数据处理

- **OpenAI SDK (AsyncOpenAI)**：调用兼容 OpenAI 协议的模型网关
- **默认模型参数**：`deepseek-ai/DeepSeek-V3.2`（可通过环境变量覆盖）
- **json_repair**：修复不规范 JSON 输出
- **openpyxl**：反馈数据导出 Excel（`.xlsx`）

### 数据库与部署

- **开发环境**：SQLite（默认 `sqlite:///./mbti.db`）
- **生产环境**：PostgreSQL（通过 `MBTI_DATABASE_URL`/`DATABASE_URL`）
- **部署配置**：`vercel.json`（Vercel）、`render.yaml`（Render）

## 🚀 快速开始 (Getting Started)

### 1) 克隆项目

```bash
git clone <your-repo-url>
cd MBTI
```

### 2) 创建并激活虚拟环境

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) 安装依赖

```bash
pip install -r requirements.txt
```

### 4) 配置环境变量（`.env`）

项目未内置 `python-dotenv` 自动加载，请在系统环境中设置，或通过启动命令前缀注入。

建议准备如下变量：

```env
# 应用签名密钥（必须：生产环境务必设置）
MBTI_APP_SECRET=replace-with-a-long-random-secret

# 数据库（可选：本地不配则使用 sqlite:///./mbti.db）
MBTI_DATABASE_URL=sqlite:///./mbti.db
# 或者使用平台注入的 DATABASE_URL

# AI 配置（启用 AI 功能必填 API Key）
MBTI_AI_API_KEY=your_api_key
MBTI_AI_BASE_URL=https://api.siliconflow.cn/v1
MBTI_AI_MODEL=deepseek-ai/DeepSeek-V3.2

# 管理后台登录（强烈建议生产环境配置）
MBTI_ADMIN_USERNAME=admin
MBTI_ADMIN_PASSWORD=change_me

# 可选：管理员会话有效期（秒）
MBTI_ADMIN_SESSION_MAX_AGE_SECONDS=1209600
```

### 5) 初始化数据库与题库

```bash
python scripts/db_upgrade.py
python scripts/seed_questions.py
```

### 6) 启动服务

```bash
uvicorn app.main:app --reload
```

访问地址：

- 首页：`http://127.0.0.1:8000/`
- 题库管理：`http://127.0.0.1:8000/admin/questions`
- 反馈看板：`http://127.0.0.1:8000/admin/dashboard?key=jackson_admin`

### 7) 运行测试（可选）

```bash
pip install -r requirements-dev.txt
pytest -q
```

## ☁️ 部署 (Deployment)

### Vercel

仓库已包含 `vercel.json`，入口为 `app/main.py`。部署时至少配置：

- `MBTI_APP_SECRET`
- `MBTI_DATABASE_URL`（或平台 `DATABASE_URL`）
- `MBTI_AI_API_KEY`（若启用 AI）
- `MBTI_ADMIN_USERNAME` / `MBTI_ADMIN_PASSWORD`

### Docker（示例思路）

项目当前未提供现成 `Dockerfile`，可按下述思路构建：

- 基于 Python 3.11+ 镜像
- `pip install -r requirements.txt`
- 启动命令：`uvicorn app.main:app --host 0.0.0.0 --port 8000`
- 通过环境变量注入数据库与密钥配置

## 🧭 未来规划 (Future Planning)

- [ ] 建立用户身份体系与历史测评档案，实现跨设备登录、结果追踪与长期画像沉淀。
- [ ] 将核心数据层升级至 PostgreSQL 集群化方案，提升并发承载能力与生产稳定性。
- [ ] 持续扩充多维心理测评题库，增强题目覆盖度、区分度与结果解释深度。
- [ ] 将部署架构从 Vercel 迁移至云服务器（IaaS）方案，构建更可控的运维与发布链路。

## 📄 许可证 (License)

本项目采用 **MIT License**。
