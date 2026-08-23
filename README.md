# MoviePilot OCR

MoviePilot 使用的站点验证码 OCR 服务。服务在本地离线识别图片中的 6 位大写英文字母和数字，并通过兼容 MoviePilot 的 HTTP API 返回结果。

## 识别方案

本项目针对站点验证码做了以下处理：

- 使用 `ddddocr` 验证码专用模型，代替面向通用文档的 PaddleOCR。
- 仅提取深色字符，去除彩色背景图案。
- 使用连通域过滤孤立噪点，保留字符笔画。
- 针对彩色遮罩、水平线和垂直线生成有限候选，并以结果一致性和模型置信度消解易混淆字符。
- 复用单个模型实例，并使用推理锁避免并发请求相互影响。
- 输出必须是 6 位 ASCII 字母和数字，并统一为大写。

仓库内的 2 张样本和额外提供的 4 张同类样本共 6 张，当前回归结果为 `6/6`。该结果只代表现有样本，不等同于所有站点验证码的通用准确率。

## Docker 部署

镜像同时支持 `linux/amd64` 和 `linux/arm64`：

```bash
docker run -d \
  --name moviepilot-ocr \
  --restart unless-stopped \
  -p 9899:9899 \
  jxxghp/moviepilot-ocr:latest
```

检查服务：

```bash
curl http://127.0.0.1:9899/
```

返回：

```json
{"message":"MoviePilot OCR API"}
```

在 MoviePilot 中将 `OCR_HOST` 设置为可访问该容器的地址，例如：

```env
OCR_HOST=http://192.168.1.10:9899
```

容器和 MoviePilot 在同一个 Compose 网络时，也可以使用容器服务名：

```env
OCR_HOST=http://moviepilot-ocr:9899
```

## API

### 识别 Base64 图片

`POST /captcha/base64`

请求：

```json
{
  "base64_img": "iVBORw0KGgoAAA..."
}
```

响应：

```json
{
  "result": "77D2A8"
}
```

`base64_img` 支持纯 Base64 和 `data:image/png;base64,...` 格式。图片解码后最大 5 MiB，最大尺寸为 4096 x 4096 像素。无效 Base64 或图片返回 `400`，超过限制返回 `413`。

本地图片调用示例：

```bash
IMAGE_B64=$(base64 < captcha.png | tr -d '\n')
curl --request POST http://127.0.0.1:9899/captcha/base64 \
  --header 'Content-Type: application/json' \
  --data "{\"base64_img\":\"${IMAGE_B64}\"}"
```

FastAPI 交互文档位于 `http://127.0.0.1:9899/docs`。

## 源码运行

需要 Python 3.10 至 3.12：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 9899
```

模型会在进程启动时加载。首次启动和首次识别通常比后续请求慢。

运行回归测试：

```bash
python -m unittest -v test.py
```

## 构建镜像

构建当前平台镜像：

```bash
docker build -t moviepilot-ocr:local .
```

GitHub Actions 在 `main` 分支的服务代码、依赖或 Docker 构建文件变更后，自动构建并推送 `jxxghp/moviepilot-ocr:latest` 多架构镜像。

## 限制

- 当前预处理针对浅色背景、深色字符的站点验证码优化。
- 当前输出统一为大写，不适用于严格区分大小写且包含小写字母的验证码。
- 滑块、点选、算术题和中文验证码不在支持范围内。

## License

[MIT](LICENSE)
