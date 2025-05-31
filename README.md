# ai_toy

AI Toy 项目用于实验和开发 AI 相关功能，包含多种深度学习模型的实现与测试。

## 推荐开发环境搭建流程

1. **安装 Python 3.12 及以上版本**
   - 推荐使用 [pyenv](https://github.com/pyenv/pyenv) 管理多版本 Python。

2. **创建虚拟环境**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows 下为 venv\Scripts\activate
   ```

3. **升级 pip**
   ```bash
   pip install --upgrade pip
   ```

4. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

5. **（可选）检查 Python 版本和依赖**
   ```bash
   python --version
   pip list
   ```

## 主要功能
- Seq2Seq RNN 翻译模型
- 基于 Transformer 的字符级模型
- 基于 RNN 的字符级模型

## 贡献
欢迎提交 issue 和 PR 以完善本项目。

## License
MIT 