# VoiceToText - Whisper 转录工具

基于 Faster-Whisper 的音频转文字工具，支持批量处理 MP3 文件。

## 功能特性

- 支持多种语言转录（日语、英语、马来语、印尼语、越南语、泰语、俄语等）
- 批量处理：选择文件夹，自动转录所有 MP3 文件
- 智能跳过：已存在同名 JSON 文件的 MP3 会自动跳过
- 现代化界面：基于 CustomTkinter 的简洁 UI

## 快速开始（推荐）

### 方法一：直接运行 EXE

1. 下载 `dist/VoiceToText_CTK.exe`
2. 双击运行
3. 首次运行会自动下载 Whisper 模型（约 500MB）
4. 选择包含 MP3 文件的文件夹，点击 "Start Transcription"

### 方法二：开发者模式（重新打包）

如果你想修改代码并重新打包：

```bash
# 1. 克隆或下载项目

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行测试
python gui_ctk.py

# 4. 打包
pyinstaller VoiceToText_ctk.spec --noconfirm
```

打包完成后，可执行文件位于 `dist/VoiceToText_CTK.exe`

## 使用说明

1. **选择文件夹**：点击 "Browse Folder" 选择包含 MP3 文件的文件夹
2. **配置参数**：
   - Model Size: tiny/base/small/medium/large/large-v2/large-v3（越大越准确，但越慢）
   - Device: cpu 或 cuda（需要 NVIDIA 显卡）
   - Compute Type: int8/int16/float16
   - Language: 选择源语言或 auto（自动检测）
   - Beam Size: 1-10，数值越大越准确，但越慢
3. **开始转录**：点击 "Start Transcription"
4. **查看结果**：JSON 文件会保存在原 MP3 文件的同一目录下

## 配置说明

在 `config.py` 中可以修改默认配置：

```python
COMPUTE_TYPE = "int16"  # 可选: int8, int16, float16
```

## 依赖

- Python 3.8+
- faster-whisper
- customtkinter
- pyinstaller

## 注意事项

- 首次运行会自动下载 Whisper 模型（~500MB）
- 模型默认缓存在 `C:\Users\<用户名>\.cache\huggingface\hub\`
- 批量处理时，已存在同名 JSON 的文件会自动跳过
- 如果遇到问题，查看同目录下的 `app.log` 日志文件

## 许可证

MIT License
