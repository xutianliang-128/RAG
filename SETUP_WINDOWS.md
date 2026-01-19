# Windows 环境变量设置指南

## 设置 OpenAI API Key

### 方法 1: 使用 .env 文件（推荐）

1. 在项目根目录创建 `.env` 文件
2. 添加以下内容：
```
OPENAI_API_KEY=your_actual_api_key_here
```

3. 运行程序：
```bash
python run_all_cities.py
```

### 方法 2: PowerShell（临时设置）

在 PowerShell 中运行：
```powershell
$env:OPENAI_API_KEY="your_api_key_here"
python run_all_cities.py
```

### 方法 3: CMD（临时设置）

在 CMD 中运行：
```cmd
set OPENAI_API_KEY=your_api_key_here
python run_all_cities.py
```

### 方法 4: 永久设置系统环境变量

1. 按 `Win + R`，输入 `sysdm.cpl`，回车
2. 点击"高级"选项卡
3. 点击"环境变量"
4. 在"用户变量"中点击"新建"
5. 变量名：`OPENAI_API_KEY`
6. 变量值：你的 API Key
7. 点击"确定"保存

## 验证设置

运行以下命令验证环境变量是否设置成功：

**PowerShell:**
```powershell
echo $env:OPENAI_API_KEY
```

**CMD:**
```cmd
echo %OPENAI_API_KEY%
```

## 注意事项

- `.env` 文件方法最简单，推荐使用
- 如果使用环境变量，确保在同一个终端窗口中运行程序
- 不要将 `.env` 文件提交到 Git（已包含在 .gitignore 中）
