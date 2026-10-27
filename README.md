# LIDA on Gemini

用 Google Gemini（也可以換成 Claude）跑 Microsoft 的 [LIDA](https://github.com/microsoft/lida)：資料 → 摘要 → 分析目標 → 繪圖程式 → 圖 → 修改／解釋／評分。

- [try_lida.ipynb](try_lida.ipynb)：試用筆記本，從上往下一格一格執行
- [lida_result.html](lida_result.html)：一次完整執行的結果（程式碼、輸出、圖）
- [gemini_textgen.py](gemini_textgen.py) / [claude_textgen.py](claude_textgen.py)：讓 LIDA 改接 Gemini / Claude 的轉接層

## 執行

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env   # 填入 GEMINI_API_KEY（到 https://aistudio.google.com/apikey 免費申請）
```

接著用 VS Code 或 Jupyter 打開 `try_lida.ipynb`，kernel 選 `.venv`。
