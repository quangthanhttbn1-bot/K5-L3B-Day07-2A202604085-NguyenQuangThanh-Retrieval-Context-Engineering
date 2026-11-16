# Code

`retrieval_experiment.py` chạy so sánh phần 2 trên tập dev:

- `tfidf`: lexical baseline, dùng chung IDF trên các chunk strategy B.
- `dense`: query embedding bằng `BAAI/bge-m3`, tìm exact cosine trong pgvector Day 6.
- `hybrid`: lấy 10 kết quả từ mỗi retriever và hợp nhất bằng Reciprocal Rank Fusion (RRF, hằng số 60).

Lệnh chạy từ thư mục gốc (sửa đường dẫn Day 6 nếu project của bạn ở nơi khác):

```powershell
python -m src.retrieval_experiment `
  --method all `
  --strategy B `
  --day6-starter "D:\2026_Vin_AI_course\Day_06\K5-L3B-Day06-2A202604085-DataIngestionEngineering\starter"
```

Output gồm `tfidf-traces.json`, `dense-traces.json`, `hybrid-traces.json` và summary tương ứng trong `starter/results/retrieval/`. Trace lưu rank/score, source/page/metadata, context, evidence hit, thời gian và lỗi. Gold chỉ được dùng sau retrieval để chấm.

Điều kiện cho dense/hybrid: bật PostgreSQL + pgvector Day 6, cài các dependency Day 6 (Sentence Transformers, psycopg và pgvector) trong cùng Python environment, có model BGE-M3 local hoặc quyền tải model. Runner chỉ đọc bảng; không ghi/xóa dữ liệu. Nếu service hoặc model không sẵn sàng, trace ghi lỗi; không dùng vector giả hay fallback giả lập.

`data/day7/benchmark_dev.json` được chuyển từ 15 câu dev Day 6; page ID unit được ánh xạ sang số trang và evidence được kiểm tra sau khi chuẩn hóa xuống dòng. Có thêm hai câu thiếu bằng chứng để kiểm tra retrieval; chúng không nằm trong mẫu số Hit@k. Runner này chặn split khác `dev` để test không bị dùng khi chọn cấu hình.

Nếu PostgreSQL hiện không chạy, có thể tái lập phần retrieval với trace dense dev đã lưu từ lần chạy Day 6 (chỉ có 15 câu trả lời được):

```powershell
python -m src.retrieval_experiment `
  --method all `
  --strategy B `
  --candidate-k 3 `
  --dense-traces "D:\2026_Vin_AI_course\Day_06\K5-L3B-Day06-2A202604085-DataIngestionEngineering\starter\results\traces.json"
```

Trong chế độ này, hai câu thiếu evidence được ghi `not_run` vì trace Day 6 không có kết quả cho chúng; kết quả retrieval vẫn tính trên 15 câu answerable, đúng mẫu số benchmark.

Vòng lặp agent dùng interface tại [starter/agentic.py](../starter/agentic.py); controller và retriever thật được nối trong `day7_agent.py`.

## Phần 3: agent và context

`day7_agent.py` nối `starter/agentic.py` với OpenRouter (mặc định) hoặc OpenAI API và retrieval Day 6 qua OpenAI-compatible Chat Completions. Model có một tool `search_evidence`; code ứng dụng thực hiện dense, TF-IDF hoặc hybrid. Agent mặc định dùng dense strategy B, top-k 5, tối đa 3 lượt search, context budget 3.000 ký tự. Context được khử trùng lặp theo chunk ID, sắp theo score cao nhất qua các lượt tìm, kèm source/page/ID; model phải trích dẫn ID đang có trong context hoặc abstain. Nội dung tài liệu được đánh dấu là dữ liệu không đáng tin và không được làm theo chỉ dẫn bên trong.

### Cấu hình API key

```powershell
Copy-Item .env.example .env
notepad .env
python -m pip install -r requirements-agent.txt
```

Trong `.env` trên máy bạn, để `AI_PROVIDER=openrouter`, điền key vào `OPENROUTER_API_KEY=...` và chọn model ID của OpenRouter ở `OPENROUTER_MODEL` (mặc định `openai/gpt-5.6-luna`). Nếu dùng OpenAI trực tiếp, đặt `AI_PROVIDER=openai` rồi dùng `OPENAI_API_KEY`/`OPENAI_MODEL`. Không gửi key qua chat. `.env` và các file `.env.*` bị Git bỏ qua; `.env.example` được phép commit vì chỉ chứa giá trị trống. Nếu key từng được commit/push, thu hồi key đó trong dashboard của nhà cung cấp rồi tạo key mới; chỉ xóa file khỏi commit mới không làm key cũ an toàn.

### Chạy trên dev

```powershell
python -m src.day7_agent `
  --split dev `
  --method dense `
  --strategy B `
  --day6-starter "D:\2026_Vin_AI_course\Day_06\K5-L3B-Day06-2A202604085-DataIngestionEngineering\starter"
```

Trace lưu từng model call (token count/latency), tool call (query/filter/top-k), kết quả, context, citation, lỗi và thời gian trong `starter/results/agent/`. Script không đưa `gold`, đáp án chuẩn hay nhãn thiếu bằng chứng vào prompt hoặc retriever. Gọi API tính phí theo mức sử dụng và cấu hình billing tài khoản API.
