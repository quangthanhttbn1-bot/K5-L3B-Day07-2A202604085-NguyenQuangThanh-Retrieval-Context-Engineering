# Báo cáo Day 7 — phần 3 và 4 (dev)

## Dữ liệu và benchmark

- Corpus: Project Gutenberg eBook 7213, *My Life and Work* (Henry Ford, Samuel Crowther); xem manifest và báo cáo Day 6 tại thư mục Day 6.
- Vector store/index: PostgreSQL + pgvector; embedding `BAAI/bge-m3`, 1.024 chiều. Các lần so sánh dense dưới đây phát lại trace Day 6 đã lưu, không tính lại thời gian embedding/query.
- Agent: OpenRouter Chat Completions, model `openai/gpt-5.6-luna`; chỉ chạy dev. Không dùng gold trong prompt hay retrieval.
- Dev có 15 câu answerable và 2 câu unanswerable. Retrieval metrics chỉ lấy 15 câu answerable làm mẫu số.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, tham số | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---:|---:|---:|---|
| Baseline | TF-IDF, strategy B, top-k 3, candidate 10, context 3.000 ký tự | Baseline lexical | dev | 5/15 (33,3%) | 9/15 (60,0%) | 8/15 (53,3%) | median 1,80 ms; local, $0 |
| Dense | BGE-M3 + pgvector trace Day 6, strategy B, top-k 3, context 3.000 ký tự | Dense baseline | dev | 11/15 (73,3%) | 13/15 (86,7%) | 12/15 (80,0%) | Không đo lại; trace replay, chi phí dịch vụ $0 |
| Hybrid RRF | TF-IDF + dense trace, RRF-60, top-k 3, context 3.000 ký tự | Kết hợp lexical và semantic ranking | dev | 11/15 (73,3%) | 13/15 (86,7%) | 13/15 (86,7%) | median 1,85 ms chưa gồm dense; local, $0 |
| Cải thiện 1 | Dense/hybrid, top-k 5, candidate 10, context 4.500 ký tự | Ngân sách context rộng hơn có thể giữ thêm evidence | dev | 11/15 (73,3%) | 13/15 (86,7%) | Dense 13/15 (86,7%); hybrid 13/15 (86,7%) | Dense replay; hybrid median 2,68 ms chưa gồm dense; $0 |
| Cải thiện 2 | Dense/hybrid, top-k 5, candidate 20, context 4.500 ký tự | Nhiều lexical candidates hơn có thể giúp RRF | dev | 11/15 (73,3%) | 13/15 (86,7%) | 13/15 (86,7%) | Dense replay; hybrid median 1,96 ms chưa gồm dense; $0 |

Hit@1 và Evidence Hit@3 lấy từ ranking; Context Evidence Rate kiểm tra evidence có nằm trong context cuối. Các câu thiếu evidence không nằm trong mẫu số retrieval. Cải thiện 1 giúp dense context rate tăng 1 câu so baseline dense; hybrid không tăng thêm so hybrid baseline. Cải thiện 2 không tăng hit/context so cải thiện 1 dù tăng candidate. Median hybrid chỉ đo TF-IDF, hợp nhất và đóng gói context vì dense được phát lại từ trace; không so trực tiếp như latency end-to-end.

Hai lỗi dev đã đọc:

1. **q06 — “What imbalance in business practice surprised Ford?”** TF-IDF top-3 trả đoạn về giá bán, bảng chỉ mục và danh sách tên; dense top-3 trả đoạn đường sắt, mục lục và một đoạn về vốn cổ phần. Không phương pháp nào lấy được evidence mục tiêu trong top-3. Câu hỏi dùng “imbalance” trong khi evidence diễn đạt bằng các khái niệm khác; cả hai retriever bị kéo về finance/business chung. Tăng candidate 10→20 và context 3.000→4.500 không sửa được ranking này.
2. **q14 — “What did Ford credit for the rapid move into large-scale production?”** TF-IDF top-3 lẫn đoạn về vòng quay lợi nhuận, máy kéo và ngân hàng; dense top-3 tập trung vào xe giá thấp/đối tượng khách hàng. Cụm “credit”, “rapid move” và “large-scale production” không đủ đặc trưng với corpus; tăng candidate/context không làm evidence vào top-3.

Các lần cải thiện hiện thử cấu hình top-k/context/candidate; chưa thử query rewrite cho q06/q14, nên hai lỗi vẫn mở. Cấu hình retrieval chọn tạm cho dev là **dense strategy B, top-k 3, context 4.500 ký tự** vì đạt Context Evidence Rate 13/15 với trace replay; hybrid đạt cùng mức nhưng không cải thiện evidence hit. Đây chưa phải cấu hình chốt cho test.

Trace và summary:

- Baseline ba phương pháp: `starter/results/retrieval/{tfidf,dense,hybrid}-{traces,summary}.json` và `all-summary.json`.
- Cải thiện 1: `starter/results/retrieval/improvement-context-4500/`.
- Cải thiện 2: `starter/results/retrieval/improvement-candidate20/`.
- Lệnh baseline: `python -m src.retrieval_experiment --method all --strategy B --top-k 3 --candidate-k 10 --budget-chars 3000 --dense-traces <Day6>/starter/results/traces.json`.

## Agent và đánh giá

- Agent thật đã chạy dev với TF-IDF do runtime không có dependency dense Day 6. Trace: `starter/results/agent/dev-tfidf.json`; summary: `starter/results/agent/dev-tfidf-summary.json`.
- Trong 17 câu: 3 answered, 9 abstain, 5 provider errors. Có 24 retrieval tool call, 36 model call, 32.277 input token và 1.874 output token. 5 lỗi là OpenRouter `429` rate limit của tài khoản mới. Chi phí USD không có trong phản hồi provider nên chưa xác định.
- Cải tiến điều phối sau trace: khi đã dùng hết search budget, agent đánh dấu lượt cuối `search_disabled` và buộc controller trả lời bằng context hiện có hoặc abstain. Thay đổi này xử lý các trace `search_limit` do model tiếp tục xin search; chưa chạy lại để đo hiệu quả vì giới hạn 429.
- Chưa gán grounded answer/citation support bằng người chấm. Citation ID tồn tại chưa đủ chứng minh ngữ nghĩa hỗ trợ câu trả lời; cần đối chiếu source/page trước khi báo các metric này.
- Vì agent dùng TF-IDF và bị rate limit, chưa có lần chạy agent so sánh dense/hybrid hoàn chỉnh. Không chạy test trong lượt này.

## Giới hạn và bước tiếp theo

- Dense và hybrid trong bảng dùng trace Day 6 đã lưu: latency replay không gồm BGE-M3 query embedding hoặc pgvector. Cần cài đủ dependency Day 6 và chạy database để đo end-to-end.
- Cần giải quyết rate limit (giãn tốc độ/retry hoặc quota phù hợp), chạy lại agent trên cùng dev với cấu hình đã chọn, rồi người chấm đọc citation/context cho ít nhất 5 câu có đáp án và 2 câu thiếu evidence.
- Đã tạo holdout mới `data/day7/benchmark_test.json` với 5 answerable + 2 unanswerable, ID và câu hỏi tách khỏi dev và test cũ Day 6. Năm gold evidence được đối chiếu với chunk strategy B; hai câu thiếu được rà trên toàn corpus. Chưa chạy retrieval/agent trên holdout mới; giữ nguyên để đánh giá sau khi chốt cấu hình.
- Trace Day 6 có 5 test answerable nhưng báo cáo Day 6 ghi test đã được xem trong quá trình xử lý lỗi; không dùng trace cũ đó như đánh giá độc lập.
