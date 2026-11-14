# DAY07 · Retrieval và Context Engineering cho tài liệu VLA

**Nguyễn Tuấn Dũng · 2A202603608 · L3A**

## Dữ liệu và benchmark

Dùng lại 885 chunk chia theo heading của DAY06 trong Milvus Standalone 3.0.5, collection `day06_vla_headings_bf3374818539`. Corpus có 32 tài liệu SmolVLA, GR00T, OpenVLA, openpi và NVIDIA Agent Skills, trong đó 31 tài liệu đạt 1.500 ký tự chuẩn hóa. Manifest DAY06 ghi phiên bản, URL và giấy phép theo nguồn. Khi tái tạo collection, giữ nguyên ID, nội dung, URL, trang logic và category của từng chunk.

Ollama phục vụ `qwen3-embedding:0.6b` (1.024 chiều) và `qwen3:4b` trên Jetson. Milvus dùng FLAT/COSINE. Dữ liệu benchmark có 17 dev (15 có đáp án, 2 thiếu evidence) và 7 test (5 có đáp án, 2 thiếu evidence). Nhãn gold phục vụ chấm điểm. Những câu thiếu evidence được rà theo nguồn và thuật ngữ liên quan trong corpus. [Bộ câu hỏi](../../data/vla_benchmark.json) · [Cách chạy](../../starter/README.md)

## Retrieval và thử nghiệm

| Phương pháp | Split | Hit@1 | Evidence Hit@3 | Context đủ evidence | Tổng thời gian truy vấn |
|---|---|---:|---:|---:|---:|
| Dense | Dev | 5/15 (33,3%) | 6/15 (40%) | 6/15 (40%) | 0,77 s |
| TF-IDF | Dev | 3/15 (20%) | 4/15 (26,7%) | 4/15 (26,7%) | 0,03 s |
| Dense + rerank | Dev | 4/15 (26,7%) | 6/15 (40%) | 6/15 (40%) | 507,94 s |
| Dense, baseline và cấu hình chọn | Test | 2/5 (40%) | 4/5 (80%) | 4/5 (80%) | 3,25 s |

Cả ba sử dụng top 3 và ngân sách context 3.200 ký tự. Filter category chỉ áp dụng khi đã có trong input. Thử nghiệm thứ nhất dùng TF-IDF để tìm tên API và thuật ngữ cụ thể trên cùng corpus. Thử nghiệm thứ hai lấy 10 ứng viên dense rồi ưu tiên các đoạn được Qwen3 chọn. Các ứng viên còn lại giữ thứ tự dense. Lượt rerank ban đầu gặp đầu ra thiếu ứng viên và hai lượt ghi chung trace. Kết quả bảng dùng lượt chạy lại ở thư mục riêng. Rerank và agent có lúc chạy đồng thời trên Jetson, còn thời gian TF-IDF chưa tính khởi tạo index. Các số thời gian chỉ phản ánh lượt đo đã ghi. Chi phí API bằng 0, chưa đo điện năng.

Ở q09, lexical tìm được tài liệu phần cứng nhưng bỏ lỡ đoạn evidence về VRAM mà dense đưa vào top 3. Ở q14, lexical tìm được đoạn chuẩn hóa openpi còn dense đưa nhầm đoạn. Đoạn gold q01 bắc qua ranh giới hai mục nên khó đạt theo tiêu chí đoạn trích liền mạch. Ở test, truy vấn gốc vla-test-03 trả đoạn tổng quan và API policy, thiếu đoạn khai báo bốn khóa cấu hình trong data_config. [Phân tích dev](dev-retrieval.md) · [Quyết định trước khi chạy test](test-selection.md)

## Agent và đánh giá

Controller có tối đa hai lượt search, mỗi lượt lấy top 3. Context bỏ ID trùng và chỉ nhận nguyên chunk vừa ngân sách. Agent trả lời với ID citation có trong context, hoặc trả trạng thái abstain. Lỗi provider được lưu riêng. Tính hợp lệ của ID chỉ kiểm tra tính toàn vẹn của trace. Audit dưới đây đọc nội dung nguồn để kiểm tra citation và câu trả lời.

| Test agent dense | Kết quả |
|---|---:|
| Gọi search | 4/7, ba câu không gọi tool |
| Context chứa đủ gold | 3/5 |
| Grounded answer mức 0 / 1 / 2 | 4 / 0 / 1, trung bình 0,4/2 |
| Citation thực sự hỗ trợ đáp án | 1/1 |
| Câu trả lời đi kèm citation | 1/1 |
| Từ chối đúng theo nội dung cuối | 2/2 |
| Từ chối nhầm câu có đáp án | 4/5 |
| Lỗi provider | 0/7 |
| Tổng thời gian agent | 279,69 s |

| Câu | Context 0–2 | Kết quả và căn cứ chấm |
|---|---:|---|
| vla-test-01 | 2 | Open-loop và mục đích đánh giá có trong context. Agent từ chối do thiếu citation hợp lệ. |
| vla-test-02 | 0 | Agent không gọi search dù truy vấn gốc lấy được đoạn về empty_cameras. |
| vla-test-03 | 1 | Có hướng dẫn đối chiếu modality keys, thiếu đoạn khai báo đủ bốn khóa. Agent từ chối. |
| vla-test-04 | 2 | Trả lời đúng ALOHA_SIM và dẫn chunk chứa cấu hình Docker Compose. |
| vla-test-05 | 2 | Context đủ bước thu episode và tiền xử lý. Agent từ chối do thiếu citation hợp lệ. |
| vla-no-evidence-test-01 | — | Từ chối, không đưa số đo năng lượng. Không gọi search. |
| vla-no-evidence-test-02 | — | Từ chối, không đưa ma trận hiệu chuẩn. Không gọi search. |

Audit context có 1/1/3 câu ở mức 0/1/2. Hai câu thiếu evidence được từ chối đúng khi chấm nội dung cuối, nhưng **0/2 hoàn thành hành vi search_and_abstain**. Trace cũng cho thấy retrieval query gốc đạt 4/5 trong khi context cuối của agent chỉ đạt 3/5. Agent tạo query riêng và đôi khi bỏ qua tool. Cổng kiểm tra citation còn khiến hai câu bị từ chối dù context đã có gold. Đây là hạn chế quan trọng hơn việc tăng số ứng viên rerank trong lượt này. Muốn thử sửa, có thể buộc thực hiện lượt search đầu và kiểm tra cách model tạo citation trên dev, sau đó dùng tập test mới cho đánh giá độc lập.

Nhãn audit test do Codex đọc trace và đối chiếu bằng chứng. Học viên cần rà lại trước khi nộp. [Trace agent](../../results/day07/agent-test-dense/traces.jsonl) · [Nhãn](../../results/day07/agent-test-dense/reviews.json) · [Điểm audit](../../results/day07/agent-test-dense/audit-metrics.json) · [Trace truy vấn gốc](../../results/day07/retrieval-test-dense/traces.jsonl)

Tái lập từ gốc repo sau khi cấu hình Ollama và Milvus theo mẫu môi trường:

```powershell
python -X utf8 -m starter.run_benchmark --split test --method dense --output-root starter/results/test-repro
python -X utf8 -m starter.run_benchmark --split test --method dense --agent --output-root starter/results/test-repro
```
