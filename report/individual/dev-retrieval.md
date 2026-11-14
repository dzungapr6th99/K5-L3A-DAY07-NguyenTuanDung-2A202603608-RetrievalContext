# So sánh retrieval trên dev

Cùng 17 câu dev, trong đó 15 câu có đáp án, cùng collection DAY06 và context budget 3.200 ký tự. Dense là baseline. TF-IDF kiểm tra giả thuyết từ khóa kỹ thuật giúp tìm đúng đoạn. Dense + rerank kiểm tra khả năng đưa evidence từ 10 ứng viên lên top 3.

| Phương pháp | Hit@1 | Evidence Hit@3 | Context có evidence | Lỗi |
|---|---:|---:|---:|---:|
| Dense | 5/15 | 6/15 | 6/15 | 0 |
| TF-IDF | 3/15 | 4/15 | 4/15 | 0 |
| Dense + rerank | 4/15 | 6/15 | 6/15 | 0 |

- q09: dense tìm được đoạn nêu yêu cầu VRAM tối thiểu. TF-IDF đưa tài liệu liên quan vào top 3 nhưng thiếu đoạn evidence đã chốt.
- q14: TF-IDF tìm đúng đoạn normalization trong openpi. Dense tìm được tài liệu đó nhưng lấy nhầm đoạn.
- q01: đoạn gold hiện bắc qua ranh giới các mục trong collection B. Không có một chunk chứa đủ đoạn gold, làm giới hạn điểm theo tiêu chí khớp nguyên văn. Câu này vẫn được tính trong mẫu số.

Chọn **dense làm ứng viên theo kết quả retrieval dev**, theo thứ tự ưu tiên Evidence Hit@3, Hit@1 rồi thời gian. Cần đối chiếu trace agent trên dev trước khi chốt đánh giá test. Thời gian rerank của lượt này chịu ảnh hưởng bởi việc chạy agent đồng thời trên Jetson. Chi phí API là 0.
