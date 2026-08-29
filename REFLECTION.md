# Reflection Questions

## 1. Rewrite email trước routing

Nên interrupt **sau node generate email** (`interrupt_after`) vì nội dung email phải được tạo
xong thì reviewer mới có dữ liệu để sửa. Một biểu diễn tương đương là interrupt ngay trước
routing node tiếp theo, nhưng nếu phải chọn giữa `interrupt_before` và `interrupt_after` đối
với node generate email thì chọn `interrupt_after`.

Sau khi reviewer cập nhật nội dung trong checkpoint, graph mới được resume tới routing node.
Không interrupt trước generation vì khi đó chưa có email để review.

## 2. Giảm alert fatigue khi có 500 email confidence 0.82

Không hạ threshold chỉ để giảm số cảnh báo khi chưa có evaluation evidence. Thay vào đó:

1. Đưa các action `send_email` vào hàng đợi review theo batch, nhóm theo risk segment và cho
   phép bulk approve với mẫu email giống nhau.
2. Ưu tiên case theo churn probability, giá trị khách hàng và mức bất định; case rủi ro cao
   hiển thị trước.
3. Chạy candidate threshold trên cùng một dataset gắn nhãn, đo false-positive/false-negative
   bằng code deterministic rồi mới thay đổi gate.
4. Theo dõi queue depth, review latency, override rate và reviewer error để phát hiện mệt mỏi.

Con người vẫn phê duyệt các bước HITL; UI chỉ giảm thao tác lặp và giúp ưu tiên đúng case.

## 3. Vì sao không tin self-reported confidence của LLM

Confidence do LLM tự khai báo không phải xác suất đã được hiệu chỉnh. Model có thể rất tự tin
trong khi TOI sai, dữ liệu đầu vào thiếu hoặc reasoning không grounded. Nếu dùng trực tiếp điểm
này làm quyền thực thi, lỗi của model có thể vượt qua safety gate.

Cách calibration:

1. Tạo dataset cố định gồm TOI, churn probability, action đúng và nhãn risk do domain expert
   duyệt.
2. Chạy original và candidate trên cùng dataset.
3. Code deterministic đo accuracy, precision/recall cho high-risk, calibration error và độ đúng
   theo từng confidence bin.
4. Dùng Platt scaling hoặc isotonic regression trên validation set nếu đủ dữ liệu, sau đó đánh
   giá lại trên test set tách biệt.
5. Đối chiếu TOI với source-of-truth bằng code trước routing; không cho LLM tự xác nhận dữ liệu.
6. Hard policy như `increase_credit_limit -> human review` luôn override confidence đã calibrate.

LLM chỉ đề xuất. Metric và gate được tính bằng code deterministic; thay đổi policy cần con người
phê duyệt.
