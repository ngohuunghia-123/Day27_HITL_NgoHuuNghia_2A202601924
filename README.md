# Lab 27 — LangGraph Human-in-the-Loop

Workflow đánh giá churn risk sử dụng Gemini để **đề xuất** action, sau đó dùng hard policy
và confidence gate viết bằng Python để quyết định auto-execute hoặc chuyển cho con người.
Các action trong lab chỉ được mô phỏng, không gửi email hoặc thay đổi hạn mức thật.

## Kiến trúc

```text
Customer data
    -> Gemini proposal (action, confidence, reasoning)
    -> deterministic routing + hard policy
       -> low-risk/high confidence: auto-execute
       -> high-risk/low confidence: interrupt before execution
    -> Streamlit Approve / Reject / Edit
    -> resume graph
    -> audit_log.json
```

Model mặc định:

- Primary: `gemini-3.5-flash-lite`
- Fallback: `gemini-3.1-flash-lite`

Fallback chỉ chuyển sang model Gemini thứ hai. Hệ thống không dùng mock khi API lỗi. Nếu cả
hai model thất bại, workflow dừng an toàn và không ghi audit giả.

## Yêu cầu

- Python 3.13
- Google Gemini API key

## Cài đặt trên Windows PowerShell

Tạo virtual environment bằng đúng Python 3.13:

```powershell
$python313 = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
& $python313 --version
& $python313 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Lệnh kiểm tra phiên bản phải trả về `Python 3.13.x`.

Tạo file cấu hình local:

```powershell
Copy-Item .env.example .env
notepad .env
```

Điền key vào `.env`:

```dotenv
GOOGLE_API_KEY=your_real_api_key
GEMINI_PRIMARY_MODEL=gemini-3.5-flash-lite
GEMINI_FALLBACK_MODEL=gemini-3.1-flash-lite
```

Không commit `.env`. File này đã được liệt kê trong `.gitignore`.

## Chạy LangGraph workflow

Chạy một workflow từ CLI:

```powershell
.\.venv\Scripts\python.exe graph.py --customer-id CUST001 --toi 50000000 --churn 0.75
```

Nếu action cần review, CLI trả trạng thái `pending_human_review`. Dùng Streamlit để thực hiện
Approve, Reject hoặc Edit.

## Chạy Streamlit UI

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

1. Nhập Customer ID, Total Operating Income, churn probability và Reviewer ID.
2. Chọn **Đánh giá khách hàng**.
3. Nếu graph dừng tại high-risk action, chọn:
   - **Approve**: thực hiện action được đề xuất.
   - **Reject**: hủy action.
   - **Edit**: chọn action hợp lệ mới rồi thực hiện dưới quyền phê duyệt của reviewer.
4. Nếu cả hai model lỗi, chọn **Retry** để tạo run mới với cùng input hoặc **Cancel**.

Compiled graph và `thread_id` được giữ trong `st.session_state` để Streamlit rerun không làm
mất pending state trong cùng process.

## Routing policy

Confidence threshold: `0.85`.

| Action | Confidence | Route |
|---|---:|---|
| `increase_credit_limit` | Bất kỳ | Human review — hard policy override |
| `send_email` | `>= 0.85` | Auto-execute |
| `send_email` | `< 0.85` | Human review |

Hard policy được kiểm tra trước confidence. Vì vậy `increase_credit_limit` với confidence
`0.99` vẫn không được auto-execute.

## Persistent state và audit

- LangGraph dùng `MemorySaver` và
  `interrupt_before=["execute_high_risk_action"]`.
- Mọi lần invoke/resume phải dùng cùng `thread_id`.
- `MemorySaver` giữ checkpoint trong process; restart app sẽ mất pending checkpoint.
- Audit được lưu tại `audit_log.json` và không bị overwrite khi thêm decision mới.
- Low-risk auto-execute dùng `reviewer_id="system"`, `decision="auto_execute"`.
- Approve, Reject và Edit lưu reviewer thực tế.

## Kiểm tra

Automated tests mock Gemini boundary nên không sử dụng API key hoặc phát sinh API cost:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy models.py graph.py audit.py app.py
```

Live smoke test cần `.env` chứa key thật:

```powershell
.\.venv\Scripts\python.exe graph.py --customer-id LIVE001 --toi 50000000 --churn 0.75
```

## Bảo mật

- Customer data được đánh dấu là untrusted data trong prompt.
- Gemini chỉ đề xuất; code deterministic quyết định routing.
- API key không xuất hiện trong source, audit hoặc structured log.
- Thiếu/sai key và lỗi cả hai model đều fail closed.
- Không commit API key, access token, password, private key hoặc `.env` thật.
