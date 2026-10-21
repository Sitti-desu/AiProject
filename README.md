# Thai Sign AI Tool (SIGNAL)
> **พลิกการสื่อสารภาษามือไทย ด้วย Computer Vision และ Machine Learning แบบเรียลไทม์**  
> 🌐 **Live Demo:** [https://thaisignaitool.onrender.com/](https://thaisignaitool.onrender.com/)

---

## 📌 ที่มาและความสำคัญ (Problem & Pain Points)

- **วิกฤตขาดแคลนล่ามภาษามือในไทย:** ประเทศไทยมีความต้องการล่ามภาษามือในระบบมากถึง **40,000 คน** แต่ปัจจุบันมีล่ามที่ขึ้นทะเบียนอย่างเป็นทางการเพียง **178 คน** เท่านั้น (ช่องว่างระดับร้อยเท่า) ทำให้ผู้พิการทางการได้ยินเข้าถึงบริการขั้นพื้นฐานได้ยากลำบาก
- **ข้อจำกัดของศูนย์บริการล่ามถ่ายทอดการสื่อสาร (TTRS):** ในช่วงเวลาที่ศูนย์ TTRS ปิดทำการ ผู้พิการทางการได้ยินแทบจะไม่สามารถติดต่อสื่อสารหรือทำธุรกรรมเร่งด่วนด้วยตัวเองได้เลย กลายเป็นอุปสรรคใหญ่ในการใช้ชีวิตประจำวัน
- **สอดคล้องกับนโยบายภาครัฐ (พม.):** กระทรวงการพัฒนาสังคมและความมั่นคงของมนุษย์ (พม.) กำลังเร่งแก้ไขปัญหานี้ โดยมีนโยบายสนับสนุนการนำเทคโนโลยี AI และนวัตกรรมดิจิทัลเข้ามาช่วยเสริมระบบบริการล่าม ซึ่งเปิดโอกาสให้เรานำนวัตกรรมนี้เข้ามาตอบโจทย์สังคมได้ทันท่วงที

---

## 💡 แนวคิดและทางออก (Solution)

**Thai Sign AI Tool** ถูกพัฒนาขึ้นเพื่อทำหน้าที่เป็น **"ล่ามดิจิทัลผ่านเว็บแอปพลิเคชัน"** ช่วยแปลท่าทางภาษามือไทย (Static Hand Shapes) ออกมาเป็นข้อความบนหน้าจอได้สะดวกรวดเร็ว:
- **Zero-Install & Hardware-Free:** ใช้งานผ่านเว็บเบราว์เซอร์ได้ทันที ไม่ต้องติดตั้งโปรแกรมเพิ่มเติม และใช้เพียงกล้องเว็บแคมทั่วไปหรือกล้องสมาร์ตโฟน ไม่ต้องพึ่งพาเซนเซอร์ราคาแพง
- **Real-time Recognition:** ประมวลผลและแปลผลลัพธ์สดผ่านระบบ WebSocket พร้อมระบบจัดประโยคอัตโนมัติ
- **Self-Service Training Studio:** มีระบบเก็บข้อมูลท่าทางใหม่และสั่งเทรนโมเดลผ่านหน้าเว็บได้โดยตรง

---

## 🛠️ ข้อมูลเชิงเทคนิค (Technical Architecture)

### 1. Computer Vision & Feature Extraction
- **MediaPipe Hands:** ตรวจจับและสกัดพิกัดข้อต่อนิ้วมือแบบ 3 มิติ ($x, y, z$) สูงสุด 2 ข้างพร้อมกัน (21 ข้อต่อ/มือ)
- **Feature Vector (126 Dimensions):** รวมพิกัด $21 \times 3 \times 2 = 126$ มิติ พร้อมระบบ **Zero-padding** กรณีทำท่ามือเดียว เพื่อให้ขนาดเวกเตอร์คงที่ 126 ค่าเสมอ

### 2. Machine Learning Model
- **Classifier:** ใช้อัลกอริทึม **Random Forest Classifier** จาก `scikit-learn` เพื่อจำแนกประเภทท่าทาง (Multi-class Classification)
- **Temporal Smoothing & Stability Filter:**
  - ใช้ Sliding Window ขนาด 10 เฟรม (`SMOOTHING_WINDOW = 10`)
  - กรองความเชื่อมั่นขั้นต่ำที่ 55% (`CONF_THRESHOLD = 0.55`)
  - ท่าทางจะถูกบันทึกเป็นคำศัพท์ลงประโยคก็ต่อเมื่อผลทำนายมีความนิ่งเกิน 70% (`STABLE_RATIO = 0.7`) และไม่ใช่คลาส `no_sign` เพื่อป้องกันคำเด้งรัว

### 3. Backend & Frontend Architecture
- **Backend:** ขับเคลื่อนด้วย **FastAPI** และ **Uvicorn** รองรับการสื่อสารแบบสองทางผ่าน **WebSocket (`/ws/predict`)** สำหรับสตรีมมิ่งภาพสด และ RESTful APIs (`/api/collect`, `/api/train`, `/api/status`)
- **Frontend:** พัฒนาด้วย **Vanilla JS (SPA)** เบา โหลดเร็ว ไม่พึ่งพาเฟรมเวิร์กขนาดใหญ่ พร้อมระบบวาด Hand Skeleton สดบนหน้าจอด้วย HTML5 Canvas
- **Deployment & Keep-Alive:** ติดตั้งบน Render.com (Python 3.11) และมอนิเตอร์สถานะผ่าน UptimeRobot ทุก 5 นาทีเพื่อป้องกันระบบเข้าสู่โหมด Sleep

---

## 🖥️ คู่มือการใช้งานระบบ (User Guide)

ระบบแบ่งออกเป็น 2 สตูดิโอหลัก:

### 1. Live Recognition Studio (โหมดแปลภาษามือสด)
1. เปิดหน้าเว็บและเลือกแท็บ **Live Studio**
2. กดปุ่ม **Start recognition** และกดอนุญาตให้เบราว์เซอร์เข้าถึงกล้องเว็บแคม
3. ยกมือทำท่าทางภาษามือในกรอบกล้อง:
   - **Hand Status:** แสดงสถานะการตรวจจับมือ (`Hand detected` / `No hand in view`)
   - **Current Recognition:** แสดงคำศัพท์ที่ทำนายได้พร้อมแถบเปอร์เซ็นต์ความมั่นใจ
   - **Transcript Box:** รวบรวมคำศัพท์ที่นิ่งแล้วเรียงต่อกันเป็นประโยคอัตโนมัติ
4. กดปุ่ม **Copy** เพื่อคัดลอกข้อความประโยคไปใช้งาน หรือกด **Clear** เพื่อล้างประโยคใหม่

### 2. Train & Collect Studio (โหมดเก็บข้อมูลและเทรนโมเดล)
1. สลับไปที่แท็บ **Train Studio** แล้วกดปุ่ม **Open Camera**
2. ปรับระยะมือจนกระทั่งเห็น **เส้นโครงกระดูกสีฟ้าและจุดสีชมพู** ปรากฏบนมือ
3. **การเก็บข้อมูลตัวอย่างท่าทาง (Data Collection):**
   - พิมพ์ชื่อท่าที่ต้องการบันทึกในช่อง **Sign Label** (เช่น `hello`, `love`, `thank_you`, `no_sign`)
   - กดปุ่ม **Record Samples (20 วินาที)**
   - รอระบบนับถอยหลัง `3... 2... 1...` กลางหน้าจอ จากนั้นทำท่าทางค้างไว้ในกล้องจนกว่าแถบ `REC` จะบันทึกครบตามเวลา
4. **การสั่งเทรนโมเดล (Model Training):**
   - เมื่อเก็บข้อมูลท่าทางครบตามต้องการ ให้เลื่อนมาที่หัวข้อ **2. Model Training**
   - กดปุ่ม **Start Real Training** เพื่อให้ระบบนำข้อมูลใน `data/gestures.csv` ไปเทรนโมเดลใหม่
   - เมื่อระบบแจ้งเตือนว่าเทรนสำเร็จ สามารถสลับกลับไปทดสอบท่าใหม่ใน **Live Studio** ได้ทันที

---

## 💻 วิธีการรันบนเครื่องคอมพิวเตอร์ (Local Setup)

```bash
# 1. Clone repository
git clone [https://github.com/your-username/sign-language-recognition.git](https://github.com/your-username/sign-language-recognition.git)
cd sign-language-recognition

# 2. สร้างและเปิดใช้งาน Virtual Environment (แนะนำ Python 3.11)
python -m venv .venv
# สำหรับ Windows:
.venv\Scripts\activate
# สำหรับ macOS/Linux:
source .venv/bin/activate

# 3. ติดตั้ง Dependencies ทั้งหมด
pip install -r requirements.txt

# 4. สตาร์ทเซิร์ฟเวอร์ทดสอบ
uvicorn backend.main:app --reload --port 8000
