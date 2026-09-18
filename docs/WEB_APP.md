# เว็บแอปวิเคราะห์วิดีโอ

เว็บแอปรับวิดีโอฟุตบอลหนึ่งไฟล์ แล้วรัน detection/tracking, สร้าง review
video และ heatmap ตามลำดับ งาน inference ถูกล็อกให้ประมวลผลครั้งละหนึ่งงาน
เพราะ TrackLab เปลี่ยน working directory ระหว่างทำงาน

## รันในเครื่อง

ใช้ environment inference ที่ติดตั้งครบแล้ว:

```powershell
.venv\Scripts\python.exe tools\web_app.py
```

เปิด <http://127.0.0.1:8001> ได้เลย หรือกำหนดพอร์ตเองด้วย
`$env:PORT = '8001'` หรือ `--port 8001`.

ตัวแปรเลือกที่เก็บข้อมูลได้:

| ตัวแปร | ค่าเริ่มต้น | หน้าที่ |
| --- | --- | --- |
| `WEB_JOBS_DIR` | `outputs/web_jobs` | วิดีโออัปโหลด, output และ log ของแต่ละงาน |
| `MODEL_DIR` | `pretrained_models` | checkpoint ที่ pipeline ดาวน์โหลด/ใช้งาน |
| `PORT` | `8001` | พอร์ต HTTP |

รองรับ MP4, MOV, AVI และ MKV สูงสุด 2 GB แต่ควรเริ่มจากคลิปสั้นมากก่อน:
pipeline แตกวิดีโอเป็นเฟรม JPEG และเก็บทั้งไฟล์ต้นฉบับ, เฟรม, output และ
review video จึงใช้พื้นที่มากกว่าขนาดไฟล์อัปโหลดหลายเท่า

## Deploy บน Render

ใน root ของ repository มี [`render.yaml`](../render.yaml) สำหรับ Blueprint ของ
Render แล้ว โดยเป็น Free demo service แบบหนึ่ง instance และใช้ temporary
storage:

```text
WEB_JOBS_DIR=/tmp/sn-gamestate/web_jobs
MODEL_DIR=/tmp/sn-gamestate/pretrained_models
MAX_UPLOAD_BYTES=104857600
```

ขั้นตอน:

1. push `render.yaml`, `tools/web_app.py` และเอกสารนี้ไปยัง Git repository
   เดียวกับโปรเจกต์
2. ใน Render เลือก **New + > Blueprint** แล้วเลือก repository/branch นั้น
3. ตรวจสอบชื่อบริการและเลือก region ที่ใกล้ผู้ใช้ จากนั้นสร้าง service
4. Blueprint ใช้ Render Free และจำกัดไฟล์อัปโหลดไว้ 100 MB
5. รอ build เสร็จ แล้วเปิด URL ของ Render; health check ที่ `/` ต้องตอบ 200
6. ส่งคลิปสั้นโดยเลือก `MVP` และ `CPU` ก่อน เมื่อโมเดลถูกดาวน์โหลดครั้งแรก
   ระบบจะเก็บไว้ใน temporary storage ของ instance นั้น

`render.yaml` pin `PYTHON_VERSION=3.9.25`, ใช้ `uv sync --frozen --no-dev` เพื่อ
ยึด dependency ตาม `uv.lock` และติดตั้ง MMCV ซึ่งเป็น dependency แยกของ pipeline. หากขั้น build ของ MMCV
หรือ `torchreid` ล้ม ให้ตรวจ build log ก่อน: โครงการนี้ pin Python 3.9,
Torch 1.13.1 และ dependency CV รุ่นเก่า จึงต้องทดสอบบน Linux ของ Render
จริงก่อนเปิดให้ผู้ใช้ใช้งาน

Render กำหนด `PORT` ให้ service และแอป bind ที่ `0.0.0.0` โดยอัตโนมัติผ่าน
`tools/web_app.py`; ไม่ต้องตั้ง `PORT` เองใน Dashboard.

### ข้อจำกัดสำคัญ

- Blueprint นี้เป็น CPU service; เลือก `CPU` หรือ `Auto` ในหน้าเว็บ อย่าเลือก
  `CUDA` เว้นแต่ย้ายไปยัง environment ที่มี CUDA และ Torch build ที่ตรงกัน
- Free service ใช้ filesystem ชั่วคราว: restart, deploy หรือ instance ถูกหยุด
  จะลบวิดีโอ, output และ model checkpoints ทั้งหมด จึงต้องดาวน์โหลดโมเดลใหม่
  ในงานถัดไป
- Free plan เหมาะกับ demo คลิปสั้นเท่านั้น ไม่รับประกันว่า dependency หนัก ๆ
  จะ build สำเร็จหรือมี RAM/CPU เพียงพอสำหรับ inference
- web app ไม่มี authentication, rate limit หรือ malware scanning. อย่าเปิดเป็น
  public upload endpoint จนกว่าจะเพิ่มชั้นป้องกันเหล่านี้
- การประมวลผลมี CPU/RAM/disk สูงและนานกว่าคำขอ HTTP ปกติ; Render เหมาะสำหรับ
  การทดลองคลิปสั้นแบบต่อคิว ไม่ใช่ production video-processing queue
