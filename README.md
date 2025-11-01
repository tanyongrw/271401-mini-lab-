# 271401 AI mini lab
MG400 block สีเขียว 31/10/25

นาย จิรภัทร เชื้อเมืองพาน 650610830

นางสาว พีรยา ใจหล้า 650610849

นางสาว รุจยา วิทยาประภาากร 650610855


# How to use
ต่อกล้องและ MG400 ให้เรียบร้อย

วาง block ทั้ง 4 บนพื้นตาราง ให้กล้องสามารถมองเห็น

run calibrate.py ปรับ ค่า HSV ของแต่ละสีให้มองเห็นสีนั้น ๆ ได้

ปรับ distorsion ให้เส้นตารางตรงและขนานกัน

เซฟไฟล์ hsv_config.json และ camera_points ที่โปรแกรม return ออกมาไว้

เปิด MG400 python ใน Dobot studio

jog แขนกล ไปหาแต่ละ block แล้วเซฟพิกัดไว้

แก้ไฟล์ camera_points และ world_points ในไฟล์ main.py

กด start หุ่นยนต์ใน dobot studio

run main.py

เลือกสีที่จะให้กล้อง track แล้วกด start

หุ่นยนต์จะหยิบ block ที่เป็นสีที่เลือกไว้ออกไปทีละอัน


# Calibrate (HSV Color & distortion Tuner)
สามารถใช้ slider ปรับค่าสี(HSV) , ค่า distortion , kernel ของ Morphological Operation ได้

บันทึกค่าที่ปรับ ไว้ในไฟล์ hsv_config.json

โปรแกรมจะ return ตำแหน่งของแต่ละสีใน camera frame ออกมา

# Main (Vision and Control)
อ่านภาพจากกล้อง และปรับตามไฟล์ที่จูนมา

มี slider ให้เลือกสีที่จะตรวจจับ

แปลงพิกัดจาก cameara frame เป็นพิกัดใน real-world frame

ส่งข้อมูลให้แขนกล

# MG400 python (dobot studio)
สื่อสารกับโปรแกรม python และรอรับค่าพิกัดจากโปรแกรม

ขยับแขนกลไปตามพิกัดที่ได้รับมา
