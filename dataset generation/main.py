import subprocess
import os
import time
import cv2
import numpy as np
import pandas as pd
from mss import mss
import win32gui
import re
from shutil import copyfile

# 설정
ARMA3_PATH = r"C:\Program Files (x86)\Steam\steamapps\common\Arma 3"
ARMA3_EXE = os.path.join(ARMA3_PATH, "arma3_x64.exe")
INIDBI_PATH = os.path.join(ARMA3_PATH, "!Workshop/@INIDBI2 - Official extension/db")
MISSION_NAME = "ex0.VR"
MISSION_ROOT = os.path.join(ARMA3_PATH, "Missions")
MISSION_PATH = os.path.join(MISSION_ROOT, MISSION_NAME)
WINDOW_PATTERN = r"^Arma 3"

# 폴더 준비
os.makedirs("screenshot", exist_ok=True)
os.makedirs("headpoint_ini", exist_ok=True)
os.makedirs("headpoint_csv", exist_ok=True)

def clear_inidbi_folder():
    for f in os.listdir(INIDBI_PATH):
        if f.endswith(".ini"):
            os.remove(os.path.join(INIDBI_PATH, f))

def get_arma_window_bbox():
    def _callback(hwnd, result):
        title = win32gui.GetWindowText(hwnd)
        if re.match(WINDOW_PATTERN, title):
            result.append(hwnd)
    result = []
    win32gui.EnumWindows(_callback, result)
    if not result:
        return None
    hwnd = result[0]
    rect = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    right, bottom = win32gui.ClientToScreen(hwnd, (rect[2], rect[3]))
    return {'top': top, 'left': left, 'width': right - left, 'height': bottom - top}

def take_screenshot(path):
    bbox = get_arma_window_bbox()
    if bbox is None:
        print("❌ Arma 3 window not found.")
        return
    with mss() as sct:
        img = np.array(sct.grab(bbox))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(path, img)

def convert_ini_to_csv(ini_path, csv_path):
    data = []
    with open(ini_path, "r", encoding="utf-8") as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.strip().split("=")
            try:
                coords = eval(value.strip('"'))  # e.g. "[1.2, 3.4, 5.6]"
                if isinstance(coords, list) and len(coords) == 3:
                    x, y = coords[0], coords[1]
                    if not (x == 0 and y == 0):
                        data.append([x, y])
            except:
                continue
    df = pd.DataFrame(data, columns=["x", "y"])
    df.to_csv(csv_path, index=False)

def prepare_mission_folder():
    subprocess.run(["python", "sqf_generation.py"], check=True)
    if os.path.exists(MISSION_PATH):
        for f in os.listdir(MISSION_PATH):
            os.remove(os.path.join(MISSION_PATH, f))
    else:
        os.makedirs(MISSION_PATH, exist_ok=True)
    for fname in os.listdir("mission_folder"):
        src = os.path.join("mission_folder", fname)
        dst = os.path.join(MISSION_PATH, fname)
        copyfile(src, dst)

def main(iteration):
    print(f"🔁 Iteration {iteration}")
    clear_inidbi_folder()
    prepare_mission_folder()

    print("🚀 Launch Arma 3...")
    params = f'-skipIntro -noSplash -window -mod="{ARMA3_PATH}\\!Workshop\\@INIDBI2 - Official extension"'
    init = f'-init=playMission["","{MISSION_NAME}",true]'
    proc = subprocess.Popen([ARMA3_EXE, params + " " + init])

    while "inidbi_checkpoint.ini" not in os.listdir(INIDBI_PATH):
        time.sleep(0.1)
    time.sleep(5)
    take_screenshot(f"screenshot/right_{iteration}.png")

    while "inidbi_checkpoint1.ini" not in os.listdir(INIDBI_PATH):
        time.sleep(0.1)
    time.sleep(2)
    take_screenshot(f"screenshot/left_{iteration}.png")

    ini_file_src = os.path.join(INIDBI_PATH, "inidbi_headpoint.ini")
    ini_file_dst = os.path.join("headpoint_ini", f"headpoint_{iteration}.ini")
    csv_file = os.path.join("headpoint_csv", f"headpoint_{iteration}.csv")
    if os.path.exists(ini_file_src):
        copyfile(ini_file_src, ini_file_dst)
        convert_ini_to_csv(ini_file_src, csv_file)

    print("🛑 Killing Arma 3 process")
    proc.kill()

if __name__ == "__main__":
    for i in range(100000):  # 반복 횟수 조절 가능
        main(i+8923)
        #4990
        #7086
        #8923

        #0, 4990, 7086, 8923
