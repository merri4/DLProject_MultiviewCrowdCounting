import sys
import os
import numpy as np
import csv
import random
import pandas as pd

agent_classes = [
    "C_man_p_beggar_F", "C_man_1", "C_Man_casual_1_F", "C_Man_casual_2_F",
    "C_Man_casual_3_F", "C_Man_casual_4_v2_F", "C_Man_casual_5_v2_F", "C_Man_casual_6_v2_F",
    "C_Man_casual_7_F", "C_Man_casual_8_F", "C_Man_casual_9_F", "C_man_sport_1_F",
    "C_man_sport_2_F", "C_man_sport_3_F", "C_Man_casual_4_F", "C_Man_casual_5_F",
    "C_Man_casual_6_F", "C_man_polo_1_F", "C_man_polo_2_F", "C_man_polo_3_F",
    "C_man_polo_4_F", "C_man_polo_5_F", "C_man_polo_6_F", "C_Man_Fisherman_01_F",
    "C_man_p_fugitive_F", "C_man_hunter_1_F", "C_journalist_F", "C_Man_Messenger_01_F"
]


def random_spaced_points(min_val=0.0, max_val=16.5, min_gap=1.0):
    # 최대 생성 가능한 개수
    max_count = int((max_val - min_val) // min_gap) + 1
    count = random.randint(0, 4)  # 랜덤한 개수 선택

    # 가능한 위치 후보
    candidates = [round(min_val + i * min_gap, 2) for i in range(max_count)]

    # 랜덤하게 count만큼 선택 (정렬된 상태 유지)
    selected = sorted(random.sample(candidates, count))
    return selected


sqf = []

sqf.append(f'''
_camera = "camera" camCreate [0, 10, 25];         
_camera camSetFov 0.6;         
_camera camSetTarget [0, 10.1, 0];        
_camera cameraEffect ["Internal", "Back"];         
showCinemaBorder false;         
_camera camCommit 0;  
''')

sqf.append(f'''
inidbi_headpoint = ["new", "inidbi_headpoint"] call OO_INIDBI;
inidbi_checkpoint = ["new", "inidbi_checkpoint"] call OO_INIDBI;
inidbi_checkpoint1 = ["new", "inidbi_checkpoint1"] call OO_INIDBI;
missionNamespace setVariable ["headpoint", []];
private _yOffsets = [0, 2, 4, 6, 8, 10, 12.5, 14.5, 16.5];
private _zOffsets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 1.1, 1.3];

private _baseObjects = [   
    ["Land_ConcretePanels_02_single_v2_F", -7, 0, 0, 90],  
    ["Land_ConcretePanels_02_single_v2_F", -5, 0, 0, 90],  
    ["Land_ConcretePanels_02_single_v2_F", -3, 0, 0, 90],  
    ["Land_ConcretePanels_02_single_v2_F", -1, 0, 0, 90],  
    ["Land_ConcretePanels_02_single_v2_F", 1, 0, 0, 90],  
    ["Land_ConcretePanels_02_single_v2_F", 3, 0, 0, 90],  
    ["Land_ConcretePanels_02_single_v2_F", 5, 0, 0, 90],  
    ["Land_ConcretePanels_02_single_v2_F", 7, 0, 0, 90],  
    ["Land_ConcretePanels_02_single_v2_F", -7, 1.5, 0, 90],  
    ["Land_ConcretePanels_02_single_v2_F", -5, 1.5, 0, 90],  
    ["Land_ConcretePanels_02_single_v2_F", -3, 1.5, 0, 90],  
    ["Land_ConcretePanels_02_single_v2_F", -1, 1.5, 0, 90],  
    ["Land_ConcretePanels_02_single_v2_F", 1, 1.5, 0, 90],  
    ["Land_ConcretePanels_02_single_v2_F", 3, 1.5, 0, 90],  
    ["Land_ConcretePanels_02_single_v2_F", 5, 1.5, 0, 90],  
    ["Land_ConcretePanels_02_single_v2_F", 7, 1.5, 0, 90],  

    // 중간 책상  
    ["Land_WallSign_01_chalkboard_F", -2, -1, 0.092, 180],  
    ["Land_WallSign_01_chalkboard_F", -1, -1, 0.092, 180],  
    ["Land_WallSign_01_chalkboard_F", 0, -1, 0.092, 180],  
    ["Land_WallSign_01_chalkboard_F", 1, -1, 0.092, 180],  
    ["Land_WallSign_01_chalkboard_F", 2, -1, 0.092, 180],  

    // 중간 책상 측면  
    ["Land_WallSign_01_chalkboard_F", -2.9, -0.1, 0.092, 270],  
    ["Land_WallSign_01_chalkboard_F", 2.9, -0.1, 0.092, 90],  

    // 가운데 줄 의자  
    ["Land_CampingChair_V2_F", -2.25, 0, 0.092, 0],  
    ["Land_CampingChair_V2_F", -1.5, 0, 0.092, 0],  
    ["Land_CampingChair_V2_F", -0.75, 0, 0.092, 0],  
    ["Land_CampingChair_V2_F", 0, 0, 0.092, 0],  
    ["Land_CampingChair_V2_F", 0.75, 0, 0.092, 0],  
    ["Land_CampingChair_V2_F", 1.5, 0, 0.092, 0],  
    ["Land_CampingChair_V2_F", 2.25, 0, 0.092, 0],  

    // 왼쪽 책상  
    ["Land_WallSign_01_chalkboard_F", -4.7, -1, 0.092, 180],  
    ["Land_WallSign_01_chalkboard_F", -5.7, -1, 0.092, 180],  
    ["Land_WallSign_01_chalkboard_F", -3.8, -0.1, 0.092, 90],  
    ["Land_WallSign_01_chalkboard_F", -6.6, -0.1, 0.092, 270],  

    // 왼쪽 의자 추가  
    ["Land_CampingChair_V2_F", -4.45, 0, 0.092, 0],  
    ["Land_CampingChair_V2_F", -5.2, 0, 0.092, 0],  
    ["Land_CampingChair_V2_F", -5.95, 0, 0.092, 0],  

   // 오른쪽 책상  
    ["Land_WallSign_01_chalkboard_F", 4.7, -1, 0.092, 180],  
    ["Land_WallSign_01_chalkboard_F", 5.7, -1, 0.092, 180],  
    ["Land_WallSign_01_chalkboard_F", 3.8, -0.1, 0.092, 270],  
    ["Land_WallSign_01_chalkboard_F", 6.6, -0.1, 0.092, 90],  

    // 오른쪽 의자 추가  
    ["Land_CampingChair_V2_F", 4.45, 0, 0.092, 0],  
    ["Land_CampingChair_V2_F", 5.2, 0, 0.092, 0],  
    ["Land_CampingChair_V2_F", 5.95, 0, 0.092, 0]  
];  

// 반복 배치  
for "_i" from 0 to ((count _yOffsets) - 1) do {{  
    private _yOffset = _yOffsets select _i;  
    private _zOffset = _zOffsets select _i;  

    {{ 
        private _type = _x select 0;  
        private _xPos = _x select 1;  
        private _yPos = (_x select 2) + _yOffset;  
        private _zPos = (_x select 3) + _zOffset;  
        private _dir = _x select 4;  

        private _obj = createVehicle [_type, [_xPos, _yPos, _zPos], [], 0, "CAN_COLLIDE"];
        _obj setDir _dir;  
    }} forEach _baseObjects;  
}};''')

df = pd.read_csv("Chair_Positions_CSV.csv")
chair = 0
for i, row in df.iterrows():
    if random.random() < 0.8:
        x, y, z = row["x"], row["y"], row["z"]
        agent_class = random.choice(agent_classes)
        sqf.append(
            f'agent_chair_{chair}=createAgent["{agent_class}",[{x + random.uniform(-0.05, 0.05):.2f},{y + random.uniform(0, 0.05):.2f},{z:.3f}],[],0,"CAN_COLLIDE"];')
        sqf.append(f'agent_chair_{chair} setDir 180;')
        sqf.append(f'agent_chair_{chair} switchAction "Crouch";')
        chair += 1

passage = 0
points = random_spaced_points()
for i in range(len(points)):
    agent_class = random.choice(agent_classes)
    sqf.append(
        f'agent_passage_{passage}=createAgent["{agent_class}",[{3.35 + random.uniform(-0.05, 0.05):.2f},{points[i]:.2f},2],[],0,"CAN_COLLIDE"];')
    sqf.append(f'agent_passage_{passage} setDir {random.uniform(0, 360):.2f};')
    passage += 1
points = random_spaced_points()
for i in range(len(points)):
    agent_class = random.choice(agent_classes)
    sqf.append(
        f'agent_passage_{passage}=createAgent["{agent_class}",[{-3.35 + random.uniform(-0.05, 0.05):.2f},{points[i]:.2f},2],[],0,"CAN_COLLIDE"];')
    sqf.append(f'agent_passage_{passage} setDir {random.uniform(0, 360):.2f};')
    passage += 1

points = random_spaced_points(min_val=-7.05, max_val=+7.05)
for i in range(len(points)):
    agent_class = random.choice(agent_classes)
    sqf.append(
        f'agent_passage_{passage}=createAgent["{agent_class}",[{points[i]:.2f},{11.25 + random.uniform(-0.05, 0.05):.2f},2],[],0,"CAN_COLLIDE"];')
    sqf.append(f'agent_passage_{passage} setDir {random.uniform(0, 360):.2f};')
    passage += 1
points = random_spaced_points(min_val=-7.05, max_val=+7.05)
for i in range(len(points)):
    agent_class = random.choice(agent_classes)
    sqf.append(
        f'agent_passage_{passage}=createAgent["{agent_class}",[{points[i]:.2f},{18 + random.uniform(-0.05, 0.05):.2f},2],[],0,"CAN_COLLIDE"];')
    sqf.append(f'agent_passage_{passage} setDir {random.uniform(0, 360):.2f};')
    passage += 1

sqf.append(f'''
_beforeTime = systemTime;
t = 0;
_currentTime = systemTime;
_beforeTime = _currentTime;
while {{t < 5}} do {{
    _currentTime = systemTime;
    if ((_currentTime select 5) isEqualTo (_beforeTime select 5)) then {{	
    }} 
    else {{
        t = t+1;
        _beforeTime = _currentTime;
    }};
}};
''')

sqf.append(f'''
_camera = "camera" camCreate [{5.5:.2f}, {-8:.2f}, {2.0}];         
_camera camSetFov 0.6;         
_camera camSetTarget [{-6.25 + random.uniform(-0.1, 0.1):.2f}, {16.5 + random.uniform(-0.1, 0.1):.2f}, {2.7 + random.uniform(-2.7, 0.1):.2f}];        
_camera cameraEffect ["Internal", "Back"];         
showCinemaBorder false;         
_camera camCommit 0;  
''')

for i in range(chair):
    sqf.append(f'''
pos = agent_chair_{i} modelToWorldVisual (agent_chair_{i} selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 
''')
for i in range(passage):
    sqf.append(f'''
pos = agent_passage_{i} modelToWorldVisual (agent_passage_{i} selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 
''')

sqf.append(f'''
["write", ["", "hi", ""]] call inidbi_checkpoint;
''')

sqf.append(f'''
t = 0;
_currentTime = systemTime;
_beforeTime = _currentTime;
while {{t < 10}} do {{
    _currentTime = systemTime;
    if ((_currentTime select 5) isEqualTo (_beforeTime select 5)) then {{	
    }} 
    else {{
        t = t+1;
        _beforeTime = _currentTime;
    }};
}};
''')

sqf.append(f'''
_camera = "camera" camCreate [{-5.5 + random.uniform(-0.5, 0.5):.2f}, {-8 + random.uniform(-0.5, 0.5):.2f}, {2.0 + random.uniform(-0.2, 0.2):.2f}];         
_camera camSetFov 0.6;         
_camera camSetTarget [{6.25 + random.uniform(-0.1, 0.1):.2f}, {16.5 + random.uniform(-0.1, 0.1):.2f}, {2.7 + random.uniform(-2.7, 0.1):.2f}];        
_camera cameraEffect ["Internal", "Back"];         
showCinemaBorder false;         
_camera camCommit 0;  

for "_i" from 0 to (count headpoint - 1) do {{
    private _chunkData = headpoint select _i;
    private _chunkString = str(_chunkData);
    ["write", ["", format["%1", _i], _chunkString]] call inidbi_headpoint;
}};
''')

sqf.append(f'''
["write", ["", "hi", ""]] call inidbi_checkpoint1;
''')

path = 'mission_folder/init.sqf'
with open(path, "w") as file:
    for string in sqf:
        file.write(string + "\n")