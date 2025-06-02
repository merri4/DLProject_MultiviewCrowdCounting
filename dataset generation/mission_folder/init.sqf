
_camera = "camera" camCreate [0, 10, 25];         
_camera camSetFov 0.6;         
_camera camSetTarget [0, 10.1, 0];        
_camera cameraEffect ["Internal", "Back"];         
showCinemaBorder false;         
_camera camCommit 0;  


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
for "_i" from 0 to ((count _yOffsets) - 1) do {  
    private _yOffset = _yOffsets select _i;  
    private _zOffset = _zOffsets select _i;  

    { 
        private _type = _x select 0;  
        private _xPos = _x select 1;  
        private _yPos = (_x select 2) + _yOffset;  
        private _zPos = (_x select 3) + _zOffset;  
        private _dir = _x select 4;  

        private _obj = createVehicle [_type, [_xPos, _yPos, _zPos], [], 0, "CAN_COLLIDE"];
        _obj setDir _dir;  
    } forEach _baseObjects;  
};
agent_chair_0=createAgent["C_Man_casual_1_F",[-5.94,0.04,0.092],[],0,"CAN_COLLIDE"];
agent_chair_0 setDir 180;
agent_chair_0 switchAction "Crouch";
agent_chair_1=createAgent["C_journalist_F",[-5.23,0.04,0.092],[],0,"CAN_COLLIDE"];
agent_chair_1 setDir 180;
agent_chair_1 switchAction "Crouch";
agent_chair_2=createAgent["C_man_polo_2_F",[-4.50,0.02,0.092],[],0,"CAN_COLLIDE"];
agent_chair_2 setDir 180;
agent_chair_2 switchAction "Crouch";
agent_chair_3=createAgent["C_Man_casual_7_F",[-2.22,0.04,0.092],[],0,"CAN_COLLIDE"];
agent_chair_3 setDir 180;
agent_chair_3 switchAction "Crouch";
agent_chair_4=createAgent["C_man_polo_3_F",[-0.80,0.02,0.092],[],0,"CAN_COLLIDE"];
agent_chair_4 setDir 180;
agent_chair_4 switchAction "Crouch";
agent_chair_5=createAgent["C_man_sport_1_F",[0.01,0.01,0.092],[],0,"CAN_COLLIDE"];
agent_chair_5 setDir 180;
agent_chair_5 switchAction "Crouch";
agent_chair_6=createAgent["C_Man_casual_7_F",[2.24,0.02,0.092],[],0,"CAN_COLLIDE"];
agent_chair_6 setDir 180;
agent_chair_6 switchAction "Crouch";
agent_chair_7=createAgent["C_journalist_F",[4.45,0.01,0.092],[],0,"CAN_COLLIDE"];
agent_chair_7 setDir 180;
agent_chair_7 switchAction "Crouch";
agent_chair_8=createAgent["C_man_1",[5.18,0.01,0.092],[],0,"CAN_COLLIDE"];
agent_chair_8 setDir 180;
agent_chair_8 switchAction "Crouch";
agent_chair_9=createAgent["C_Man_casual_2_F",[-5.93,2.01,0.192],[],0,"CAN_COLLIDE"];
agent_chair_9 setDir 180;
agent_chair_9 switchAction "Crouch";
agent_chair_10=createAgent["C_Man_casual_2_F",[-2.23,2.03,0.192],[],0,"CAN_COLLIDE"];
agent_chair_10 setDir 180;
agent_chair_10 switchAction "Crouch";
agent_chair_11=createAgent["C_Man_casual_9_F",[-0.04,2.04,0.192],[],0,"CAN_COLLIDE"];
agent_chair_11 setDir 180;
agent_chair_11 switchAction "Crouch";
agent_chair_12=createAgent["C_man_sport_2_F",[1.45,2.05,0.192],[],0,"CAN_COLLIDE"];
agent_chair_12 setDir 180;
agent_chair_12 switchAction "Crouch";
agent_chair_13=createAgent["C_man_polo_5_F",[4.48,2.02,0.192],[],0,"CAN_COLLIDE"];
agent_chair_13 setDir 180;
agent_chair_13 switchAction "Crouch";
agent_chair_14=createAgent["C_journalist_F",[5.19,2.03,0.192],[],0,"CAN_COLLIDE"];
agent_chair_14 setDir 180;
agent_chair_14 switchAction "Crouch";
agent_chair_15=createAgent["C_man_polo_1_F",[5.95,2.01,0.192],[],0,"CAN_COLLIDE"];
agent_chair_15 setDir 180;
agent_chair_15 switchAction "Crouch";
agent_chair_16=createAgent["C_man_sport_3_F",[-5.18,4.01,0.292],[],0,"CAN_COLLIDE"];
agent_chair_16 setDir 180;
agent_chair_16 switchAction "Crouch";
agent_chair_17=createAgent["C_man_sport_1_F",[-4.45,4.00,0.292],[],0,"CAN_COLLIDE"];
agent_chair_17 setDir 180;
agent_chair_17 switchAction "Crouch";
agent_chair_18=createAgent["C_Man_casual_6_F",[-2.27,4.02,0.292],[],0,"CAN_COLLIDE"];
agent_chair_18 setDir 180;
agent_chair_18 switchAction "Crouch";
agent_chair_19=createAgent["C_Man_casual_4_v2_F",[-1.51,4.01,0.292],[],0,"CAN_COLLIDE"];
agent_chair_19 setDir 180;
agent_chair_19 switchAction "Crouch";
agent_chair_20=createAgent["C_Man_casual_6_v2_F",[-0.73,4.04,0.292],[],0,"CAN_COLLIDE"];
agent_chair_20 setDir 180;
agent_chair_20 switchAction "Crouch";
agent_chair_21=createAgent["C_Man_casual_1_F",[1.48,4.03,0.292],[],0,"CAN_COLLIDE"];
agent_chair_21 setDir 180;
agent_chair_21 switchAction "Crouch";
agent_chair_22=createAgent["C_Man_casual_3_F",[2.23,4.02,0.292],[],0,"CAN_COLLIDE"];
agent_chair_22 setDir 180;
agent_chair_22 switchAction "Crouch";
agent_chair_23=createAgent["C_man_polo_2_F",[4.50,4.04,0.292],[],0,"CAN_COLLIDE"];
agent_chair_23 setDir 180;
agent_chair_23 switchAction "Crouch";
agent_chair_24=createAgent["C_man_p_beggar_F",[5.19,4.03,0.292],[],0,"CAN_COLLIDE"];
agent_chair_24 setDir 180;
agent_chair_24 switchAction "Crouch";
agent_chair_25=createAgent["C_man_p_fugitive_F",[5.95,4.02,0.292],[],0,"CAN_COLLIDE"];
agent_chair_25 setDir 180;
agent_chair_25 switchAction "Crouch";
agent_chair_26=createAgent["C_Man_casual_5_v2_F",[-5.99,6.03,0.392],[],0,"CAN_COLLIDE"];
agent_chair_26 setDir 180;
agent_chair_26 switchAction "Crouch";
agent_chair_27=createAgent["C_man_polo_1_F",[-5.23,6.01,0.392],[],0,"CAN_COLLIDE"];
agent_chair_27 setDir 180;
agent_chair_27 switchAction "Crouch";
agent_chair_28=createAgent["C_Man_casual_5_F",[-4.46,6.00,0.392],[],0,"CAN_COLLIDE"];
agent_chair_28 setDir 180;
agent_chair_28 switchAction "Crouch";
agent_chair_29=createAgent["C_man_p_fugitive_F",[-1.45,6.05,0.392],[],0,"CAN_COLLIDE"];
agent_chair_29 setDir 180;
agent_chair_29 switchAction "Crouch";
agent_chair_30=createAgent["C_Man_casual_5_v2_F",[-0.78,6.05,0.392],[],0,"CAN_COLLIDE"];
agent_chair_30 setDir 180;
agent_chair_30 switchAction "Crouch";
agent_chair_31=createAgent["C_man_polo_2_F",[0.05,6.05,0.392],[],0,"CAN_COLLIDE"];
agent_chair_31 setDir 180;
agent_chair_31 switchAction "Crouch";
agent_chair_32=createAgent["C_Man_casual_5_v2_F",[0.77,6.02,0.392],[],0,"CAN_COLLIDE"];
agent_chair_32 setDir 180;
agent_chair_32 switchAction "Crouch";
agent_chair_33=createAgent["C_journalist_F",[1.55,6.01,0.392],[],0,"CAN_COLLIDE"];
agent_chair_33 setDir 180;
agent_chair_33 switchAction "Crouch";
agent_chair_34=createAgent["C_Man_casual_6_F",[2.22,6.02,0.392],[],0,"CAN_COLLIDE"];
agent_chair_34 setDir 180;
agent_chair_34 switchAction "Crouch";
agent_chair_35=createAgent["C_man_sport_1_F",[4.43,6.01,0.392],[],0,"CAN_COLLIDE"];
agent_chair_35 setDir 180;
agent_chair_35 switchAction "Crouch";
agent_chair_36=createAgent["C_Man_casual_5_v2_F",[5.22,6.04,0.392],[],0,"CAN_COLLIDE"];
agent_chair_36 setDir 180;
agent_chair_36 switchAction "Crouch";
agent_chair_37=createAgent["C_man_1",[5.97,6.05,0.392],[],0,"CAN_COLLIDE"];
agent_chair_37 setDir 180;
agent_chair_37 switchAction "Crouch";
agent_chair_38=createAgent["C_man_p_fugitive_F",[-5.16,8.02,0.492],[],0,"CAN_COLLIDE"];
agent_chair_38 setDir 180;
agent_chair_38 switchAction "Crouch";
agent_chair_39=createAgent["C_man_p_beggar_F",[-4.45,8.02,0.492],[],0,"CAN_COLLIDE"];
agent_chair_39 setDir 180;
agent_chair_39 switchAction "Crouch";
agent_chair_40=createAgent["C_Man_casual_5_F",[-2.30,8.04,0.492],[],0,"CAN_COLLIDE"];
agent_chair_40 setDir 180;
agent_chair_40 switchAction "Crouch";
agent_chair_41=createAgent["C_Man_casual_1_F",[-1.48,8.00,0.492],[],0,"CAN_COLLIDE"];
agent_chair_41 setDir 180;
agent_chair_41 switchAction "Crouch";
agent_chair_42=createAgent["C_man_polo_6_F",[-0.80,8.03,0.492],[],0,"CAN_COLLIDE"];
agent_chair_42 setDir 180;
agent_chair_42 switchAction "Crouch";
agent_chair_43=createAgent["C_Man_casual_1_F",[0.03,8.01,0.492],[],0,"CAN_COLLIDE"];
agent_chair_43 setDir 180;
agent_chair_43 switchAction "Crouch";
agent_chair_44=createAgent["C_Man_casual_5_v2_F",[0.77,8.03,0.492],[],0,"CAN_COLLIDE"];
agent_chair_44 setDir 180;
agent_chair_44 switchAction "Crouch";
agent_chair_45=createAgent["C_Man_casual_4_v2_F",[1.50,8.04,0.492],[],0,"CAN_COLLIDE"];
agent_chair_45 setDir 180;
agent_chair_45 switchAction "Crouch";
agent_chair_46=createAgent["C_man_sport_3_F",[4.47,8.02,0.492],[],0,"CAN_COLLIDE"];
agent_chair_46 setDir 180;
agent_chair_46 switchAction "Crouch";
agent_chair_47=createAgent["C_Man_casual_4_v2_F",[5.17,8.01,0.492],[],0,"CAN_COLLIDE"];
agent_chair_47 setDir 180;
agent_chair_47 switchAction "Crouch";
agent_chair_48=createAgent["C_Man_casual_8_F",[6.00,8.04,0.492],[],0,"CAN_COLLIDE"];
agent_chair_48 setDir 180;
agent_chair_48 switchAction "Crouch";
agent_chair_49=createAgent["C_man_sport_2_F",[-5.94,10.05,0.592],[],0,"CAN_COLLIDE"];
agent_chair_49 setDir 180;
agent_chair_49 switchAction "Crouch";
agent_chair_50=createAgent["C_Man_casual_9_F",[-4.43,10.02,0.592],[],0,"CAN_COLLIDE"];
agent_chair_50 setDir 180;
agent_chair_50 switchAction "Crouch";
agent_chair_51=createAgent["C_Man_casual_8_F",[-2.21,10.00,0.592],[],0,"CAN_COLLIDE"];
agent_chair_51 setDir 180;
agent_chair_51 switchAction "Crouch";
agent_chair_52=createAgent["C_Man_casual_6_v2_F",[-1.48,10.04,0.592],[],0,"CAN_COLLIDE"];
agent_chair_52 setDir 180;
agent_chair_52 switchAction "Crouch";
agent_chair_53=createAgent["C_Man_casual_4_v2_F",[-0.02,10.01,0.592],[],0,"CAN_COLLIDE"];
agent_chair_53 setDir 180;
agent_chair_53 switchAction "Crouch";
agent_chair_54=createAgent["C_Man_casual_8_F",[0.76,10.02,0.592],[],0,"CAN_COLLIDE"];
agent_chair_54 setDir 180;
agent_chair_54 switchAction "Crouch";
agent_chair_55=createAgent["C_man_sport_1_F",[1.51,10.04,0.592],[],0,"CAN_COLLIDE"];
agent_chair_55 setDir 180;
agent_chair_55 switchAction "Crouch";
agent_chair_56=createAgent["C_Man_casual_5_F",[2.27,10.04,0.592],[],0,"CAN_COLLIDE"];
agent_chair_56 setDir 180;
agent_chair_56 switchAction "Crouch";
agent_chair_57=createAgent["C_Man_casual_3_F",[5.19,10.01,0.592],[],0,"CAN_COLLIDE"];
agent_chair_57 setDir 180;
agent_chair_57 switchAction "Crouch";
agent_chair_58=createAgent["C_man_polo_3_F",[5.95,10.04,0.592],[],0,"CAN_COLLIDE"];
agent_chair_58 setDir 180;
agent_chair_58 switchAction "Crouch";
agent_chair_59=createAgent["C_Man_casual_2_F",[-5.97,12.52,0.992],[],0,"CAN_COLLIDE"];
agent_chair_59 setDir 180;
agent_chair_59 switchAction "Crouch";
agent_chair_60=createAgent["C_man_sport_3_F",[-5.22,12.53,0.992],[],0,"CAN_COLLIDE"];
agent_chair_60 setDir 180;
agent_chair_60 switchAction "Crouch";
agent_chair_61=createAgent["C_man_polo_3_F",[-4.44,12.53,0.992],[],0,"CAN_COLLIDE"];
agent_chair_61 setDir 180;
agent_chair_61 switchAction "Crouch";
agent_chair_62=createAgent["C_man_sport_2_F",[-2.26,12.54,0.992],[],0,"CAN_COLLIDE"];
agent_chair_62 setDir 180;
agent_chair_62 switchAction "Crouch";
agent_chair_63=createAgent["C_Man_casual_5_F",[-1.51,12.52,0.992],[],0,"CAN_COLLIDE"];
agent_chair_63 setDir 180;
agent_chair_63 switchAction "Crouch";
agent_chair_64=createAgent["C_Man_Messenger_01_F",[-0.79,12.51,0.992],[],0,"CAN_COLLIDE"];
agent_chair_64 setDir 180;
agent_chair_64 switchAction "Crouch";
agent_chair_65=createAgent["C_man_polo_6_F",[0.02,12.52,0.992],[],0,"CAN_COLLIDE"];
agent_chair_65 setDir 180;
agent_chair_65 switchAction "Crouch";
agent_chair_66=createAgent["C_man_hunter_1_F",[1.53,12.50,0.992],[],0,"CAN_COLLIDE"];
agent_chair_66 setDir 180;
agent_chair_66 switchAction "Crouch";
agent_chair_67=createAgent["C_Man_casual_5_F",[4.48,12.53,0.992],[],0,"CAN_COLLIDE"];
agent_chair_67 setDir 180;
agent_chair_67 switchAction "Crouch";
agent_chair_68=createAgent["C_man_sport_3_F",[5.19,12.52,0.992],[],0,"CAN_COLLIDE"];
agent_chair_68 setDir 180;
agent_chair_68 switchAction "Crouch";
agent_chair_69=createAgent["C_man_polo_5_F",[5.93,12.53,0.992],[],0,"CAN_COLLIDE"];
agent_chair_69 setDir 180;
agent_chair_69 switchAction "Crouch";
agent_chair_70=createAgent["C_Man_casual_1_F",[-5.92,14.53,1.192],[],0,"CAN_COLLIDE"];
agent_chair_70 setDir 180;
agent_chair_70 switchAction "Crouch";
agent_chair_71=createAgent["C_Man_casual_5_F",[-4.47,14.50,1.192],[],0,"CAN_COLLIDE"];
agent_chair_71 setDir 180;
agent_chair_71 switchAction "Crouch";
agent_chair_72=createAgent["C_Man_casual_5_F",[-2.29,14.53,1.192],[],0,"CAN_COLLIDE"];
agent_chair_72 setDir 180;
agent_chair_72 switchAction "Crouch";
agent_chair_73=createAgent["C_Man_Messenger_01_F",[-1.45,14.53,1.192],[],0,"CAN_COLLIDE"];
agent_chair_73 setDir 180;
agent_chair_73 switchAction "Crouch";
agent_chair_74=createAgent["C_man_sport_2_F",[-0.78,14.51,1.192],[],0,"CAN_COLLIDE"];
agent_chair_74 setDir 180;
agent_chair_74 switchAction "Crouch";
agent_chair_75=createAgent["C_journalist_F",[0.76,14.52,1.192],[],0,"CAN_COLLIDE"];
agent_chair_75 setDir 180;
agent_chair_75 switchAction "Crouch";
agent_chair_76=createAgent["C_man_hunter_1_F",[1.50,14.52,1.192],[],0,"CAN_COLLIDE"];
agent_chair_76 setDir 180;
agent_chair_76 switchAction "Crouch";
agent_chair_77=createAgent["C_man_polo_2_F",[2.21,14.52,1.192],[],0,"CAN_COLLIDE"];
agent_chair_77 setDir 180;
agent_chair_77 switchAction "Crouch";
agent_chair_78=createAgent["C_man_1",[4.50,14.54,1.192],[],0,"CAN_COLLIDE"];
agent_chair_78 setDir 180;
agent_chair_78 switchAction "Crouch";
agent_chair_79=createAgent["C_man_hunter_1_F",[5.15,14.52,1.192],[],0,"CAN_COLLIDE"];
agent_chair_79 setDir 180;
agent_chair_79 switchAction "Crouch";
agent_chair_80=createAgent["C_Man_casual_5_F",[5.91,14.52,1.192],[],0,"CAN_COLLIDE"];
agent_chair_80 setDir 180;
agent_chair_80 switchAction "Crouch";
agent_chair_81=createAgent["C_Man_casual_9_F",[-5.98,16.52,1.392],[],0,"CAN_COLLIDE"];
agent_chair_81 setDir 180;
agent_chair_81 switchAction "Crouch";
agent_chair_82=createAgent["C_man_polo_6_F",[-5.23,16.52,1.392],[],0,"CAN_COLLIDE"];
agent_chair_82 setDir 180;
agent_chair_82 switchAction "Crouch";
agent_chair_83=createAgent["C_Man_casual_2_F",[-4.43,16.54,1.392],[],0,"CAN_COLLIDE"];
agent_chair_83 setDir 180;
agent_chair_83 switchAction "Crouch";
agent_chair_84=createAgent["C_Man_casual_7_F",[-2.25,16.52,1.392],[],0,"CAN_COLLIDE"];
agent_chair_84 setDir 180;
agent_chair_84 switchAction "Crouch";
agent_chair_85=createAgent["C_man_polo_3_F",[0.02,16.53,1.392],[],0,"CAN_COLLIDE"];
agent_chair_85 setDir 180;
agent_chair_85 switchAction "Crouch";
agent_chair_86=createAgent["C_man_1",[0.79,16.55,1.392],[],0,"CAN_COLLIDE"];
agent_chair_86 setDir 180;
agent_chair_86 switchAction "Crouch";
agent_chair_87=createAgent["C_Man_casual_2_F",[1.54,16.50,1.392],[],0,"CAN_COLLIDE"];
agent_chair_87 setDir 180;
agent_chair_87 switchAction "Crouch";
agent_chair_88=createAgent["C_man_sport_2_F",[2.26,16.54,1.392],[],0,"CAN_COLLIDE"];
agent_chair_88 setDir 180;
agent_chair_88 switchAction "Crouch";
agent_chair_89=createAgent["C_man_sport_1_F",[4.48,16.52,1.392],[],0,"CAN_COLLIDE"];
agent_chair_89 setDir 180;
agent_chair_89 switchAction "Crouch";
agent_chair_90=createAgent["C_Man_casual_3_F",[5.92,16.52,1.392],[],0,"CAN_COLLIDE"];
agent_chair_90 setDir 180;
agent_chair_90 switchAction "Crouch";
agent_passage_0=createAgent["C_man_polo_4_F",[3.32,2.00,2],[],0,"CAN_COLLIDE"];
agent_passage_0 setDir 108.78;
agent_passage_1=createAgent["C_man_polo_5_F",[-3.36,3.00,2],[],0,"CAN_COLLIDE"];
agent_passage_1 setDir 229.39;
agent_passage_2=createAgent["C_man_hunter_1_F",[-6.05,11.30,2],[],0,"CAN_COLLIDE"];
agent_passage_2 setDir 181.68;
agent_passage_3=createAgent["C_Man_casual_2_F",[-5.05,11.26,2],[],0,"CAN_COLLIDE"];
agent_passage_3 setDir 106.66;
agent_passage_4=createAgent["C_man_polo_1_F",[-2.05,11.30,2],[],0,"CAN_COLLIDE"];
agent_passage_4 setDir 1.80;
agent_passage_5=createAgent["C_man_p_fugitive_F",[6.95,11.26,2],[],0,"CAN_COLLIDE"];
agent_passage_5 setDir 248.67;
agent_passage_6=createAgent["C_Man_Messenger_01_F",[-5.05,18.01,2],[],0,"CAN_COLLIDE"];
agent_passage_6 setDir 33.56;
agent_passage_7=createAgent["C_journalist_F",[-1.05,18.00,2],[],0,"CAN_COLLIDE"];
agent_passage_7 setDir 316.74;
agent_passage_8=createAgent["C_Man_casual_7_F",[2.95,18.02,2],[],0,"CAN_COLLIDE"];
agent_passage_8 setDir 246.11;
agent_passage_9=createAgent["C_Man_casual_5_v2_F",[3.95,18.02,2],[],0,"CAN_COLLIDE"];
agent_passage_9 setDir 194.80;

_beforeTime = systemTime;
t = 0;
_currentTime = systemTime;
_beforeTime = _currentTime;
while {t < 5} do {
    _currentTime = systemTime;
    if ((_currentTime select 5) isEqualTo (_beforeTime select 5)) then {	
    } 
    else {
        t = t+1;
        _beforeTime = _currentTime;
    };
};


_camera = "camera" camCreate [5.50, -8.00, 2.0];         
_camera camSetFov 0.6;         
_camera camSetTarget [-6.17, 16.52, 1.72];        
_camera cameraEffect ["Internal", "Back"];         
showCinemaBorder false;         
_camera camCommit 0;  


pos = agent_chair_0 modelToWorldVisual (agent_chair_0 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_1 modelToWorldVisual (agent_chair_1 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_2 modelToWorldVisual (agent_chair_2 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_3 modelToWorldVisual (agent_chair_3 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_4 modelToWorldVisual (agent_chair_4 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_5 modelToWorldVisual (agent_chair_5 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_6 modelToWorldVisual (agent_chair_6 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_7 modelToWorldVisual (agent_chair_7 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_8 modelToWorldVisual (agent_chair_8 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_9 modelToWorldVisual (agent_chair_9 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_10 modelToWorldVisual (agent_chair_10 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_11 modelToWorldVisual (agent_chair_11 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_12 modelToWorldVisual (agent_chair_12 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_13 modelToWorldVisual (agent_chair_13 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_14 modelToWorldVisual (agent_chair_14 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_15 modelToWorldVisual (agent_chair_15 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_16 modelToWorldVisual (agent_chair_16 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_17 modelToWorldVisual (agent_chair_17 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_18 modelToWorldVisual (agent_chair_18 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_19 modelToWorldVisual (agent_chair_19 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_20 modelToWorldVisual (agent_chair_20 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_21 modelToWorldVisual (agent_chair_21 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_22 modelToWorldVisual (agent_chair_22 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_23 modelToWorldVisual (agent_chair_23 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_24 modelToWorldVisual (agent_chair_24 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_25 modelToWorldVisual (agent_chair_25 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_26 modelToWorldVisual (agent_chair_26 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_27 modelToWorldVisual (agent_chair_27 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_28 modelToWorldVisual (agent_chair_28 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_29 modelToWorldVisual (agent_chair_29 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_30 modelToWorldVisual (agent_chair_30 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_31 modelToWorldVisual (agent_chair_31 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_32 modelToWorldVisual (agent_chair_32 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_33 modelToWorldVisual (agent_chair_33 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_34 modelToWorldVisual (agent_chair_34 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_35 modelToWorldVisual (agent_chair_35 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_36 modelToWorldVisual (agent_chair_36 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_37 modelToWorldVisual (agent_chair_37 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_38 modelToWorldVisual (agent_chair_38 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_39 modelToWorldVisual (agent_chair_39 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_40 modelToWorldVisual (agent_chair_40 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_41 modelToWorldVisual (agent_chair_41 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_42 modelToWorldVisual (agent_chair_42 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_43 modelToWorldVisual (agent_chair_43 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_44 modelToWorldVisual (agent_chair_44 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_45 modelToWorldVisual (agent_chair_45 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_46 modelToWorldVisual (agent_chair_46 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_47 modelToWorldVisual (agent_chair_47 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_48 modelToWorldVisual (agent_chair_48 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_49 modelToWorldVisual (agent_chair_49 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_50 modelToWorldVisual (agent_chair_50 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_51 modelToWorldVisual (agent_chair_51 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_52 modelToWorldVisual (agent_chair_52 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_53 modelToWorldVisual (agent_chair_53 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_54 modelToWorldVisual (agent_chair_54 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_55 modelToWorldVisual (agent_chair_55 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_56 modelToWorldVisual (agent_chair_56 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_57 modelToWorldVisual (agent_chair_57 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_58 modelToWorldVisual (agent_chair_58 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_59 modelToWorldVisual (agent_chair_59 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_60 modelToWorldVisual (agent_chair_60 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_61 modelToWorldVisual (agent_chair_61 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_62 modelToWorldVisual (agent_chair_62 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_63 modelToWorldVisual (agent_chair_63 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_64 modelToWorldVisual (agent_chair_64 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_65 modelToWorldVisual (agent_chair_65 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_66 modelToWorldVisual (agent_chair_66 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_67 modelToWorldVisual (agent_chair_67 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_68 modelToWorldVisual (agent_chair_68 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_69 modelToWorldVisual (agent_chair_69 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_70 modelToWorldVisual (agent_chair_70 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_71 modelToWorldVisual (agent_chair_71 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_72 modelToWorldVisual (agent_chair_72 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_73 modelToWorldVisual (agent_chair_73 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_74 modelToWorldVisual (agent_chair_74 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_75 modelToWorldVisual (agent_chair_75 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_76 modelToWorldVisual (agent_chair_76 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_77 modelToWorldVisual (agent_chair_77 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_78 modelToWorldVisual (agent_chair_78 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_79 modelToWorldVisual (agent_chair_79 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_80 modelToWorldVisual (agent_chair_80 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_81 modelToWorldVisual (agent_chair_81 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_82 modelToWorldVisual (agent_chair_82 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_83 modelToWorldVisual (agent_chair_83 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_84 modelToWorldVisual (agent_chair_84 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_85 modelToWorldVisual (agent_chair_85 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_86 modelToWorldVisual (agent_chair_86 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_87 modelToWorldVisual (agent_chair_87 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_88 modelToWorldVisual (agent_chair_88 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_89 modelToWorldVisual (agent_chair_89 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_chair_90 modelToWorldVisual (agent_chair_90 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_passage_0 modelToWorldVisual (agent_passage_0 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_passage_1 modelToWorldVisual (agent_passage_1 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_passage_2 modelToWorldVisual (agent_passage_2 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_passage_3 modelToWorldVisual (agent_passage_3 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_passage_4 modelToWorldVisual (agent_passage_4 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_passage_5 modelToWorldVisual (agent_passage_5 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_passage_6 modelToWorldVisual (agent_passage_6 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_passage_7 modelToWorldVisual (agent_passage_7 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_passage_8 modelToWorldVisual (agent_passage_8 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


pos = agent_passage_9 modelToWorldVisual (agent_passage_9 selectionPosition ["head", "ViewGeometry", "AveragePoint"]);
headpoint pushBack pos; 


["write", ["", "hi", ""]] call inidbi_checkpoint;


t = 0;
_currentTime = systemTime;
_beforeTime = _currentTime;
while {t < 10} do {
    _currentTime = systemTime;
    if ((_currentTime select 5) isEqualTo (_beforeTime select 5)) then {	
    } 
    else {
        t = t+1;
        _beforeTime = _currentTime;
    };
};


_camera = "camera" camCreate [-5.11, -8.45, 2.16];         
_camera camSetFov 0.6;         
_camera camSetTarget [6.31, 16.59, 0.23];        
_camera cameraEffect ["Internal", "Back"];         
showCinemaBorder false;         
_camera camCommit 0;  

for "_i" from 0 to (count headpoint - 1) do {
    private _chunkData = headpoint select _i;
    private _chunkString = str(_chunkData);
    ["write", ["", format["%1", _i], _chunkString]] call inidbi_headpoint;
};


["write", ["", "hi", ""]] call inidbi_checkpoint1;

