MODULE MainModule
    CONST robtarget HomeL:=[[-8.88,181.45,197.44],[0.0562826,0.840162,-0.130281,0.523438],[0,0,0,4],[102.093,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Pre_HomeL:=[[26.39,428.32,242.51],[0.0562577,0.840114,-0.130248,0.523526],[0,0,0,4],[102.097,9E+09,9E+09,9E+09,9E+09,9E+09]];


    VAR syncident syncHome;

    PERS tasks task_All{2}:=[["T_ROB_L"],["T_ROB_R"]];

    VAR syncident sync01_0;
    VAR syncident sync01_1;
    VAR syncident sync01_2;
    VAR syncident sync01_3;
    VAR syncident sync01_4;
    VAR syncident sync01_5;
    VAR syncident sync01_6;
    VAR syncident sync01_7;
    VAR syncident sync01_8;
    VAR syncident sync01_9;

    VAR syncident sync01_10;
    VAR syncident sync01_11;
    VAR syncident sync01_12;
    VAR syncident sync01_13;
    VAR syncident sync01_14;
    VAR syncident sync01_15;
    VAR syncident sync01_16;
    VAR syncident sync01_17;
    VAR syncident sync01_18;
    VAR syncident sync01_19;
    
    CONST jointtarget home_joint_L:=[[0,-130,30,0,40,0],[135, 9E9, 9E9, 9E9, 9E9, 9E9]];
    CONST robtarget Pick_L:=[[271.93,-184.27,142.13],[0.286082,0.650509,0.271824,-0.648928],[-2,0,2,4],[-128.536,9E+09,9E+09,9E+09,9E+09,9E+09]];

    PERS num Action;

    VAR num offset:=15;
    VAR num height_obj:=10;
    CONST robtarget Pre_pick_L10:=[[282.13,121.61,157.44],[0.28607,0.650506,0.271828,-0.648934],[-1,0,2,4],[-128.534,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Home3:=[[356.22,303.96,157.45],[0.515342,0.489375,0.496558,-0.498362],[-1,0,2,4],[-160.819,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget PlaceRec:=[[312.34,467.09,121.65],[0.515395,0.489274,0.496624,-0.498342],[-1,0,2,4],[-165.444,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget PlaceCir:=[[312.34,392.96,121.65],[0.515394,0.489274,0.496623,-0.498343],[-1,0,2,4],[-165.443,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Home2:=[[175.42,303.96,223.02],[0.515339,0.489367,0.496556,-0.498375],[0,0,2,4],[133.861,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget Home1:=[[105.22,347.61,237.14],[0.0384974,-0.729155,0.612553,-0.302704],[0,1,0,4],[106.395,9E+09,9E+09,9E+09,9E+09,9E+09]];
    
    PROC main()

        WaitSyncTask sync01_1,task_All;
        SyncMoveOn sync01_2,task_All;
       ! MoveJ HomeL\ID:=101,v1000,z50,tool0;
        MoveAbsJ home_joint_L\ID:=101, v1000, z0, tool0;
        SyncMoveOff sync01_3;
  
        WHILE TRUE do
        WaitSyncTask sync01_4,task_All;
            IF Action=1 THEN
                
            ELSEIF Action=2 THEN
                
            ELSEIF Action=3 THEN

            ELSEIF Action=4 THEN
                
            ELSEIF Action=0 THEN
                Center_Signal := "x";
                TPWrite "No Action";
            ENDIF
        
        ENDWHILE
        
    ENDPROC
ENDMODULE