MODULE MainModule

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
  
    CONST jointtarget home_joint_R:=[[0,-130,30,0,40,0],[-135, 9E9, 9E9, 9E9, 9E9, 9E9]];
    CONST robtarget Cap_pose:=[[307.57,-185.05,239.10],[0.650995,-0.758804,-0.0130378,-0.0158432],[1,0,1,4],[169.065,9E+09,9E+09,9E+09,9E+09,9E+09]];
    CONST robtarget p20:=[[301.72,-196.35,239.10],[0,0,0,0],[1,0,1,4],[169.066,9E+09,9E+09,9E+09,9E+09,9E+09]];
    
    VAR socketdev client;
   ! VAR string message;
    VAR rawbytes data;
    VAR string rcv_data;
    VAR string Center_Signal;
    
    
    
    VAR num flag;
    PERS num Action:=0;
    
    VAR bool bResult;

    PROC main()
      SocketCreate client;
      SocketConnect client, "192.168.125.5" ,65432;
      TPErase;
      TPWrite "Connection Successful";
      
   !   SocketAccept server,client, \Time:=WAIT_MAX;
      SocketSend client,\Str:=" hello server!";
      rcv_data := " ";
      
    WaitSyncTask sync01_1,task_All;   
    SyncMoveOn sync01_2,task_All;
    MoveAbsJ home_joint_R\ID:=101, v1000, z0, tool0;
 !   MoveJ home\ID:=101,v100,z50,tool0;
    SyncMoveOff sync01_3;
   
      WHILE rcv_data <> "x" DO
          TPWrite "Start wait _ _ _ _";
          SocketReceive client \Str:= rcv_data\Time:=1000;
         IF rcv_data ="home"  THEN
           MoveAbsJ home_joint_R, v1000, z0, tool0;
         ENDIF
         
        IF rcv_data ="Start"  THEN
            TPWrite "Go to the capture position";
            TPWrite "-----------------------";
            MoveJ [[163.25,-329.41,208.68],[0.0218428,-0.85309,-0.107468,-0.510109],[0,-1,1,4],[-100.882,9E+09,9E+09,9E+09,9E+09,9E+09]], v1000, z200, tool0;
            MoveJ [[181.20,-339.04,213.17],[0.675818,-0.596382,0.239381,-0.360963],[0,0,1,4],[-123.112,9E+09,9E+09,9E+09,9E+09,9E+09]], v1000, z200, tool0;
		    MoveJ Cap_pose, v1000, z0, tool0;
            
            
            WaitTime 1;
            SocketSend client,\Str:="32";
            TPWrite "Has send Trigger Signal to Cognex Cam";
            TPWrite "-----------------------";
            
            WaitTime 2; !Doi cam chup
            SocketSend client,\Str:="Done_Capture";
            TPWrite "Has send 'Done_Capture'";
            TPWrite "-----------------------";
            
            MoveAbsJ home_joint_R, v1000, z0, tool0;
            
            Center_Signal := " ";
            WHILE Center_Signal <> "x" DO
                
                TPWrite "Wait action_sig";
                TPWrite "-----------------------";
                SocketReceive client \Str:= rcv_data\Time:=1000;
 
                IF rcv_data = "Gio tay trái" THEN
                    Action :=1;
                ELSEIF rcv_data = "Gio tay phai" THEN
                    Action :=2;
                ELSEIF rcv_data = "Gio hai tay" THEN
                    Action :=3;
                ELSEIF rcv_data = "Dua tay trái" THEN
                    Action :=4;
                ENDIF
                
                WaitSyncTask sync01_4,task_All;
                
                IF Action=1 THEN
                    TPWrite "R_Arm do not action";
                ELSEIF Action=2 THEN
                    
                ELSEIF Action=3 THEN

                ELSEIF Action=4 THEN
                    
                ELSEIF Action=0 THEN
                    Center_Signal := "x";
                    TPWrite "No Action";
                ENDIF
                
            ENDWHILE
            
         ENDIF
         
         IF rcv_data ="move"  THEN
            MoveJ [[163.25,-329.41,208.68],[0.0218428,-0.85309,-0.107468,-0.510109],[0,-1,1,4],[-100.882,9E+09,9E+09,9E+09,9E+09,9E+09]], v1000, z200, tool0;
            MoveJ [[181.20,-339.04,213.17],[0.675818,-0.596382,0.239381,-0.360963],[0,0,1,4],[-123.112,9E+09,9E+09,9E+09,9E+09,9E+09]], v1000, z200, tool0;
		    MoveJ Cap_pose, v1000, z0, tool0;

         ENDIF

      ENDWHILE
      SocketClose client;
    ENDPROC
ENDMODULE


