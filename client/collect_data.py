'''
此代码用于采集图像和轨迹数据并保存为RLDS格式的hdf5后缀文件中
'''

import cv2
import time
import os
import numpy as np
import h5py
from datetime import datetime
from piper_sdk import *

# 设置基本参数配置
CAMERA_ID = 2
CAN_PORT = "can0"
CAPTURE_RES = (640, 480)
MODEL_RES = (224, 224)
FPS = 10 
SAVE_DIR = "dataset_hdf5"
GRIPPER_THRESHOLD = 3000 

PIPER_raw2deg = 0.001
DEG_TO_RAD = 3.1415926535 / 180.0


class DataCollector:
    def __init__(self):
        self.recording = False
        self.exit_flag = False
        self.buffer = []
        self.instruction = ""
        
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_RES[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_RES[1])
        print(f"摄像头 (ID={CAMERA_ID})")

        
        self.piper = C_PiperInterface_V2(CAN_PORT)
        self.piper.ConnectPort()
        print(f"连接机械臂 ({CAN_PORT})")

        while not self.piper.EnablePiper():
            time.sleep(0.01)
        
        # 回到零点
        print("机械臂回零")
        factor = 57295.7795  # 1000*180/3.1415926
        # 零点位置1
        # position = [0.017, 0.171, -0.611, 0.027, 0.846, -0.762, 1]
        # 零点位置2
        position = [0.547, 1.258, -1.552, 0.003, 1.315, -0.576, 1]

        joint_0 = round(position[0] * factor)
        joint_1 = round(position[1] * factor)
        joint_2 = round(position[2] * factor)
        joint_3 = round(position[3] * factor)
        joint_4 = round(position[4] * factor)
        joint_5 = round(position[5] * factor)
        joint_6 = round(position[6] * 1000 * 1000)
        
        self.piper.ModeCtrl(0x01, 0x01, 30, 0x00)
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

        time.sleep(3.0)
    
        self.piper.MotionCtrl_2(0x02, 0x00, 0x00)  # 示教模式
        
        time.sleep(1.0)  # 等一秒让数据稳定
        
        # 读取原始值
        first_msg = self.piper.GetArmGripperMsgs().gripper_state
        self.init_gripper_val = first_msg.grippers_angle

    def get_robot_state(self):
        j_msg = self.piper.GetArmJointMsgs().joint_state
        g_msg = self.piper.GetArmGripperMsgs().gripper_state
        
        joints = [
            j_msg.joint_1 * PIPER_raw2deg * DEG_TO_RAD,
            j_msg.joint_2 * PIPER_raw2deg * DEG_TO_RAD,
            j_msg.joint_3 * PIPER_raw2deg * DEG_TO_RAD,
            j_msg.joint_4 * PIPER_raw2deg * DEG_TO_RAD,
            j_msg.joint_5 * PIPER_raw2deg * DEG_TO_RAD,
            j_msg.joint_6 * PIPER_raw2deg * DEG_TO_RAD,
        ]

        current_raw = g_msg.grippers_angle
        
        if current_raw >= (self.init_gripper_val - GRIPPER_THRESHOLD):
            gripper_binary = 1.0  # 张开
        else:
            gripper_binary = 0.0  # 闭合/正在抓取

        return np.array(joints + [gripper_binary], dtype=np.float32)

    def capture_step(self):
        ret, frame = self.cap.read()
        if not ret: return None, None

        frame = cv2.flip(frame, -1) 

        img_model = cv2.resize(frame, MODEL_RES)
        img_rgb = cv2.cvtColor(img_model, cv2.COLOR_BGR2RGB)
        
        state_vector = self.get_robot_state()

        step_data = {
            'image': img_rgb,       
            'state': state_vector,  
            'state_ref': state_vector,  # 用于计算 Action
            'timestamp': time.time()
        }
        return step_data, frame

    def save_episode(self, is_success):
        if len(self.buffer) < 2: return

        final_reward = 1.0 if is_success else -1.0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status_str = "SUCCESS" if is_success else "FAIL"
        filename = f"{SAVE_DIR}/ep_{status_str}_{timestamp}_{len(self.buffer)}.hdf5"

        all_states = np.array([x['state_ref'] for x in self.buffer], dtype=np.float32)  # (N, 7)
        all_images = np.array([x['image'] for x in self.buffer], dtype=np.uint8)        # (N, 224, 224, 3)
        
        num_steps = len(all_states)
        actions = np.zeros_like(all_states)  

        actions[:-1, :6] = all_states[1:, :6] - all_states[:-1, :6]
        actions[:-1, 6] = all_states[1:, 6]
        
        actions[-1, :6] = 0.0
        actions[-1, 6] = all_states[-1, 6]

        with h5py.File(filename, 'w') as f:
            f.attrs['language_instruction'] = self.instruction
            f.attrs['reward'] = final_reward
            f.attrs['sim'] = False

            f.create_dataset('action', data=actions)
            
            obs_group = f.create_group('observations')
            obs_group.create_dataset('images', data=all_images, compression='gzip')
            obs_group.create_dataset('state', data=all_states)

        print(f"\n[{status_str}] 已保存 HDF5: {filename}")
        self.buffer = []
    
    # 开始采集
    def run(self):
        print("\n" + "="*60)
        print("Piper 数据采集")
        print("  [S] 开始录制")
        print("  [Y/N] 结束录制 (成功/失败)")
        print("="*60)
        
        self.instruction = input("请输入任务指令: ").strip()
        interval = 1.0 / FPS
        
        try:
            while not self.exit_flag:
                start_time = time.time()
                
                step_data, display_frame = self.capture_step()
                if step_data is None: continue

                # 获取当前的夹爪状态
                curr_g_val = step_data['state'][6]
                
                # 显示夹爪状态
                if curr_g_val > 0.5:
                    g_text = "Gripper: OPEN (1.0)"
                    g_color = (0, 255, 0)  # 绿
                else:
                    g_text = "Gripper: CLOSED (0.0)"
                    g_color = (0, 0, 255)  # 红

                cv2.putText(display_frame, g_text, (30, 430), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, g_color, 2)

                if self.recording:
                    self.buffer.append(step_data)
                    cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1)
                    cv2.putText(display_frame, f"REC {len(self.buffer)}", (50, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, "IDLE", (30, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Collector', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.exit_flag = True
                elif key == ord('s') and not self.recording:
                    print("\n>>> 开始录制...")
                    self.recording = True
                    self.buffer = []
                elif key == ord('y') and self.recording:
                    print("<<< 成功 (+1)")
                    self.recording = False
                    self.save_episode(True)
                    '''
                    # 回到零点
                    print("机械臂回零")
                    factor = 57295.7795  # 1000*180/3.1415926
                    # 零点位置1
                    # position = [0.017, 0.171, -0.611, 0.027, 0.846, -0.762, 1]
                    # 零点位置2
                    position = [0.547, 1.258, -1.552, 0.003, 1.315, -0.576, 1]

                    joint_0 = round(position[0] * factor)
                    joint_1 = round(position[1] * factor)
                    joint_2 = round(position[2] * factor)
                    joint_3 = round(position[3] * factor)
                    joint_4 = round(position[4] * factor)
                    joint_5 = round(position[5] * factor)
                    joint_6 = round(position[6] * 1000 * 1000)
        
                    self.piper.ModeCtrl(0x01, 0x01, 30, 0x00)
                    self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
                    self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

                    time.sleep(3.0)
                    '''
                elif key == ord('n') and self.recording:
                    print("<<< 失败 (-1)")
                    self.recording = False
                    self.save_episode(False)

                elapsed = time.time() - start_time
                if elapsed < interval:
                    time.sleep(interval - elapsed)

        finally:
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    DataCollector().run()
