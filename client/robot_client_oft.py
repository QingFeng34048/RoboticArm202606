# 安全限位 — OFT Action-Chunk 客户端
import cv2
import time
import json
import requests
import numpy as np
import threading
from piper_sdk import *

# ================= 配置区域 =================
SERVER_URL = "http://localhost:8001/act"

# 硬件配置
CAMERA_ID = 1
CAN_PORT = "can0"

# 图像配置
CAPTURE_RES = (640, 480)
SEND_RES = (224, 224)

# --- 核心换算系数 ---
RAD_TO_SDK_INT = 57295.7795

GRIPPER_OPEN_SDK = 80000
GRIPPER_CLOSE_SDK = 0
GRIPPER_THRESHOLD_RAW = 3000

CONTROL_FREQ = 10

# --- OFT Action Chunk 配置 ---
# 是否使用服务端返回的完整 action chunk（多步开环执行）
# 设为 False 则退化为和原来一样只用第一个 action
USE_ACTION_CHUNK = True
# ===========================================


class RobotClient:
    def __init__(self):
        self.step_count = 0

        # 1. 初始化摄像头
        print(f"正在打开摄像头 (ID={CAMERA_ID})...")
        self.cap = cv2.VideoCapture(CAMERA_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_RES[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_RES[1])
        if not self.cap.isOpened():
            raise Exception("无法打开摄像头")

        # 后台线程：专门负责清空摄像头底层缓存，保证拿到的永远是最新帧
        self.latest_frame = None
        self.camera_lock = threading.Lock()
        self.is_running = True
        self.camera_thread = threading.Thread(target=self._update_camera_frame, daemon=True)
        self.camera_thread.start()

        while self.latest_frame is None:
            time.sleep(0.01)

        # 2. 初始化机械臂
        print(f"正在连接机械臂 ({CAN_PORT})...")
        self.piper = C_PiperInterface_V2(CAN_PORT)
        self.piper.ConnectPort()

        self.enable_and_reset()
        self.last_target_joints = None

        print(">>> 系统初始化完成，等待指令 <<<")

    def _update_camera_frame(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.camera_lock:
                    self.latest_frame = frame.copy()
            else:
                time.sleep(0.01)

    def enable_and_reset(self):
        print("正在检查使能状态...")
        time.sleep(0.1)
        while not self.piper.EnablePiper():
            time.sleep(0.01)
        print("使能成功!!!!")

        print("正在移动到初始位置")
        position = [0.547, 1.258, -1.552, 0.003, 1.315, -0.576, 1]
        factor = RAD_TO_SDK_INT

        joint_0 = round(position[0] * factor)
        joint_1 = round(position[1] * factor)
        joint_2 = round(position[2] * factor)
        joint_3 = round(position[3] * factor)
        joint_4 = round(position[4] * factor)
        joint_5 = round(position[5] * factor)

        self.piper.ModeCtrl(0x01, 0x01, 80, 0x00)
        self.piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        self.piper.GripperCtrl(abs(GRIPPER_OPEN_SDK), 1000, 0x01, 0)

        time.sleep(3.0)

    def get_robot_state_rad(self):
        j_msg = self.piper.GetArmJointMsgs().joint_state
        g_msg = self.piper.GetArmGripperMsgs().gripper_state

        joints = [
            j_msg.joint_1 / RAD_TO_SDK_INT,
            j_msg.joint_2 / RAD_TO_SDK_INT,
            j_msg.joint_3 / RAD_TO_SDK_INT,
            j_msg.joint_4 / RAD_TO_SDK_INT,
            j_msg.joint_5 / RAD_TO_SDK_INT,
            j_msg.joint_6 / RAD_TO_SDK_INT,
        ]

        current_g_raw = g_msg.grippers_angle
        gripper_state = 1.0 if current_g_raw > GRIPPER_THRESHOLD_RAW else 0.0

        return joints, gripper_state

    def capture_image_bytes(self):
        with self.camera_lock:
            if self.latest_frame is None:
                return None, None
            frame = self.latest_frame.copy()

        frame = cv2.flip(frame, -1)
        img_resized = cv2.resize(frame, SEND_RES)
        _, img_encoded = cv2.imencode('.jpg', img_resized)
        return img_encoded.tobytes(), img_resized

    def execute_action(self, current_joints, action_pred):
        delta_joints = np.array(action_pred[:6])

        # ==================== 安全锁 1：单步增量截断 ====================
        max_delta = 0.05
        delta_joints = np.clip(delta_joints, -max_delta, max_delta)
        # ================================================================

        target_joints_rad = np.array(current_joints) + delta_joints

        # ==================== 安全锁 2：绝对物理限位 ====================
        # target_joints_rad[1] = np.clip(target_joints_rad[1], 0.39, 1.45)
        # target_joints_rad[2] = np.clip(target_joints_rad[2], -0.50, 3.15)
        # ================================================================

        target_gripper_state = action_pred[6]

        cmd_joints = [int(round(j * RAD_TO_SDK_INT)) for j in target_joints_rad]
        cmd_gripper = GRIPPER_OPEN_SDK if target_gripper_state > 0.5 else GRIPPER_CLOSE_SDK

        self.piper.JointCtrl(*cmd_joints)
        self.piper.GripperCtrl(cmd_gripper, 1000, 0x01, 0)
        return target_joints_rad.tolist()

    def display_frame(self, display_img, extra_text=""):
        """在画面上叠加调试信息并刷新窗口，返回是否需要退出。"""
        if display_img is None:
            return False
        info = f"Step: {self.step_count} (OFT)"
        if extra_text:
            info += f" | {extra_text}"
        cv2.putText(display_img, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow("Robot View", display_img)
        return (cv2.waitKey(1) & 0xFF) == ord('q')

    def run(self):
        print("\n" + "=" * 50)
        print("OpenVLA-OFT 远程推理客户端 (Action-Chunk Mode)")
        print(f"Server: {SERVER_URL}")
        print(f"USE_ACTION_CHUNK: {USE_ACTION_CHUNK}")
        print("=" * 50)

        instruction = input("请输入文本指令 (例如 'pick up the banana'): ").strip()
        print(f"当前任务: {instruction}")
        print(">>> 按下回车键开始推理，按 Ctrl+C 停止...")
        input()

        interval = 1.0 / CONTROL_FREQ

        try:
            while True:
                # ============================================================
                # 阶段 A：采集当前观测，向服务端请求一次推理，拿到 action chunk
                # ============================================================
                img_bytes, display_img = self.capture_image_bytes()

                # 先刷新一帧画面
                if self.display_frame(display_img, "requesting..."):
                    break

                if img_bytes is None:
                    time.sleep(0.01)
                    continue

                curr_joints, curr_gripper = self.get_robot_state_rad()
                state_list = curr_joints + [curr_gripper]
                state_json = json.dumps(state_list)

                # 发送请求
                try:
                    files = {'image': ('obs.jpg', img_bytes, 'image/jpeg')}
                    data = {'instruction': instruction, 'state': state_json}
                    response = requests.post(SERVER_URL, files=files, data=data, timeout=10)
                except Exception as e:
                    print(f"Req Failed: {e}")
                    time.sleep(0.05)
                    continue

                if response.status_code != 200:
                    print(f"Server Error: {response.status_code}")
                    continue

                result = response.json()

                # ---------------------------------------------------------
                # 解析服务端返回：优先使用 action_chunk，否则回退到 action
                # ---------------------------------------------------------
                if USE_ACTION_CHUNK and "action_chunk" in result:
                    action_chunk = result["action_chunk"]
                else:
                    # 兼容旧逻辑：action 可能是单个 list 或 list of lists
                    raw_action = result["action"]
                    if isinstance(raw_action[0], list):
                        action_chunk = raw_action
                    else:
                        action_chunk = [raw_action]

                chunk_len = len(action_chunk)
                print(f"[OFT] 收到 action chunk, 长度={chunk_len}, 首action={action_chunk[0][:3]}...")

                # ============================================================
                # 阶段 B：逐步执行 chunk 中的每一个 action（开环）
                # ============================================================
                for idx, action in enumerate(action_chunk):
                    step_start = time.time()

                    self.step_count += 1

                    # 基于上一步目标关节角做累加（保持平滑）
                    base_joints = (self.last_target_joints
                                   if self.last_target_joints is not None
                                   else curr_joints)
                    target_joints = self.execute_action(base_joints, action)
                    self.last_target_joints = target_joints

                    # 执行期间持续刷新画面
                    _, disp = self.capture_image_bytes()
                    if self.display_frame(disp, f"chunk {idx+1}/{chunk_len}"):
                        raise KeyboardInterrupt  # 用户按 q 退出

                    print(f"  step={self.step_count}, chunk[{idx+1}/{chunk_len}], "
                          f"action={action[:3]}...")

                    # 控频
                    elapsed = time.time() - step_start
                    if elapsed < interval:
                        time.sleep(interval - elapsed)

        except KeyboardInterrupt:
            print("\n停止运行...")
        finally:
            self.is_running = False
            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    client = RobotClient()
    client.run()
