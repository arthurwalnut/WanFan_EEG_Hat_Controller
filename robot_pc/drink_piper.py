import argparse
import importlib.util
import os
import time

piper = None  # Initialized only by explicit main(); importing never contacts CAN.

# ===== 点位 =====
HOME_J = [475, 5477, -4271, 1375, 6047, -3175]  # 新原点
target_j1 = [2524, 108580, -12725, 2552, -75088, -8776]
target_j2 = [2524, 20595, -12004, 53674, -63572, 1705]

# ===== 夹爪参数 =====
OPEN_WIDTH_MM = 69
CLOSE_WIDTH_MM = 47
OPEN_FORCE_N = 0
GRIP_FORCE_N = 3

# 原点处夹爪参数：距离和力都为 0
HOME_WIDTH_MM = 0
HOME_FORCE_N = 0

# ===== 运动参数 =====
MOVE_SPEED = 10
SEND_HZ = 50
JOINT_TOL_LIST = [800, 800, 800, 800, 6000, 800]
STABLE_CNT = 5
MOVE_TIMEOUT_S = 15.0


def mm_to_gripper_raw(mm: float) -> int:
    return int(mm * 1000)


def n_to_force_raw(n: float) -> int:
    return int(n * 1000)


def get_joint_raw():
    msg = piper.GetArmJointMsgs()
    js = msg.joint_state
    return [js.joint_1, js.joint_2, js.joint_3, js.joint_4, js.joint_5, js.joint_6]


def set_gripper(width_mm: float, force_n: float, wait_s: float = 0.8):
    piper.GripperCtrl(mm_to_gripper_raw(width_mm), n_to_force_raw(force_n), 0x01, 0)
    time.sleep(wait_s)


def joints_reached(target, tol_list=JOINT_TOL_LIST):
    cur = get_joint_raw()
    err = [abs(c - t) for c, t in zip(cur, target)]
    ok = all(e <= tol for e, tol in zip(err, tol_list))
    return ok, cur, err


def move_joint_blocking(target, speed=MOVE_SPEED,
                        keep_grip=False,
                        grip_width_mm=CLOSE_WIDTH_MM,
                        grip_force_n=GRIP_FORCE_N,
                        timeout_s=MOVE_TIMEOUT_S):
    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)

    dt = 1.0 / SEND_HZ
    stable = 0
    t0 = time.time()

    while time.time() - t0 < timeout_s:
        piper.JointCtrl(
            target[0], target[1], target[2],
            target[3], target[4], target[5]
        )

        # 运动过程中持续保持夹紧
        if keep_grip:
            piper.GripperCtrl(
                mm_to_gripper_raw(grip_width_mm),
                n_to_force_raw(grip_force_n),
                0x01,
                0
            )

        ok, cur, err = joints_reached(target)
        if ok:
            stable += 1
            if stable >= STABLE_CNT:
                print("arrived:", cur)
                return True
        else:
            stable = 0

        time.sleep(dt)

    print("move timeout")
    print("target :", target)
    print("current:", get_joint_raw())
    return False


def move_to_target():
    time.sleep(0.5)

    # 1. 先张开
    print("open gripper")
    set_gripper(OPEN_WIDTH_MM, OPEN_FORCE_N, wait_s=0.8)

    # 2. 到 j1（抓取点）
    print("move to j1:", target_j1)
    if not move_joint_blocking(target_j1, keep_grip=False):
        raise RuntimeError("j1 未到位")

    # 3. 在 j1 夹紧
    print("close gripper at j1")
    set_gripper(CLOSE_WIDTH_MM, GRIP_FORCE_N, wait_s=0.8)

    # 4. 去 j2，途中保持夹紧
    print("move to j2:", target_j2)
    if not move_joint_blocking(
        target_j2,
        keep_grip=True,
        grip_width_mm=CLOSE_WIDTH_MM,
        grip_force_n=GRIP_FORCE_N
    ):
        raise RuntimeError("j2 未到位")

    # 5. 先回到 j1，途中继续保持夹紧
    print("return to j1:", target_j1)
    if not move_joint_blocking(
        target_j1,
        keep_grip=True,
        grip_width_mm=CLOSE_WIDTH_MM,
        grip_force_n=GRIP_FORCE_N
    ):
        raise RuntimeError("返回 j1 未到位")

    # 6. 到 j1 后松开夹爪
    print("open gripper at j1")
    set_gripper(OPEN_WIDTH_MM, OPEN_FORCE_N, wait_s=0.8)

    # 7. 再返回原点 HOME_J
    print("return to home:", HOME_J)
    if not move_joint_blocking(
        HOME_J,
        keep_grip=False
    ):
        raise RuntimeError("返回 HOME_J 未到位")

    # 8. 到原点后把夹爪设为 0 宽度、0 力
    print("set gripper at home: width=0, force=0")
    set_gripper(HOME_WIDTH_MM, HOME_FORCE_N, wait_s=0.8)


def main(argv=None):
    global piper
    parser = argparse.ArgumentParser(description="Original Piper trajectory using the installed SDK's default CAN interface.")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--can-name", default=os.environ.get("PIPER_CAN_NAME"), help="optional activated SocketCAN interface; otherwise retain SDK default")
    args = parser.parse_args(argv)
    if args.check:
        available = importlib.util.find_spec("piper_sdk") is not None
        print("piper_sdk: " + ("available; CAN was not contacted" if available else "not installed"))
        return 0 if available else 1
    from piper_sdk import C_PiperInterface
    piper = C_PiperInterface(can_name=args.can_name) if args.can_name else C_PiperInterface()
    piper.ConnectPort()
    piper.EnableArm(7)
    print("before:", get_joint_raw())
    move_to_target()
    print("after :", get_joint_raw())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
