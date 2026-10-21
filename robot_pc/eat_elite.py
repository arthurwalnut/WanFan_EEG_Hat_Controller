import argparse
import importlib.util
import ipaddress
import os
import time

joint1 = [
    168.50103305785123,
    -88.15728305785125,
    104.1615099009901,
    -105.96026234567901,
    92.07831790123457,
    85.40547839506172,
]

joint2 = [
    168.49690082644628,
    -102.4034090909091,
    77.02629950495049,
    -17.90162037037038,
    60.20100308641975,
    85.41975308641975,
]

home_joint = [
    170.35020661157026,
    -85.64927685950413,
    90.22710396039605,
    -89.79282407407408,
    89.9074074074074,
    85.40123456790123,
]

ROBOT_IP = "192.168.137.200"

# 关节运动参数
MOVE_SPEED = 200
MOVE_ACC = 100
MOVE_DEC = 100

# 直线推进参数
LINE_SPEED = 300.0
LINE_SPEED_TYPE = 0   # 0: V
LINE_ACC = 100
LINE_DEC = 100
FORWARD_MM = 50.0     # 5 cm


def to_joint8(j6):
    if len(j6) == 8:
        return j6
    if len(j6) == 6:
        return j6 + [0.0, 0.0]
    raise ValueError(f"joint length must be 6 or 8, got {len(j6)}")


def move_joint_and_wait(ec, target6, name):
    target8 = to_joint8(target6)
    print(f"[MOVE] {name}: {target6}")

    ok = ec.move_joint(
        target_joint=target8,
        speed=MOVE_SPEED,
        acc=MOVE_ACC,
        dec=MOVE_DEC,
    )
    if not ok:
        raise RuntimeError(f"move_joint 到 {name} 下发失败")

    ec.wait_stop()
    cur = ec.get_joint()
    print(f"[ARRIVED] {name}: current={cur}")


def move_along_tool_z_single(
    ec,
    delta_mm,
    speed=LINE_SPEED,
    speed_type=LINE_SPEED_TYPE,
    acc=LINE_ACC,
    dec=LINE_DEC,
):
    """
    沿当前 TCP 局部 Z 方向一次性直线移动:
    delta_mm > 0 -> Tool +Z
    delta_mm < 0 -> Tool -Z
    """
    if abs(delta_mm) < 1e-9:
        return ec.get_tcp_pose(unit_type=0)

    cur_pose = ec.get_tcp_pose(unit_type=0)
    if cur_pose is None or len(cur_pose) != 6:
        raise RuntimeError(f"invalid current tcp pose: {cur_pose}")

    ok = ec.move_line_in_coord(
        target_user_pose=[0.0, 0.0, float(delta_mm), 0.0, 0.0, 0.0],
        speed=speed,
        speed_type=speed_type,
        user_coord=cur_pose,
        acc=acc,
        dec=dec,
        unit_type=0,
    )
    if not ok:
        raise RuntimeError("move_line_in_coord 下发失败")

    ec.wait_stop()
    return ec.get_tcp_pose(unit_type=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Original Elite trajectory; run only after checking the physical workspace.")
    parser.add_argument("--ip", default=os.environ.get("ELITE_ROBOT_IP", ROBOT_IP),
                        help="Elite IPv4; original configuration is 192.168.137.200")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        available = importlib.util.find_spec("elite") is not None
        print("elite SDK: " + ("available; robot was not contacted" if available else "not installed"))
        return 0 if available else 1
    robot_ip = str(ipaddress.IPv4Address(args.ip))
    from elite import EC
    ec = EC(ip=robot_ip, auto_connect=True)

    print(f"[INFO] mode={ec.mode}, state={ec.state}")
    print(f"[INFO] servo_status={ec.servo_status}, sync_status={ec.sync_status}")

    if not ec.servo_status:
        ec.robot_servo_on()
        time.sleep(0.5)

    if not ec.sync_status:
        raise RuntimeError("机器人未同步，先在示教器完成同步后再运行")

    print("[INFO] current joint:", ec.get_joint())
    print("[INFO] current tcp(mm,deg):", ec.get_tcp_pose(unit_type=0))

    # 1) 到 j1
    move_joint_and_wait(ec, joint1, "j1")

    # 2) 到 j2
    move_joint_and_wait(ec, joint2, "j2")

    # 3) 到 j2 后，沿当前 TCP 局部 +Z 一段前进 5 cm
    print(f"[MOVE] forward along current TCP +Z: {FORWARD_MM} mm")
    tcp_after = move_along_tool_z_single(ec, delta_mm=FORWARD_MM)
    print("[ARRIVED] tcp after forward(mm,deg):", tcp_after)

    # 4) 回原点
    move_joint_and_wait(ec, home_joint, "home")

    print("[DONE] sequence finished")


if __name__ == "__main__":
    raise SystemExit(main())
