from robot_api import RobotAPI
import time

def main():
    # Initialize robot connection
    robot = RobotAPI("192.168.1.7")
    
    print("🤖 Starting basic robot movement demo...")
    
    # Turn on torque to enable movement
    #robot.send_torque(1, wait_time=2)
    time.sleep(5)
    
    # Get current position
    # print("\n📍 Current position:")
    # position = robot.get_position()
    # if position:
    #     print(f"   {position}")
    
    # Move to a simple position
    print("\n🎯 Moving to position")
    robot.move_to(x=0, y=-450, z=150, t=-300, speed=0, wait_time=3)

    print("\n🎯 SLAP")
    robot.move_base(ang=-20, speed=0, wait_time=0.6)  # 0.785 rad = 45 degrees
    robot.move_base(ang=-140, speed=0, wait_time=0.6)  # 0.785 rad = 45 degrees
    
    # Move back to center
    robot.move_base(ang=-90, speed=0, wait_time=0.6)  # 0.785 rad = 45 degrees
    
    # Turn on LED
    print("\n💡 Turning on LED...")
    robot.light_on(True, wait_time=3)
    # Turn off LED
    print("💡 Turning off LED...")
    robot.light_on(False, wait_time=1)

    
    robot.move_shoulder(ang=200, speed=0, wait_time=0.6)  # Move shoulder to center
    robot.move_elbow(ang=-200, speed=0, wait_time=1.6)  # Move elbow to center
    robot.move_shoulder(ang=10, speed=0, wait_time=1.6)  # Move shoulder to center
    #robot.move_elbow(ang=70, speed=0, wait_time=0.6)  # Move shoulder to center
    robot.move_to(x=0, y=-450, z=150, t=-300, speed=0, wait_time=0.6)
    robot.move_shoulder(ang=200, speed=0, wait_time=0.2)  # Move shoulder to center
    robot.move_elbow(ang=0, speed=0, wait_time=1.6)  # Move elbow to center
    
    robot.move_hand(ang=200, speed=0, wait_time=0.6)  # Move hand to center
    robot.move_hand(ang=0, speed=0, wait_time=0.6)  # Move hand to center
    robot.move_hand(ang=200, speed=0, wait_time=0.6)
    robot.move_hand(ang=0, speed=0, wait_time=0.6)  # Move hand to center
    robot.move_hand(ang=200, speed=0, wait_time=2)

    # Wait a bit
    time.sleep(2)
    robot.move_to(50, 0, -40, 3.14, speed=10, wait_time=1)

    #robot.close()
    
   
    print("\n✅ Demo complete!")

if __name__ == "__main__":
    main()