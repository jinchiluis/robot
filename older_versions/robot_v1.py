from robot_api import RobotAPI

# Roboter-Instanz erstellen
robot = RobotAPI("192.168.178.98")

# Use robot methods directly for basic movements

def initialize():
    robot.move_to(50, 0, -40, 3.14, speed=10, wait_time=1) #move back

def show_the_object(x, y, z=-10, speed=10):   
    robot.move_to(310, 10, 200, 3.14, speed=10, wait_time=1) #move away/start position

    robot.move_to(310, y, 80, 2.14, speed=10, wait_time=1) #move in the general direction first so it does not collide
    robot.move_to(x, y, z, 2.14, speed=10, wait_time=1) #move down to object/open hand
    
    robot.move_to(x, y, z, 3.14, speed=10, wait_time=1) #grab object
    robot.move_elbow(35)  #lift object up
    
    robot.move_to(210, -430, 160, 3.14, speed=10, wait_time=3) #show it to the audience
    
    robot.move_to(410, y, 180, 3.14, speed=10, wait_time=1) #smooth return
    robot.move_to(x, y, z, 3.14, speed=10, wait_time=2) #put object down
    robot.move_to(x, y, z, 2.14, speed=10, wait_time=1) #release object

    robot.move_elbow(35)  #move hand up so not to collide
    robot.move_to(310, 10, 200, 3.14, speed=10, wait_time=1) #move away/start position

def process_red(x, y, z, speed):
    robot.move_to(310, 10, 200, 3.14, speed=10, wait_time=1) #move away/start position

    robot.move_to(310, y, 80, 2.14, speed=10, wait_time=1) #move in the general direction first so it does not collide
    robot.move_to(x, y, z, 2.14, speed=10, wait_time=1) #move down to object/open hand
    
    robot.move_to(x, y, z, 3.14, speed=10, wait_time=1) #grab object
    robot.move_elbow(35)  #lift object up

    robot.move_to(210, -430, 160, 3.14, speed=10, wait_time=3) #show it to the audience
    robot.move_to(210, -430, 160, 2.14, speed=10, wait_time=3) #release

    robot.move_to(310, 10, 200, 3.14, speed=10, wait_time=1) #move away/start position

def process_blue(x, y, z, speed):
    robot.move_to(310, 10, 200, 3.14, speed=10, wait_time=1) #move away/start position

    robot.move_to(310, y, 80, 2.14, speed=10, wait_time=1) #move in the general direction first so it does not collide
    robot.move_to(x, y, z, 2.14, speed=10, wait_time=1) #move down to object/open hand
    
    robot.move_to(x, y, z, 3.14, speed=10, wait_time=1) #grab object
    robot.move_elbow(35)  #lift object up

    robot.move_to(x, y, z, 3.14, speed=10, wait_time=2) #put object down
    robot.move_to(x, y, z, 2.14, speed=10, wait_time=1) #release object

    robot.move_to(310, 10, 200, 3.14, speed=10, wait_time=1) #move away/start position

def process_red_plane(x, y, z, speed):
    ####### Red Plane #########
    #robot.move_to(310, 10, 200, 3.14, speed=10, wait_time=1) #move away/start position

    robot.move_to(310, y, 80, 3.14, speed=10, wait_time=1) #move in the general direction first so it does not collide
    robot.move_to(x, y, z, 3.14, speed=10, wait_time=1) #move down to object/open hand

    position = robot.get_position()
    if position:
        robot.move_elbow(rad=position["e"]+0.1, speed=0, wait_time=1) #must rad+0.2 
    robot.light_on(on=True)
    robot.light_on(on=False)
    robot.move_to(x, y+120, -80, 3.14, speed=10, wait_time=1) #swipe 
    robot.move_to(x, -150, -80, 3.14, speed=0, wait_time=1) #swipe y=-110 statisch
    
    

    robot.move_to(50, 0, -40, 3.14, speed=10, wait_time=1) #move back

def process_blue_plane(x, y, z, speed):
    ####### Blue Plane #########
    #robot.move_to(310, 10, 200, 3.14, speed=10, wait_time=1) #move away/start position

    robot.move_to(310, y, 80, 3.14, speed=10, wait_time=1) #move in the general direction first so it does not collide
    robot.move_to(x, y, z, 3.14, speed=10, wait_time=1) #move down to object/open hand
    
    position = robot.get_position()
    if position:
        robot.move_elbow(rad=position["e"]+0.1, speed=0, wait_time=1) #must rad+0.2 
    robot.light_on(on=True)
    robot.light_on(on=False)
    robot.move_to(x, y, z, 3.14, speed=10, wait_time=1) #move down to object/open hand
    robot.move_hand(1.5, speed=0, wait_time=1)
    robot.move_hand(3.14, speed=0, wait_time=2)

    robot.move_to(50, 0, -40, 3.14, speed=10, wait_time=1) #move back

if __name__ == "__main__":
    print("RoArm-M2-S Controller")
    print("-" * 30)
    
    # Torque On für Bewegung
    #send_torque(1)
    
    # Bewegungssequenz
    #move_to(310, 10, 200, 3.14, speed=10, wait_time=1) #move away/start position
    #move_to_joint_angles(base=-45, shoulder=45, elbow=90, hand=45, speed=50, acc=5, wait_time=2)
    #move_to(210, -430, 160, 3.14, speed=30, wait_time=2) #show it to the audience
    #grab_the_object()

    #Check Some coordinate points (for later pixel coordinate to robor coordinate cacls)
    #robot.move_to(510, 0, -10, 2.14, speed=10, wait_time=1)
    #robot.move_base(-18)
    result = robot.get_position()
    robot.move_to(50, 0, -40, 3.14, speed=10, wait_time=1) #move away/start position
    print(result)
    if result:
        print("e:", result["e"])
    #show_the_object(260, 90, -10, speed=10)

    # Optional: Torque Off am Ende
    # robot.send_torque(0)