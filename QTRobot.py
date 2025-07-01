from PySide6.QtCore import QObject, Signal, QThread
from robot_api import RobotAPI


class RobotWorker(QThread):
    """Worker thread for robot movements."""
    
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, robot_api, steps):
        super().__init__()
        self.robot_api = robot_api
        self.steps = steps
        self._should_stop = False
        
    def run(self):
        """Execute robot steps in thread."""
        try:
            # Initialize robot position
            self.robot_api.move_to(50, 0, -40, 3.14, speed=10, wait_time=1)
            
            # Process each step
            for step_type, x, y in self.steps:
                if self._should_stop:
                    break
                    
                if step_type == 'process_red_plane':
                    self._process_red_plane(x, y)
                elif step_type == 'process_blue_plane':
                    self._process_blue_plane(x, y)
                    
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(f"Robot error: {str(e)}")
            
    def initialize(self):
        self.robot_api.move_to(50, 0, -40, 3.14, speed=10, wait_time=1) #init position    
    
    def _process_red_plane(self, x, y):
        """Process red plane object."""
        z = -10  # Standard z value
        
        # Move in general direction first to avoid collision
        self.robot_api.move_to(310, y, 80, 3.14, speed=10, wait_time=1)
        # Move down to object
        self.robot_api.move_to(x, y, z, 3.14, speed=10, wait_time=1)
        
        # Adjust elbow
        position = self.robot_api.get_position()
        if position:
            self.robot_api.move_elbow(rad=position["e"]+0.1, speed=0, wait_time=1)
            
        # Light sequence
        self.robot_api.light_on(on=True)
        self.robot_api.light_on(on=False)
        
        # Swipe movements
        if x > 300:
            move_x = x
        else:
            move_x = x-20
        
        double_swipe = False
        if -75 < y < 40 and x < 300:
            double_swipe = True

        if y < 0:
            self.robot_api.move_to(move_x, y+120, -100, 3.14, speed=10, wait_time=1)
            self.robot_api.move_to(move_x, -180, -100, 3.14, speed=0, wait_time=1)
        else:
            self.robot_api.move_to(move_x, y-120, -100, 3.14, speed=10, wait_time=1)
            self.robot_api.move_to(move_x, 200, -100, 3.14, speed=0, wait_time=1)
        
        if double_swipe:
            if y < 0:
                self.robot_api.move_to(move_x, y, 0, 3.14, speed=0, wait_time=1)
                self.robot_api.move_to(move_x, y, -100, 3.14, speed=0, wait_time=1)
                self.robot_api.move_to(move_x-20, -180, -100, 3.14, speed=0, wait_time=1)
            else:
                self.robot_api.move_to(move_x, y, 0, 3.14, speed=0, wait_time=1)
                self.robot_api.move_to(move_x, y, -100, 3.14, speed=0, wait_time=1)
                self.robot_api.move_to(move_x-20, 200, -100, 3.14, speed=0, wait_time=1)

        # Return to initial position
        self.robot_api.move_to(50, 0, -40, 3.14, speed=10, wait_time=1)
        
    def _process_blue_plane(self, x, y):
        """Process blue plane object."""
        z = -10  # Standard z value
        
        # Move in general direction first to avoid collision
        self.robot_api.move_to(310, y, 80, 3.14, speed=10, wait_time=1)
        # Move down to object
        self.robot_api.move_to(x, y, z, 3.14, speed=10, wait_time=1)
        
        # Adjust elbow
        position = self.robot_api.get_position()
        if position:
            self.robot_api.move_elbow(rad=position["e"]+0.1, speed=0, wait_time=1)
            
        # Light sequence
        self.robot_api.light_on(on=True)
        self.robot_api.light_on(on=False)
        
        # Hand movements
        self.robot_api.move_to(x, y, z, 3.14, speed=10, wait_time=1)
        self.robot_api.move_hand(1.5, speed=0, wait_time=0)
        self.robot_api.move_hand(3.14, speed=0, wait_time=2)
        
        # Return to initial position
        self.robot_api.move_to(50, 0, -40, 3.14, speed=10, wait_time=1)


class QTRobot(QObject):
    """Qt wrapper for robot control with threaded execution."""
    
    robot_complete = Signal()
    robot_error = Signal(str)
    
    def __init__(self, ip_address):
        super().__init__()
        self.robot_api = RobotAPI(ip_address)
        self.worker = None
       
        # Initialize robot position directly (blocking call - no problem)
        try:
            self.initialize()
        except Exception as e:
            self.robot_error.emit(f"Robot initialization error: {str(e)}")

    def get_position(self):
        return self.robot_api.get_position
    
    def initialize(self):
        self.robot_api.move_to(50, 0, -40, 3.14, speed=10, wait_time=1)
        self.light_on(False)

    def light_on(self, on=True):
        self.robot_api.light_on(on)

    def controller_move(self, direction):
        position = self.robot_api.get_position()
        if position:
            if direction == "U":
                self.robot_api.move_to(x=position["x"]+0, y=position["y"]+20, z=-10, t=position["t"], speed=0, wait_time=0)
            elif direction == "D":
                self.robot_api.move_to(x=position["x"]+0, y=position["y"]-20, z=-10, t=position["t"], speed=0, wait_time=0)
            elif direction == "L":
                self.robot_api.move_to(x=position["x"]-20, y=position["y"]+0, z=-10, t=position["t"], speed=0, wait_time=0)
            elif direction == "R":
                self.robot_api.move_to(x=position["x"]+20, y=position["y"]+0, z=-10, t=position["t"], speed=0, wait_time=0)
            elif direction == "UL":
                self.robot_api.move_to(x=position["x"]-20, y=position["y"]+20, z=-10, t=position["t"], speed=0, wait_time=0)
            elif direction == "UR":
                self.robot_api.move_to(x=position["x"]+20, y=position["y"]+20, z=-10, t=position["t"], speed=0, wait_time=0)
            elif direction == "DL":
                self.robot_api.move_to(x=position["x"]-20, y=position["y"]-20, z=-10, t=position["t"], speed=0, wait_time=0)
            elif direction == "DR":     
                self.robot_api.move_to(x=position["x"]+20, y=position["y"]-20, z=-10, t=position["t"], speed=0, wait_time=0)

        return self.robot_api.get_position()

    def process_steps_qt(self, steps):
        """
        Process a list of robot steps in a separate thread.
        
        Args:
            steps: List of tuples (step_type, x, y) where step_type is 
                   'process_red_plane' or 'process_blue_plane'
        """
        if self.worker and self.worker.isRunning():
            self.robot_error.emit("Robot is already executing commands")
            return
            
        # Create and configure worker
        self.worker = RobotWorker(self.robot_api, steps)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        
        # Start execution
        self.worker.start()
        
    def _on_worker_finished(self):
        """Handle worker completion."""
        self.worker = None
        self.robot_complete.emit()
        
    def _on_worker_error(self, error_msg):
        """Handle worker error."""
        self.worker = None
        self.robot_error.emit(error_msg)