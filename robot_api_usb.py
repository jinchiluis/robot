import requests
import json
import time
import serial
import serial.tools.list_ports

class RobotAPI:
    """API-Klasse für RoArm-M2-S Roboter-Arm mit Netzwerk- und USB-Unterstützung"""
    
    def __init__(self, serial_port=None, baudrate=115200):
        """
        Initialisiert die RobotAPI
        
        Args:
            serial_port: COM-Port für USB-Verbindung (z.B. "COM3" oder "/dev/ttyUSB0")
                        Wenn None, wird automatisch nach verfügbaren Ports gesucht
            baudrate: Baudrate für serielle Verbindung (Standard: 115200)
        """
        self.serial_port = None
        self.serial = None
        self.baudrate = baudrate
        self._init_usb_connection(serial_port)
    
    def _init_usb_connection(self, port=None):
        """Initialisiert die USB/Serielle Verbindung"""
        try:
            if port is None:
                # Automatische Port-Erkennung
                ports = list(serial.tools.list_ports.comports())
                print("Verfügbare Ports:")
                for p in ports:
                    print(f"  - {p.device}: {p.description}")
                
                # Suche nach typischen Roboter-Beschreibungen
                for p in ports:
                    if any(keyword in p.description.lower() for keyword in ['robot', 'arm', 'arduino', 'ch340', 'ft232']):
                        port = p.device
                        print(f"✅ Automatisch erkannter Port: {port}")
                        break
                
                if port is None and ports:
                    port = ports[0].device
                    print(f"⚠️  Verwende ersten verfügbaren Port: {port}")
            
            if port:
                self.serial = serial.Serial(port, self.baudrate, timeout=0.01)
                self.serial_port = port
                time.sleep(2)  # Warte auf Arduino-Reset
                print(f"✅ USB-Verbindung hergestellt auf {port}")
            else:
                raise Exception("Kein serieller Port gefunden")
                
        except Exception as e:
            print(f"❌ USB-Verbindungsfehler: {e}")
            self.serial = None
    
    # Entfernt: switch_to_network und switch_to_usb
    
    def close(self):
        """Schließt die Verbindung"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("✅ USB-Verbindung geschlossen")
    
    def send_command(self, command_dict):
        """Sendet einen Befehl an den RoArm-M2-S (nur USB)"""
        return self._send_usb_command(command_dict)
    
    # Entfernt: _send_network_command
    
    def _send_usb_command(self, command_dict):
        """Sendet einen Befehl über USB/Seriell"""
        if not self.serial or not self.serial.is_open:
            print("❌ Keine USB-Verbindung vorhanden")
            return None
        
        try:
            # Clear input buffer first
            self.serial.reset_input_buffer()
            
            # JSON-String erstellen und mit Newline abschließen
            json_str = json.dumps(command_dict) + '\n'
            print(f"🔌 Sende Befehl: {json_str.strip()}")
            
            # Sende Befehl
            self.serial.write(json_str.encode('utf-8'))
            
            # Read multiple lines to get the actual response
            # First line is usually the echo
            echo = self.serial.readline().decode('utf-8').strip()
            print(f"🔌 Echo: {echo}")
            
            # Second line should be the actual response
            response_data = self.serial.readline().decode('utf-8').strip()
            print(f"🔌 Antwort: {response_data}")
            
            # If response is still empty, wait a bit more
            if not response_data or response_data == "":
                time.sleep(0.1)
                response_data = self.serial.readline().decode('utf-8').strip()
            
            if response_data:
                # Erstelle ein Response-ähnliches Objekt für Kompatibilität
                class USBResponse:
                    def __init__(self, text):
                        self.text = text
                        self.status_code = 200
                
                return USBResponse(response_data)
            else:
                #return echo
                print("⚠️  Keine Antwort vom Roboter erhalten")
                return None
                
        except Exception as e:
            print(f"❌ USB-Übertragungsfehler: {e}")
            return None

    def get_position(self):
        command = {"T": 105}
        response = self.send_command(command)
        if response:
            try:
                result = json.loads(response.text)
                return result
            except json.JSONDecodeError as e:
                print(f"❌ Fehler beim Parsen der JSON-Antwort: {e}")
                return None

    def send_torque(self, state, wait_time=1):
        """Sendet Torque Befehl (0=off, 1=on)"""
        command = {"T": 210, "cmd": state}
        result = self.send_command(command)
        if result and wait_time > 0:
            time.sleep(wait_time)
        return result

    def move_to(self, x, y, z, t, speed=10, wait_time=1):
        """Bewegt den Arm zu Position (x,y,z) und wartet"""
        command = {"T": 1041, "x": x, "y": y, "z": z, "t": t, "spd": speed}
        result = self.send_command(command)
        if wait_time > 0:
            time.sleep(wait_time)
        return result

    def move_to_joint_angles(self, base=None, shoulder=None, elbow=None, hand=None, speed=40, acc=5, wait_time=1):
        """Zu spezifischen Gelenkwinkeln fahren (in Radians)"""
        command = {"T": 122, "spd": speed, "acc": acc}
            
        if base is not None:
            command["b"] = base
        if shoulder is not None:
            command["s"] = shoulder  
        if elbow is not None:
            command["e"] = elbow
        if hand is not None:
            command["h"] = hand
                
        result = self.send_command(command)
        if wait_time > 0:
            time.sleep(wait_time)
        return result

    def move_to_joint_angles_rad(self, base=None, shoulder=None, elbow=None, hand=None, speed=0, acc=1, wait_time=0):
        """Zu spezifischen Gelenkwinkeln fahren (in Radians)"""
        command = {"T": 102, "spd": speed, "acc": acc}
            
        if base is not None:
            command["base"] = base
        if shoulder is not None:
            command["shoulder"] = shoulder  
        if elbow is not None:
            command["elbow"] = elbow
        if hand is not None:
            command["hand"] = hand
                
        result = self.send_command(command)
        if wait_time > 0:
            time.sleep(wait_time)
        return result
    
    def move_base(self, ang, speed=40, acc=10, wait_time=1):
        """Bewegt nur das Base-Gelenk (joint 0) zu angegebenem Winkel in Radians"""
        command = {"T": 121, "joint": 1, "angle": ang, "spd": speed, "acc": acc}
        result = self.send_command(command)
        if wait_time > 0:
            time.sleep(wait_time)
        return result

    def move_shoulder(self, ang, speed=40, acc=10, wait_time=1):
        """Bewegt nur das Shoulder-Gelenk (joint 1) zu angegebenem Winkel in Radians"""
        command = {"T": 121, "joint": 2, "angle": ang, "spd": speed, "acc": acc}
        result = self.send_command(command)
        if wait_time > 0:
            time.sleep(wait_time)
        return result

    def move_elbow(self, ang=None, rad=None, speed=40, acc=10, wait_time=1):
        if ang is not None:
            command = {"T": 121, "joint": 3, "angle": ang, "spd": speed, "acc": acc}
        elif rad is not None:
            command = {"T": 101, "joint": 3, "rad": rad, "spd": speed, "acc": acc}
                
        result = self.send_command(command)
        if wait_time > 0:
            time.sleep(wait_time)
        return result

    def move_hand(self, ang, speed=40, acc=10, wait_time=1):
        """Bewegt nur das Hand-Gelenk (joint 3) zu angegebenem Winkel in Radians"""
        command = {"T": 121, "joint": 4, "angle": ang, "spd": speed, "acc": acc}
        result = self.send_command(command)
        if wait_time > 0:
            time.sleep(wait_time)
        return result

    def light_on(self, on=True, wait_time=1):
        if on == True:
            command = {"T": 114, "led":255}
        else:
            command = {"T": 114, "led":0}
        result = self.send_command(command)
        if wait_time > 0:
            time.sleep(wait_time)
        return result
    
    def create_mission(self, name, intro=""):
        """Erstellt eine neue Mission"""
        command = {"T": 220, "name": name, "intro": intro}
        result = self.send_command(command)
        return result

    def add_mission_step(self, name, x, y, z, t, speed=80):
        """Fügt einen Schritt zu einer bestehenden Mission hinzu"""
        # Build the step as a dictionary first, then convert to JSON string
        step_dict = {
            "T": 1041,  # Use 1041 for direct movement command
            "x": x,
            "y": y,
            "z": z,
            "t": t,
            "spd": speed
        }
        # Convert to JSON string (note: no wait_time in standard commands)
        step_json = json.dumps(step_dict)
        
        command = {"T": 222, "mission": name, "step": step_json}
        result = self.send_command(command)
        return result
    
    def play_mission(self, name):
        """Spielt eine Mission ab"""
        command = {"T": 242, "name": name}
        result = self.send_command(command)
        return result 
    
    def mission_content(self, name):
        """Lädt den Inhalt einer Mission"""
        command = {"T": 221, "name": name}
        response = self.send_command(command)
        print(f"response: {response.text}")
        if response:
            try:
                result = json.loads(response.text)
                return result
            except json.JSONDecodeError as e:
                print(f"❌ Fehler beim Parsen der JSON-Antwort: {e}")
                return None
        return None
    
# Beispiel-Verwendung:
if __name__ == "__main__":
    # USB-Verbindung mit automatischer Port-Erkennung
    robot_usb = RobotAPI()
    
    # USB-Verbindung mit spezifischem Port
    # robot_usb_manual = RobotAPI(serial_port="COM3")
    
    robot_usb.send_torque(1)
    robot_usb.move_to(100, 0, 150, 0, speed=20)
    position = robot_usb.get_position()
    
    # Verbindung schließen (wichtig bei USB)
    robot_usb.close()