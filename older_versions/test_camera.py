import cv2

def find_available_cameras(max_tested=10):
    """Findet alle verfügbaren Kameras und ihre Indizes."""
    available_cameras = []
    
    for index in range(max_tested):
        # Versuche verschiedene Backends
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Auto")
        ]
        
        for backend, backend_name in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    available_cameras.append({
                        'index': index,
                        'backend': backend_name,
                        'backend_id': backend,
                        'resolution': f"{width}x{height}"
                    })
                    cap.release()
                    break  # Gefunden, nächster Index
                cap.release()
    
    return available_cameras

def test_camera(index, backend=cv2.CAP_DSHOW):
    """Testet eine spezifische Kamera."""
    print(f"\nTeste Kamera Index {index} mit Backend {backend}...")
    cap = cv2.VideoCapture(index, backend)
    
    if not cap.isOpened():
        print(f"❌ Konnte Kamera {index} nicht öffnen")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print(f"❌ Konnte kein Bild von Kamera {index} lesen")
        cap.release()
        return False
    
    print(f"✅ Kamera {index} funktioniert!")
    print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
    
    # Zeige das Bild kurz an
    cv2.imshow(f'Test Camera {index}', frame)
    print("   Drücke eine Taste im Fenster um fortzufahren...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cap.release()
    return True

if __name__ == "__main__":
    print("🔍 Suche nach verfügbaren Kameras...\n")
    
    cameras = find_available_cameras()
    
    if not cameras:
        print("❌ Keine Kameras gefunden!")
        print("\nMögliche Probleme:")
        print("1. Webcam ist nicht angeschlossen")
        print("2. Treiber fehlen")
        print("3. Andere Software nutzt die Kamera")
        print("4. Windows Datenschutz blockiert Kamerazugriff")
    else:
        print(f"✅ {len(cameras)} Kamera(s) gefunden:\n")
        for cam in cameras:
            print(f"📷 Index: {cam['index']}")
            print(f"   Backend: {cam['backend']}")
            print(f"   Auflösung: {cam['resolution']}")
            print()
        
        # Teste die erste gefundene Kamera
        if cameras:
            first_cam = cameras[0]
            test_camera(first_cam['index'], first_cam['backend_id'])