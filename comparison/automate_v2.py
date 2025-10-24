import subprocess
import pyautogui
import pygetwindow as gw
import time
import os
import shutil
import pygetwindow as gw


# Rutas necesarias
matlab_script_dir = r"C:\Users\anton\OneDrive\Escritorio\Clustering\test-mat-AutoUFS\AutoUFSTool-main"
log_file = r"C:\Users\anton\OneDrive\Escritorio\Clustering\test-mat-AutoUFS\AutoUFSTool-main\matlab_output.txt"

def mover_resultados_a_carpeta():
    output_dir = os.path.join(matlab_script_dir, "results")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(matlab_script_dir):
        if file.endswith((".mat", ".png", ".fig", ".txt")) and not file.startswith("main"):
            full_path = os.path.join(matlab_script_dir, file)
            print(f"[PYTHON] Moviendo archivo: {file}")
            shutil.move(full_path, os.path.join(output_dir, file))


def run_matlab_and_monitor():
    print("[PYTHON] Lanzando MATLAB")

    # Elimina el log anterior si existe
    if os.path.exists(log_file):
        os.remove(log_file)

    # Comando que lanza MATLAB y guarda la salida en un log
    matlab_cmd = f"cd('{matlab_script_dir}'); main_comparacion"

    process = subprocess.Popen([
        r"C:\Program Files\MATLAB\R2022b\bin\matlab.exe",
        "-nosplash",
        "-nodesktop",
        "-logfile", log_file,
        "-r", matlab_cmd
    ])

    # Paso 1: Esperamos que MATLAB cargue y activamos su ventana
    time.sleep(10)



    
    matlab_window = None
    for w in gw.getWindowsWithTitle("MATLAB"):
        if "MATLAB" in w.title:
            matlab_window = w
            break

    if matlab_window:
        print("[PYTHON] Activando ventana de MATLAB...")
        matlab_window.activate()
        time.sleep(1)
    else:
        print("[ERROR] No se pudo encontrar la ventana de MATLAB.")

    # Guardamos ventanas actuales para comparar luego
    ventanas_iniciales = {w._hWnd for w in gw.getAllWindows()}

    # Paso 3: Monitorear log y detectar "?" + ventana emergente nueva
    ultima_longitud = 0
    while True:
        time.sleep(1)

        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                contenido = f.read()

            nuevo_texto = contenido[ultima_longitud:]
            ultima_longitud = len(contenido)

            if nuevo_texto:
                print("[MATLAB OUTPUT]:", nuevo_texto.strip())

                if "?" in nuevo_texto:
                    print("[PYTHON] Pregunta detectada. Enviando '1'...")
                    pyautogui.typewrite("1")
                    pyautogui.press("enter")

        # Paso 4: Comprobar si se ha abierto una ventana nueva
        ventanas_actuales = {w._hWnd for w in gw.getAllWindows()}
        nuevas_ventanas = ventanas_actuales - ventanas_iniciales

        if nuevas_ventanas:
            print("[PYTHON] Detectada ventana emergente. Finalizando...")
            matlab_window.activate()
            time.sleep(1)
            pyautogui.typewrite("exit")
            pyautogui.press("enter")
            time.sleep(2)
            # mover_resultados_a_carpeta()
            return

        #Paso 5: También salimos si el proceso termina
        """ if process.poll() is not None:
            print("[PYTHON] MATLAB ha terminado.")
            break """
        
if __name__ == "__main__":
    run_matlab_and_monitor()
