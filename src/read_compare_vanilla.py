import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import numpy as np

######################################################################
######################################################################
# PARA VANILLA (solo GA simple) VS GAUFS ORIGINAL
######################################################################
######################################################################


###################################
#PARA SINTéTICOS
###################################

# Ruta base
original_path = './results_old/results_seleccionados_sinteticos/'

# Obtener todas las carpetas en el directorio
folders = [f for f in os.listdir(original_path) if os.path.isdir(os.path.join(original_path, f))]

# Listas para almacenar los datos
folder_names = []
ami_values = []
valor_3_values = []

# Iterar sobre cada carpeta
for folder in folders:
    # Extraer el nombre base después de "results_parallel_"
    base_name = folder.replace('results_parallel_', 'results_')
    
    # Construir la ruta completa del archivo CSV
    file_path = os.path.join(original_path, folder, f"{base_name}.csv")
    
    # Verificar si el archivo existe
    if os.path.exists(file_path):
        # Leer el CSV
        df = pd.read_csv(file_path)
        
        # Extraer los valores
        ami = df['ami_mejor_individuo'].values[0]
        
        # Leer el archivo JSON
        json_name = base_name.replace('results_', '') + '_silhouette_ward.json'
        json_path = os.path.join(original_path, folder, 'analyisis_significance_and_num_cluster', json_name)
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # Extraer valores
            valores = json_data['dicc_soluciones']['(0.5, 1)']
            valor_3 = valores[3]
            
            # Guardar datos
            folder_names.append(folder.replace('results_parallel_', ''))
            ami_values.append(ami)
            valor_3_values.append(valor_3)

# Crear figura con dos subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Gráfico de barras original
x = range(len(folder_names))
width = 0.35

bars1 = ax1.bar([i - width/2 for i in x], ami_values, width, label='AMI mejor cromosoma (VANILLA)', alpha=0.8)
bars2 = ax1.bar([i + width/2 for i in x], valor_3_values, width, label='Valor Con GAUFS Original (Completo)', alpha=0.8)

ax1.set_xlabel('Carpetas')
ax1.set_ylabel('Valores')
ax1.set_title('Comparación AMI vanilla vs GAUFS en Sintéticos')
ax1.set_xticks(x)
ax1.set_xticklabels(folder_names, rotation=90, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Subplot 2: Violin plot con estadísticas
data_for_violin = [ami_values, valor_3_values]
positions = [1, 2]

parts = ax2.violinplot(data_for_violin, positions=positions, showmeans=True, showmedians=True)

# Añadir estadísticas como texto
ami_mean = np.mean(ami_values)
ami_std = np.std(ami_values)
valor3_mean = np.mean(valor_3_values)
valor3_std = np.std(valor_3_values)

ax2.text(1, ax2.get_ylim()[1] * 0.95, f'Media: {ami_mean:.4f}\nStd: {ami_std:.4f}', 
         ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax2.text(2, ax2.get_ylim()[1] * 0.95, f'Media: {valor3_mean:.4f}\nStd: {valor3_std:.4f}', 
         ha='center', va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

ax2.set_xticks([1, 2])
ax2.set_xticklabels(['AMI VANILLA', 'GAUFS Original'])
ax2.set_ylabel('Valores')
ax2.set_title('Distribución y Estadísticas en Sintéticos')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Imprimir estadísticas en consola
print(f"\nEstadísticas:")
print(f"AMI VANILLA - Media: {ami_mean:.4f}, Desv. Típica: {ami_std:.4f}")
print(f"GAUFS Original - Media: {valor3_mean:.4f}, Desv. Típica: {valor3_std:.4f}")

###################################
# PARA REALES
###################################


# Ruta base
original_path = './results_old/results_seleccionados_reales/'

# Obtener todas las carpetas en el directorio
folders = [f for f in os.listdir(original_path) if os.path.isdir(os.path.join(original_path, f))]

# Listas para almacenar los datos
folder_names = []
ami_values = []
valor_3_values = []

# Iterar sobre cada carpeta
for folder in folders:
    # Extraer el nombre base después de "results_parallel_"
    base_name = folder.replace('results_parallel_', 'results_')
    
    # Construir la ruta completa del archivo CSV
    file_path = os.path.join(original_path, folder, f"{base_name}.csv")
    
    print(f"Procesando archivo: {file_path}")

    # Omitir archivos que contienen "Yale" en su nombre porque esta corrupto el csv
    # if "Yale" in file_path:
    #     folder_names.append("Yale")
    #     ami_values.append(0.11251141797480321)
    #     valor_3_values.append(valor_3)
    #     continue
    # Verificar si el archivo existe
    if os.path.exists(file_path):
        # Leer el CSV
        df = pd.read_csv(file_path)
        
        # Extraer los valores
        ami = df['ami_mejor_individuo'].values[0]
        
        # Leer el archivo JSON
        json_name = base_name.replace('results_', '') + '_silhouette_ward.json'
        json_path = os.path.join(original_path, folder, 'analyisis_significance_and_num_cluster', json_name)
        print(f"Procesando archivo: {json_name}")
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # Extraer valores
            valores = json_data['dicc_soluciones']['(0.5, 1)']
            valor_3 = valores[3]
            
            # Guardar datos
            folder_names.append(folder.replace('results_parallel_', ''))
            ami_values.append(ami)
            valor_3_values.append(valor_3)

# Crear figura con dos subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Gráfico de barras original
x = range(len(folder_names))
width = 0.35

bars1 = ax1.bar([i - width/2 for i in x], ami_values, width, label='AMI mejor cromosoma (VANILLA)', alpha=0.8)
bars2 = ax1.bar([i + width/2 for i in x], valor_3_values, width, label='Valor Con GAUFS Original (Completo)', alpha=0.8)

ax1.set_xlabel('Carpetas')
ax1.set_ylabel('Valores')
ax1.set_title('Comparación AMI vanilla vs GAUFS en Reales')
ax1.set_xticks(x)
ax1.set_xticklabels(folder_names, rotation=90, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Subplot 2: Violin plot con estadísticas
data_for_violin = [ami_values, valor_3_values]
positions = [1, 2]

parts = ax2.violinplot(data_for_violin, positions=positions, showmeans=True, showmedians=True)

# Añadir estadísticas como texto
ami_mean = np.mean(ami_values)
ami_std = np.std(ami_values)
valor3_mean = np.mean(valor_3_values)
valor3_std = np.std(valor_3_values)

ax2.text(1, ax2.get_ylim()[1] * 0.95, f'Media: {ami_mean:.4f}\nStd: {ami_std:.4f}', 
         ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax2.text(2, ax2.get_ylim()[1] * 0.95, f'Media: {valor3_mean:.4f}\nStd: {valor3_std:.4f}', 
         ha='center', va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

ax2.set_xticks([1, 2])
ax2.set_xticklabels(['AMI VANILLA', 'GAUFS Original'])
ax2.set_ylabel('Valores')
ax2.set_title('Distribución y Estadísticas en Reales')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Imprimir estadísticas en consola
print(f"\nEstadísticas:")
print(f"AMI VANILLA - Media: {ami_mean:.4f}, Desv. Típica: {ami_std:.4f}")
print(f"GAUFS Original - Media: {valor3_mean:.4f}, Desv. Típica: {valor3_std:.4f}")

######################################################################
######################################################################
# PARA GAUFS con num_clusters estimado beforehand VS GAUFS ORIGINAL
######################################################################
######################################################################


###################################
#PARA SINTéTICOS
###################################

# Ruta base
original_path = './results_old/results_seleccionados_sinteticos/'

# Obtener todas las carpetas en el directorio
folders = [f for f in os.listdir(original_path) if os.path.isdir(os.path.join(original_path, f))]

# Listas para almacenar los datos
folder_names = []
gaufs_cluster_beforehand = []
gaufs_original = []

# Iterar sobre cada carpeta
for folder in folders:
    # Extraer el nombre base después de "results_parallel_"
    base_name = folder.replace('results_parallel_', 'results_')
    
    # Construir la ruta completa del archivo CSV
    file_path = os.path.join(original_path, folder, f"{base_name}.csv")
    
    # Verificar si el archivo existe
    if os.path.exists(file_path):
        # Leer el CSV
        df = pd.read_csv(file_path)
        

        # Leer el archivo JSON de GAUFS con num_clusters estimado beforehand
        json_name = base_name.replace('results_', '') + '_silhouette_ward.json'
        json_path = os.path.join("./results_cluster_number_before_hand/results_seleccionados_sinteticos/", folder, 'analyisis_significance_and_num_cluster', json_name)
        
        print(original_path)
        print(folder)
        print(json_name)

        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # Extraer valores
            valores = json_data['dicc_soluciones']['(0.5, 1)']
            valor_3 = valores[3]
            
            # Guardar datos
            gaufs_cluster_beforehand.append(valor_3)

        # Leer el archivo JSON
        json_name = base_name.replace('results_', '') + '_silhouette_ward.json'
        json_path = os.path.join(original_path, folder, 'analyisis_significance_and_num_cluster', json_name)
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # Extraer valores
            valores = json_data['dicc_soluciones']['(0.5, 1)']
            valor_3 = valores[3]
            
            # Guardar datos
            folder_names.append(folder.replace('results_parallel_', ''))
            gaufs_original.append(valor_3)

# Crear figura con dos subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Gráfico de barras original
x = range(len(folder_names))
width = 0.35

bars1 = ax1.bar([i - width/2 for i in x], gaufs_cluster_beforehand, width, label='Valor con GAUFS estimating cluster number beforehand', alpha=0.8)
bars2 = ax1.bar([i + width/2 for i in x], gaufs_original, width, label='Valor Con GAUFS Original (Completo)', alpha=0.8)

ax1.set_xlabel('Carpetas')
ax1.set_ylabel('Valores')
ax1.set_title('Comparación AMI GAUFS estimating cluster number beforehand vs GAUFS en Sintéticos')
ax1.set_xticks(x)
ax1.set_xticklabels(folder_names, rotation=90, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Subplot 2: Violin plot con estadísticas
data_for_violin = [gaufs_cluster_beforehand, gaufs_original]
positions = [1, 2]

parts = ax2.violinplot(data_for_violin, positions=positions, showmeans=True, showmedians=True)

# Añadir estadísticas como texto
gaufs_cluster_beforehand_mean = np.mean(gaufs_cluster_beforehand)
gaufs_cluster_beforehand_std = np.std(gaufs_cluster_beforehand)
gaufs_original_mean = np.mean(gaufs_original)
gaufs_original_std = np.std(gaufs_original)

ax2.text(1, ax2.get_ylim()[1] * 0.95, f'Media: {gaufs_cluster_beforehand_mean:.4f}\nStd: {gaufs_cluster_beforehand_std:.4f}', 
         ha='center', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax2.text(2, ax2.get_ylim()[1] * 0.95, f'Media: {gaufs_original_mean:.4f}\nStd: {gaufs_original_std:.4f}', 
         ha='center', va='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

ax2.set_xticks([1, 2])
ax2.set_xticklabels(['GAUFS Estimating Cluster number beforehand', 'GAUFS Original'])
ax2.set_ylabel('Valores')
ax2.set_title('Distribución y Estadísticas en Sintéticos')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Imprimir estadísticas en consola
print(f"\nEstadísticas:")
print(f"GAUFS Estimating Cluster number beforehand - Media: {gaufs_cluster_beforehand_mean:.4f}, Desv. Típica: {gaufs_cluster_beforehand_std:.4f}")
print(f"GAUFS Original - Media: {gaufs_original_mean:.4f}, Desv. Típica: {gaufs_original_std:.4f}")
