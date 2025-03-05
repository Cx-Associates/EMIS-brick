#Points above or below 10% of the limit probably needs to be a different color or something to make it easily understandable. #Todo: This is for after basline is established
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import matplotlib.dates as mdates
from correlated_modeling import Report_df_final, subfolder_path, Month, csv_file_path, energy_history_df, total_energy_system_level, total_energy_system_corr
import subprocess
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import shutil

#Calculate hourly means
Report_df_final.index = Report_df_final.index.time
hourly_avg_df = Report_df_final.groupby(Report_df_final.index).mean()
hourly_avg_df.index = [time.strftime("%H:%M") for time in hourly_avg_df.index] #Converting the time index to strings for easier plotting
#hourly_avg_df.to_csv('hourly_avg_df.csv')

#Baseline

Heating_system_baseline = total_energy_system_corr['Heating Plant (Normalized)'].sum() #todo: Add more systems below

#Plotting Heating System

#For reporting
plt.figure(figsize=(10, 6))
plt.plot(hourly_avg_df.index, hourly_avg_df['Total Heating Plant Energy Consumption (MMBtu)'], marker='o', linestyle='-', color='b')
plt.title('Heating Plant Energy Consumption', fontsize = 16, pad=20)
plt.xlabel('Hour of the Day', fontsize = 14, labelpad=15)
plt.ylabel('Average NG Consumption (MMBtu)', fontsize = 14, labelpad=15)
plt.xticks(rotation=45, fontsize = 12)# Rotate x-axis labels for better readability
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent overlapping of labels
plt.subplots_adjust(top=0.88)
graph_path = os.path.join(subfolder_path, f"HeatingSystem.png")
plt.savefig(graph_path)
plt.close()

#Adding Equipment Level Data
plt.figure(figsize=(10, 6))
plt.plot(hourly_avg_df.index, hourly_avg_df['Total Boiler NG Consumption (MMBtu)'], marker='o', linestyle='-', color='steelblue', label = 'Boiler NG Consumption (MMBtu)')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 4a kW (Correlated)'], marker='v', linestyle='-', color='g', label = 'Pump 4a (kWh)')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 4b kW (Correlated)'], marker='^', linestyle='-', color='peru', label = 'Pump 4b (kWh)')
plt.title('Heating Plant Equipment Energy Consumption', fontsize = 16, pad=20)
plt.xlabel('Hour of the Day', fontsize = 14, labelpad=15)
plt.ylabel('Energy Consumption', fontsize = 14, labelpad=15)
plt.xticks(rotation=45, fontsize = 12)# Rotate x-axis labels for better readability
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.legend(loc='upper right', bbox_to_anchor=(1, 1.05))  # Place legend outside
plt.subplots_adjust(top=0.88)
graph_path = os.path.join(subfolder_path, f"HeatingSystemEquipment.png")
plt.savefig(graph_path)
plt.close()

#Plotting AHU19
plt.figure(figsize=(10, 6))
plt.plot(hourly_avg_df.index, hourly_avg_df['AHU 19 Total kW (Correlated)'], marker='o', linestyle='-', color='b')
plt.title('AHU19 Energy Consumption', fontsize = 16, pad=20)
plt.xlabel('Hour of the Day', fontsize = 14, labelpad=15)
plt.ylabel('Energy Consumption (kWh)', fontsize = 14, labelpad=15)
plt.xticks(rotation=45, fontsize = 12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
graph_path = os.path.join(subfolder_path, f"AHU19.png")
plt.savefig(graph_path)
plt.close()

#Plotting HRU
plt.figure(figsize=(10, 6))
plt.plot(hourly_avg_df.index, hourly_avg_df['HRU Total kW (Correlated)'], marker='o', linestyle='-', color='b')
plt.title('HRU Energy Consumption', fontsize = 16, pad=20)
plt.xlabel('Hour of the Day', fontsize = 14, labelpad=15)
plt.ylabel('Energy Consumption (kWh)', fontsize = 14, labelpad=15)
plt.xticks(rotation=45, fontsize = 12)
plt.yticks(fontsize = 12)
plt.grid(True)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
graph_path = os.path.join(subfolder_path, f"HRU.png")
plt.savefig(graph_path)
plt.close()

#Plotting CHW System
plt.figure(figsize=(10, 6))
plt.plot(hourly_avg_df.index, hourly_avg_df['Total CHW kW'], marker='o', linestyle='-', color='b')
plt.title('CHW System Energy Consumption', fontsize = 16, pad=20)
plt.xlabel('Hour of the Day', fontsize = 14, labelpad=15)
plt.ylabel('Energy Consumption (kWh)', fontsize = 14, labelpad=15)
plt.xticks(rotation=45, fontsize = 12)
plt.yticks(fontsize = 12)
plt.grid(True)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
graph_path = os.path.join(subfolder_path, f"CHWSystem.png")
plt.savefig(graph_path)
plt.close()

#Plotting just chiller
plt.figure(figsize=(10, 6))
plt.plot(hourly_avg_df.index, hourly_avg_df['Chiller kW'], marker='o', linestyle='-', color='darkkhaki')
plt.title('Chiller Energy Consumption', fontsize = 16, pad=20)
plt.xlabel('Hour of the Day', fontsize = 14, labelpad=15)
plt.ylabel('Energy Consumption (kWh)', fontsize = 14, labelpad=15)
plt.xticks(rotation=45, fontsize = 12)
plt.yticks(fontsize = 12)
plt.grid(True)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
graph_path = os.path.join(subfolder_path, f"Chiller.png")
plt.savefig(graph_path)
plt.close()

#Adding Equipment Level Data
plt.figure(figsize=(14, 6))

plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 1a kW (Formula Based)'], marker='^', linestyle='-', color='g', label='Pump 1a')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 1b kW (Formula Based)'], marker='s', linestyle='-', color='c', label='Pump 1b')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 3a kW (Formula Based)'], marker='p', linestyle='-', color='m', label='Pump 3a')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 3b kW (Formula Based)'], marker='x', linestyle='-', color='y', label='Pump 3b')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 2a kW (Correlated)'], marker='o', linestyle='-', color='b', label='Pump 2a')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 2b kW (Correlated)'], marker='v', linestyle='-', color='lightcoral', label='Pump 2b')
plt.plot(hourly_avg_df.index, hourly_avg_df['Tower Fan 1 kW (Correlated)'], marker='*', linestyle='-', color='k', label='Tower Fan 1')
plt.plot(hourly_avg_df.index, hourly_avg_df['Tower Fan 2 kW (Correlated)'], marker='h', linestyle='-', color='orange', label='Tower Fan 2')
plt.title('CHW System Ancillary Equipment Energy Consumption', fontsize = 18, pad=20)
plt.xlabel('Hour of the Day', fontsize = 16, labelpad=15)
plt.ylabel('Energy Consumption (kWh)', fontsize = 16, labelpad=15)
plt.xticks(rotation=45, fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))  # Adjust location if needed
plt.tight_layout()
plt.subplots_adjust(top=0.88)
graph_path = os.path.join(subfolder_path, f"CHWSystemAncillaryEquipment.png")
plt.savefig(graph_path)
plt.close()

#Adding Chiller to Equipment Level Data
plt.figure(figsize=(14, 6))
plt.plot(hourly_avg_df.index, hourly_avg_df['Chiller kW'], marker='.', linestyle='-', color='plum', label='Chiller')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 1a kW (Formula Based)'], marker='^', linestyle='-', color='g', label='Pump 1a')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 1b kW (Formula Based)'], marker='s', linestyle='-', color='c', label='Pump 1b')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 2a kW (Correlated)'], marker='o', linestyle='-', color='b', label='Pump 2a')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 2b kW (Correlated)'], marker='v', linestyle='-', color='black', label='Pump 2b')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 3a kW (Formula Based)'], marker='p', linestyle='-', color='m', label='Pump 3a')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 3b kW (Formula Based)'], marker='x', linestyle='-', color='y', label='Pump 3b')
plt.plot(hourly_avg_df.index, hourly_avg_df['Tower Fan 1 kW (Correlated)'], marker='*', linestyle='-', color='k', label='Tower Fan 1')
plt.plot(hourly_avg_df.index, hourly_avg_df['Tower Fan 2 kW (Correlated)'], marker='h', linestyle='-', color='orange', label='Tower Fan 2')
plt.title('CHW System All Equipment Energy Consumption', fontsize = 18, pad=20)
plt.xlabel('Hour of the Day', fontsize = 16, labelpad=15)
plt.ylabel('Energy Consumption (kWh)', fontsize = 16, labelpad=15)
plt.xticks(rotation=45, fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1, 1))  # Adjust location if needed
plt.tight_layout()
plt.subplots_adjust(top=0.88)
graph_path = os.path.join(subfolder_path, f"CHWSystemEquipment.png")
plt.savefig(graph_path)
plt.close()

#Total energy consumption graphs

#Total of all systems combined
energy_history_df.dropna(subset=["Month-Year", "Total Energy (MMBtu)"], inplace=True) #Just do it
months = energy_history_df["Month-Year"]
energy_values = energy_history_df["Total Energy (MMBtu)"]
fig, ax = plt.subplots(figsize=(12, len(months)*2))
energy_values = pd.to_numeric(energy_values, errors='coerce')
max_energy_value = energy_values.max()
#Create the gradient background
gradient = np.linspace(0, 1, int(max_energy_value) + 200).reshape(1, -1)
gradient = np.vstack((gradient, gradient))
#Plot each row with the corresponding month
for i, (month, value) in enumerate(zip(months, energy_values)):
    #Background gradient for each bar
    ax.imshow(gradient, aspect='auto', cmap='Oranges', extent=[0, max_energy_value + 200, i - 0.5, i + 0.5])

    #Plot the purple bar for energy consumption
    ax.barh(i, value, color='purple', height=0.5)

    #Add the label with energy consumption value, placing it just outside the boundary of the graph
    ax.text(max_energy_value + 230, i, f'{value} MMBtu', va='center', ha='left', color='black', fontsize=18)
ax.set_yticks(np.arange(len(months)))
ax.set_yticklabels(months, fontsize=18)
ax.set_xlim([0, max_energy_value + 200])  # Add a buffer for the label space
ax.set_ylim([-0.5, len(months) - 0.5])  # Adjust for the number of months
ax.set_xlabel('Energy Consumption (MMBtu)', fontsize=14, labelpad=15)
ax.set_title('Total Energy Consumption by Month', fontsize=16, pad=20)
ax.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
#plt.subplots_adjust(top=0.88)
graph_path = os.path.join(subfolder_path, f"TotalEnergy.png")
plt.savefig(graph_path)
plt.close()

#Total system level
systems = total_energy_system_level.columns
system_values = total_energy_system_level.iloc[0]
fig, ax = plt.subplots(figsize=(12, len(systems) * 2))
max_energy_value = system_values.max()
gradient = np.linspace(0, 1, int(max_energy_value) + 200).reshape(1, -1)
gradient = np.vstack((gradient, gradient))
for i, (system, value) in enumerate(zip(systems, system_values)):
    ax.imshow(gradient, aspect='auto', cmap='Oranges', extent=[0, max_energy_value + 200, i - 0.5, i + 0.5])
    ax.barh(i, value, color='purple', height=0.5)

    if system == "Heating Plant":
        ax.axvline(
            x=Heating_system_baseline,
            color='black',
            linestyle='-',
            linewidth=2,
            ymin=(i-0.5) / (len(systems)),  # Adjust ymin ensuring overlap with the correct row. #Todo: Find an automated method of doing this
            ymax=(i + 1) / (len(systems))  # Adjust ymax
        )

        difference = value - Heating_system_baseline
        difference_text = "above" if difference > 0 else "below"

        # Add the "how much higher/lower" label for heating system
        ax.text(
            max_energy_value + 230, i - 0.2,  # Slightly below the main label
            f'{abs(difference):.1f} MMBtu {difference_text} baseline',
            va='center', ha='left',
            color='darkred' if difference > 0 else 'green',  # Red for above, green for below
            fontsize=11
        )


    ax.text(max_energy_value + 230, i, f'{value:.1f} MMBtu', va='center', ha='left', color='black', fontsize=13)
ax.set_yticks(np.arange(len(systems)))
ax.set_yticklabels(systems, fontsize=14)
ax.set_xlim([0, max_energy_value + 200])
ax.set_ylim([-0.5, len(systems) - 0.5])
ax.set_xlabel('Energy Consumption (MMBtu)', fontsize=14, labelpad=15)
ax.set_title('System Level Energy Consumption', fontsize=16, pad=20)
ax.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
graph_path = os.path.join(subfolder_path, f"System_Level_Total_Energy.png")
plt.savefig(graph_path)
plt.close()

#Write the report via tex
tex_file_path = r"F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Reporting\Draft_1.tex" #Paths for the .tex file and the subfolder for the report
tex_copy_path = os.path.join(subfolder_path, f'{Month} EMIS Report.tex').replace("\\", "/")
shutil.copy(tex_file_path,tex_copy_path)

os.chdir(subfolder_path.replace("\\", "/")) #Change the working directory to the subfolder
try:
    subprocess.run(["lualatex", f"{Month} EMIS Report.tex"], check=True)  # Compile the copied .tex file
    print(f"Report successfully generated and saved in {subfolder_path}")
except subprocess.CalledProcessError as e:
    print(f"Error during report generation: {e}")