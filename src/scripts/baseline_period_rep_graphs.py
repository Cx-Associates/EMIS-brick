#Points above or below 10% of the limit probably needs to be a different color or something to make it easily understandable. #Todo: This is for the
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from correlated_modeling import Report_df_final, file_path
from datetime import date
from dateutil.relativedelta import relativedelta
import pandas as pd

#Calculate hourly means
Report_df_final.index = pd.to_datetime(Report_df_final.index)
Report_df_final.index = Report_df_final.index.time
hourly_avg_df = Report_df_final.groupby(Report_df_final.index).mean()
hourly_avg_df.index = [time.strftime("%H:%M") for time in hourly_avg_df.index] # Converting the time index to strings for easier plotting
#hourly_avg_df.to_csv('hourly_avg_df.csv')

#Plotting Heating System

#For reporting
plt.figure(figsize=(10, 6))
plt.plot(hourly_avg_df.index, hourly_avg_df['Total Heating Plant Energy Consumption (MMBtu)'], marker='o', linestyle='-', color='b')
plt.title('Heating Plant Energy Consumption (MMBtu)')
plt.xlabel('Hour of the Day')
plt.ylabel('Average NG Consumption (MMBtu)')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent overlapping of labels
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\For MMM\HeatingSystem.png')
plt.close()

#Plotting AHU19
plt.figure(figsize=(10, 6))
plt.plot(hourly_avg_df.index, hourly_avg_df['AHU 19 Total kW (Correlated)'], marker='o', linestyle='-', color='b')
plt.title('AHU19 Energy Consumption (kW)')
plt.xlabel('Hour of the Day')
plt.ylabel('Energy Consumption (kW)')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent overlapping of labels
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\For MMM\AHU19.png')
plt.close()

#Plotting HRU
plt.figure(figsize=(10, 6))
plt.plot(hourly_avg_df.index, hourly_avg_df['HRU Total kW (Correlated)'], marker='o', linestyle='-', color='b')
plt.title('HRU Energy Consumption (kW)')
plt.xlabel('Hour of the Day')
plt.ylabel('Energy Consumption (kW)')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent overlapping of labels
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\For MMM\HRU.png')
plt.close()

#Plotting CHW System
plt.figure(figsize=(10, 6))
plt.plot(hourly_avg_df.index, hourly_avg_df['Total CHW kW'], marker='o', linestyle='-', color='b')
plt.title('CHW System Energy Consumption (kW)')
plt.xlabel('Hour of the Day')
plt.ylabel('Energy Consumption (kW)')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.tight_layout()  # Adjust layout to prevent overlapping of labels
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\For MMM\CHW System.png')
plt.close()

#Adding Equipment Level Data
plt.figure(figsize=(10, 6))
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 2a kW (Correlated)'], marker='o', linestyle='-', color='b', label='Pump 2a kW')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 2b kW (Correlated)'], marker='o', linestyle='-', color='r', label='Pump 2b kW')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 1a kW (Formula Based)'], marker='o', linestyle='-', color='g', label='Pump 1a kW')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 1b kW (Formula Based)'], marker='o', linestyle='-', color='c', label='Pump 1b kW')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 3a kW (Formula Based)'], marker='o', linestyle='-', color='m', label='Pump 3a kW')
plt.plot(hourly_avg_df.index, hourly_avg_df['Pump 3b kW (Formula Based)'], marker='o', linestyle='-', color='y', label='Pump 3b kW')
plt.plot(hourly_avg_df.index, hourly_avg_df['Tower Fan 1 kW (Correlated)'], marker='o', linestyle='-', color='k', label='Tower Fan 1 kW')
plt.plot(hourly_avg_df.index, hourly_avg_df['Tower Fan 2 kW (Correlated)'], marker='o', linestyle='-', color='orange', label='Tower Fan 2 kW')
plt.plot(hourly_avg_df.index, hourly_avg_df['Chiller kW'], marker='o', linestyle='-', color='chocolate', label='Chiller kW')
plt.title('CHW System Equipment Energy Consumption (kW)')
plt.xlabel('Hour of the Day')
plt.ylabel('Energy Consumption (kW)')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))  # Adjust location if needed
plt.tight_layout()  # Adjust layout to prevent overlapping of labels
plt.savefig(r'F:\PROJECTS\1715 Main Street Landing EMIS Pilot\code\Plots\For MMM\CHW System Equiipment.png')
plt.close()
