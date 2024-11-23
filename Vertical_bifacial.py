import pandas as pd
import pvlib
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import seaborn as sns
import osmnx as ox
import geopandas as gpd

# Creating consumption data set
df_germany = pd.read_csv('/Users/ansgarkamratowski/Desktop/Solar stuff/Total Load - Day Ahead _ Actual_202101010000-202201010000.csv')
df_germany['Datetime'] = pd.to_datetime(df_germany['Time (CET/CEST)'].str.split(' - ').str[0], format='%d.%m.%Y %H:%M')
# Drop the first two columns by position
# normalising total consumption with reference to 2008 Household data for comparison
df_germany['KWH'] = df_germany['Actual Total Load [MW] - Germany (DE)'].div(210779)
df_germany = df_germany.drop(df_germany.columns[:2], axis=1)
df_germany.set_index('Datetime', inplace=True)
df_germany= df_germany.resample('30min').sum()
df_germany['month'] = df_germany.index.month
df_germany['year'] = df_germany.index.year
df_germany['time_of_day'] = df_germany.index.time

# Reading household consumption data
read_file = pd.read_csv('/Users/ansgarkamratowski/Desktop/household_power_consumption.txt', sep = ';')
df_consumption = read_file.to_csv ('household_power_consumption.csv', index=None)
df_consumption = pd.read_csv('household_power_consumption.csv')
columns_to_drop = ['Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_3', 'Sub_metering_2']
df_consumption = df_consumption.drop(columns=columns_to_drop)
print(df_consumption.head())
# implement Nan for non integer i.e. missing values
columns_to_check = ['Global_active_power', 'Global_reactive_power']

for column in columns_to_check:
    df_consumption[column] = pd.to_numeric(df_consumption[column], errors='coerce')  # This will turn `?` and other non-numeric values into NaN

# Display the updated DataFrame to confirm
print(df_consumption.head())

df_consumption['apparent_power'] = np.sqrt(df_consumption['Global_active_power']**2 + df_consumption['Global_reactive_power']**2)
df_consumption['Datetime'] = pd.to_datetime(df_consumption['Date'] + ' ' + df_consumption['Time'], format='%d/%m/%Y %H:%M:%S')
df_consumption =df_consumption.drop(columns='Date')
df_consumption =df_consumption.drop(columns='Time')
df_consumption.set_index('Datetime', inplace=True)
df_consumption = df_consumption.resample('30min').mean().ffill()
df_consumption['KWH'] = df_consumption['apparent_power']*0.5

# https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.location.Location.html#pvlib.location.Location
def calculate_solar_production(location, times, orientations, tilt, system_capacity=9, efficiency=0.23):
    """Calculates production of panels based on an average of orentiations and tilt for a specified time and location."""
    # Get solar position and clear sky data
    solar_position = location.get_solarposition(times)
    clear_sky = location.get_clearsky(times)
    # Dictionary to store irradiance for each orientation
    irradiance_results = {}
    
    # Calculate irradiance for each orientation
    for orientation in orientations:
        irradiance = pvlib.irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=orientation,
            dni=clear_sky['dni'],
            ghi=clear_sky['ghi'],
            dhi=clear_sky['dhi'],
            solar_zenith=solar_position['apparent_zenith'],
            solar_azimuth=solar_position['azimuth']
        )
        irradiance_results[orientation] = irradiance['poa_global']  # Plane of Array irradiance in W/m²
    

    irradiance_df = pd.DataFrame(irradiance_results, index=times)    
    # Average irradiance across all orientations
    irradiance_df['average_irradiance'] = irradiance_df.mean(axis=1)
    
    # Convert irradiance to power output (kW) considering system capacity and efficiency
    irradiance_df['power_output_kw'] = irradiance_df['average_irradiance'] * (system_capacity * efficiency / 1000)
    production_df = irradiance_df['power_output_kw']
    
    return production_df

# Example usage
latitude = 52.52
longitude = 13.405
location = pvlib.location.Location(latitude, longitude)

times = pd.date_range('2006-12-15 23:00', '2010-11-26 21:00:00', freq='30min')
solar_position = location.get_solarposition(times)
clear_sky = location.get_clearsky(times)

# Define panel configurations
orientations_38deg = [90, 180, 270]
orientations_vertical_bifacial = [0, 45, 90, 135, 180, 225, 270, 315]
orientations_east_west_bifacial = [90, 270]
orientations_north_south_bifacial = [0, 180] 
#due to time zone differences we have to shift our data
df_consumption['production_38deg_south_only'] = calculate_solar_production(location, times, orientations=[180,180], tilt=38).shift(2)
df_consumption['production_38deg'] = calculate_solar_production(location, times, orientations_38deg, tilt=38).shift(2)
df_consumption['production_vertical_bifacial'] = (calculate_solar_production(location, times, orientations_vertical_bifacial, tilt=90) * 2).shift(2)
df_consumption['production_east_west_bifacial'] = (calculate_solar_production(location, times, orientations_east_west_bifacial, tilt=90) * 2).shift(2)
df_consumption['production_north_south_bifacial'] = (calculate_solar_production(location, times, orientations_north_south_bifacial, tilt=90) * 2).shift(2)

times = pd.date_range('2020-12-31 23:00:00', '2021-12-31 23:30:00', freq='30min')
df_germany['production_38deg_south_only'] = calculate_solar_production(location, times, orientations=[180,180], tilt=38).shift(2)
df_germany['production_38deg'] = calculate_solar_production(location, times, orientations_38deg, tilt=38).shift(2)
df_germany['production_vertical_bifacial'] = (calculate_solar_production(location, times, orientations_vertical_bifacial, tilt=90) * 2).shift(2)
df_germany['production_east_west_bifacial'] = (calculate_solar_production(location, times, orientations_east_west_bifacial, tilt=90) * 2).shift(2)
df_germany['production_north_south_bifacial'] = (calculate_solar_production(location, times, orientations_north_south_bifacial, tilt=90) * 2).shift(2)


# Filter DataFrame for the year 2008
df_2008 = df_consumption[df_consumption.index.year == 2008]
df_2008['month'] = df_2008.index.month
df_2008['year'] = df_2008.index.year
df_2008['time_of_day'] = df_2008.index.time
quarter_mapping = {
    1: 'Winter', 2: 'Winter', 12: 'Winter',   # Winter: December to February
    3: 'Spring', 4: 'Spring', 5: 'Spring',    # Spring: March to May
    6: 'Summer', 7: 'Summer', 8: 'Summer',    # Summer: June to August
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn'   # Autumn: September to November
}
# Map each month to its respective quarter, used for displaying production of solar systems
df_2008['quarter'] = df_2008['month'].map(quarter_mapping)

# Returns daily produciton and consumption curves
def production_consumption_daily(df):
    """Returns the production/consumption graph of a specified dataset and its systems"""
    systems = ['production_38deg_south_only', 'production_vertical_bifacial', 
        'production_east_west_bifacial', 'production_38deg', 
        'production_north_south_bifacial']
    average_daily_profile_yearly = df.groupby(['year', 'time_of_day'])[['KWH'] + systems].mean().reset_index()
    # Convert `time_of_day` to minutes from midnight for easier plotting
    average_daily_profile_yearly['minutes_from_midnight'] = average_daily_profile_yearly['time_of_day'].apply(lambda t: t.hour * 60 + t.minute)

    # Pivot the data to have `minutes_from_midnight` as rows and `year` as columns for each variable
    average_consumption_pivot = average_daily_profile_yearly.pivot(index='minutes_from_midnight', columns='year', values='KWH')

    # Create pivots for each system to plot production data
    production_pivots = {system: average_daily_profile_yearly.pivot(index='minutes_from_midnight', columns='year', values=system) for system in systems}

    plt.figure(figsize=(12, 8))
    # Plot consumption curves for each year
    for year in average_consumption_pivot.columns:
        plt.plot(average_consumption_pivot.index, average_consumption_pivot[year], label=f'{year} Consumption')
    # Plot production curves for each system and year
    for system, pivot_data in production_pivots.items():
        for year in pivot_data.columns:
            plt.plot(pivot_data.index, pivot_data[year], linestyle='--', label=f'{year} Production {system.replace("_", " ").capitalize()}')
    plt.xlabel('Time of Day')
    plt.ylabel('Average Energy (kWh)')
    plt.title(f'Average Daily Energy Consumption and Production Profile by Year with {df} consumption')
    plt.xticks(ticks=range(0, 1441, 120), labels=[f'{int(i/60):02d}:{i%60:02d}' for i in range(0, 1441, 120)])
    plt.legend(title="Year/System", loc='upper right')
    plt.grid(True)
    plt.tight_layout()

def daytime(dataset, start_time, end_time, start_month, end_month):
    """Helps you filter a dataset for a view of a specific daily/monthly time frame if you want to select all months use 1 and 12, or all hours 00:00 and 24:00"""
    daytime_data = dataset.between_time(start_time, end_time)
    daytime_data = daytime_data[(daytime_data.index.month >= start_month) & (daytime_data.index.month <= end_month)]
    return daytime_data

def totals(dataset):
   """Returns the yearly total/s of KWH for a system """
   return dataset.groupby(dataset.index.year)['KWH'].sum()

def simulate_battery(df, production_column, battery_capacity, outside_column):
    """"Simulates the charging behaviour of battery given a consumption dataset, its production column(if its part of the same dataframe)."""
    battery_charge = 0  # Initial battery charge in kWh
    surplus_total = 0
    deficit_total = 0
    saved_total = 0
    
    if outside_column=='no':
        # Loop through each time period in the DataFrame
        for index, row in df.iterrows():
            # Calculate initial surplus/deficit
            production = row[production_column]  # Production column
            usage = row['KWH']  # Usage column
            net_energy = production - usage
            
            # Update battery charge and surplus/deficit based on net energy
            if net_energy > 0:  # Surplus case
                battery_charge += net_energy
                if battery_charge > battery_capacity:
                    # Battery is full, set surplus to the remainder
                    surplus_total += battery_charge - battery_capacity
                    battery_charge = battery_capacity  # Cap battery at max capacity
                    saved_total += usage
                else:
                    saved_total += usage  # Save all production when battery is not full
            else:  # Deficit case
                if battery_charge + net_energy >= 0:
                    # Battery has enough to cover deficit
                    battery_charge += net_energy  # Reduce battery charge
                    saved_total += usage
                else:
                    # Battery depletes to zero, and there's still a deficit left
                    deficit_total += battery_charge + net_energy  # Remaining deficit
                    saved_total += (battery_charge + production)
                    battery_charge = 0  # Battery is empty
    else:
        for index, row in df.iterrows():
            # Calculate initial surplus/deficit
            usage = row['KWH']
            production = production_column.loc[index]
            net_energy = production - usage
            # Update battery charge and surplus/deficit based on net energy
            if net_energy > 0:  # Surplus case
                battery_charge += net_energy
                if battery_charge > battery_capacity:
                    # Battery is full, set surplus to the remainder
                    surplus_total += battery_charge - battery_capacity
                    battery_charge = battery_capacity  # Cap battery at max capacity
                    saved_total += usage
                else:
                    saved_total += usage  # Save all production when battery is not full
            else:  # Deficit case
                if battery_charge + net_energy >= 0:
                    # Battery has enough to cover deficit
                    battery_charge += net_energy  # Reduce battery charge
                    saved_total += usage
                else:
                    # Battery depletes to zero, and there's still a deficit left
                    deficit_total += battery_charge + net_energy  # Remaining deficit
                    saved_total += (battery_charge + production)
                    battery_charge = 0  # Battery is empty

    relative_surplus = surplus_total / (surplus_total + saved_total) if (surplus_total + saved_total) > 0 else 0


    return surplus_total, deficit_total, saved_total, relative_surplus

# Initialise capacities and systems
battery_capacities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # Example capacities in kWh
system_capacities = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
solar_array_columns = ['production_vertical_bifacial', 'production_38deg', 'production_east_west_bifacial','production_38deg_south_only', 'production_north_south_bifacial']
surplus_totals = []
deficit_totals = []
saved_totals = []


def quaterly_graph():
    """Returns winter, autumn, spring and summer production quaters for different systems"""
    average_quarterly = df_2008.groupby(['quarter'])[['KWH', 'production_38deg_south_only', 'production_vertical_bifacial', 'production_east_west_bifacial', 'production_38deg', 'production_north_south_bifacial']].sum().reset_index()
    quarters = average_quarterly['quarter']
    # Define the systems in the DataFrame (column names) to use as x labels
    systems = ['production_38deg_south_only', 'production_vertical_bifacial', 
            'production_east_west_bifacial', 'production_38deg', 
            'production_north_south_bifacial']
    totals = average_quarterly[systems].sum()

    # Plotting
    plt.figure(figsize=(10, 6))

    # Initialise bottom for stacking (in percentage terms, so start with zeros)
    bottom = [0] * len(systems)

    # Loop through each quarter and stack the values as percentages within each system bar
    for quarter in quarters:
        # Get values for each system for the current quarter and normalize to percentages
        values = average_quarterly[average_quarterly['quarter'] == quarter][systems].values.flatten()
        values_percentage = (values / totals) * 100  # Convert to percentage

        # Plot the percentage values, stacking them
        plt.bar(systems, values_percentage, label=f'Quarter {quarter}', bottom=bottom)
        bottom += values_percentage  # Update bottom for stacking

    plt.xlabel('System')
    plt.ylabel('Percentage of Total Production (%)')
    plt.title('Quarterly Stacked Energy Production by System (Percentage)')
    plt.legend(title='Quarter', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()

def battery_surplus_deficit(df, system):
    """Returns the surplus, deficit and saved energy for a particular consumption pattern'df' and a specific solar system 'system'"""
    surplus_totals = []
    deficit_totals = []
    saved_totals = []
    for capacity in battery_capacities:
        surplus, deficit, saved, relative_surplus = simulate_battery(df, production_column=system, battery_capacity=capacity, outside_column='no')
        surplus_totals.append(surplus)
        deficit_totals.append(deficit)
        saved_totals.append(saved)
    # Converts results into a DataFrame for easier plotting
    results_df = pd.DataFrame({
        'Battery Capacity (kWh)': battery_capacities,
        'Surplus': surplus_totals,
        'Deficit': deficit_totals,
        'Saved': saved_totals
    })

    # Plot stacked column chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(results_df['Battery Capacity (kWh)'], results_df['Deficit'], label='Deficit')
    ax.bar(results_df['Battery Capacity (kWh)'], results_df['Surplus'], label='Surplus')
    ax.bar(results_df['Battery Capacity (kWh)'], results_df['Saved'], bottom=results_df['Surplus'], label='Saved')

    # Labels and title
    ax.set_xlabel('Battery Capacity (kWh)')
    ax.set_ylabel('Energy (kWh)')
    ax.set_title(f'Energy Distribution by Battery Capacity for consumption pattern {df} with system: {system}')
    ax.legend()

    plt.show()

def line_graph_for_relative_surplus(df):
    """creates a line graph for a the relative surplus given different electricty generation systems given battery size."""
    results = []
    # Loop over each solar array and battery capacity
    for solar_array in solar_array_columns:
        for capacity in battery_capacities:
            surplus, deficit_total, saved, relative_surplus = simulate_battery(df, solar_array, capacity, outside_column='no')
            results.append({
                'Solar Array': solar_array,
                'Battery Capacity': capacity,
                'Surplus': surplus,
                'Saved': saved,
                'Relative Surplus': relative_surplus
            })

    # Convert results into a DataFrame for easy analysis
    results_df = pd.DataFrame(results)

    # Plotting the Relative Surplus for different capacities and arrays
    fig, ax = plt.subplots(figsize=(12, 6))
    for solar_array in solar_array_columns:
        subset = results_df[results_df['Solar Array'] == solar_array]
        ax.plot(subset['Battery Capacity'], subset['Relative Surplus'], label=solar_array)
    ax.set_xlabel('Battery Capacity (kWh)')
    ax.set_ylabel('Relative Surplus (Surplus / (Surplus + Saved))')
    ax.set_title(f'Relative Surplus by Battery Capacity for Different Solar Arrays {df}')
    ax.legend(title="Solar Array")

    plt.show()

def heatmap_metrics(df, battery_capacities, system_capacities, production_column):
    """Initalises initial heatmap metrics from which we can use in further analysis such as surplus, deficit and savings."""
    # Initialise a dictionary to store results for each metric
    results = {
        'SystemType': [],
        'battery_capacity': [],
        'system_capacity_kWp': [],
        'savings': [],
        'surplus': [],
        'deficit': []
    }
    
    # Loop over each system capacity (kWp)
    for system_capacity in system_capacities:
        # Scale production by system capacity, the saved system capacity is 9
        scaled_production = df[production_column] * (system_capacity / 9)
        
        # Loop over each battery capacity (kWh)
        for battery_capacity in battery_capacities:
            # Simulate battery performance with current capacities
            total_surplus, total_deficit, total_saved,relative_surplus = simulate_battery(
                df=df,
                production_column=scaled_production,
                battery_capacity=battery_capacity,
                outside_column='yes'
            )
            
            # Store the results
            results['SystemType'].append(production_column)
            results['battery_capacity'].append(battery_capacity)
            results['system_capacity_kWp'].append(system_capacity)
            results['savings'].append(total_saved)
            results['surplus'].append(total_surplus)
            results['deficit'].append(total_deficit)

    # Convert results to a DataFrame for analysis and heatmap creation
    results_df = pd.DataFrame(results)
    
    return results_df

def plot_heatmap(data, x, y, value, title, xlabel='Battery Capacity (kWh)', ylabel='System Capacity (kWp)'):
    """Function to plot heatmaps more easily"""
    # Corrected pivot syntax using named parameters
    heatmap_data = data.pivot(index=y, columns=x, values=value)
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".2f", annot_kws={"size": 8})
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.gca().invert_yaxis() 
    plt.show()

def all_results_array(df):
    """Function to intialise performance metrics for a dataset"""
    all_results_local = pd.DataFrame()  # Use a local variable

    # Loop over each system type and collect results
    for system_type in solar_array_columns:
        results = heatmap_metrics(df, battery_capacities, system_capacities, production_column=system_type)
        all_results_local = pd.concat([all_results_local, results], ignore_index=True)

    # Calculate metrics in the local DataFrame
    all_results_local['grid_independence'] = all_results_local['savings'] / (all_results_local['deficit'].abs() + all_results_local['savings'])
    all_results_local['self_use'] = all_results_local['savings'] / (all_results_local['surplus'] + all_results_local['savings'])
    all_results_local['grid_balance'] = all_results_local['savings'] / (all_results_local['deficit'].abs() + all_results_local['surplus'] + all_results_local['savings'])
    return all_results_local

# Call the function and assign the result globally
all_results = all_results_array(df_2008)


# Calculate best performing system on different metrics

# Group by SystemType and calculate the sum for each column
system_sums = all_results.groupby('SystemType').sum(numeric_only=True)
max_systems_for_sums = {}
# Loop through each column to find the SystemType with the maximum sum
for column in system_sums.columns:
    max_system = system_sums[column].idxmax()
    max_value = system_sums[column].max()
    max_systems_for_sums[column] = {
        'SystemType': max_system,
        'MaxSum': max_value
    }
# Display the results
for column, result in max_systems_for_sums.items():
    print(f"Column: {column}, SystemType: {result['SystemType']}, MaxSum: {result['MaxSum']}")


def Heatmap_results(system1, system2):
    """Displays the difference and individual in percentage points of performance metrics, as well as matching performances (measured battery and system capacities) in terms of manhattan distance as well as needed price increases for matched performances. These are displayed in heatmaps for different battery and system sizes."""
    results_1 = all_results[all_results['SystemType'] == system1]
    results_2 = all_results[all_results['SystemType'] == system2]
    for metric in ['grid_independence', 'self_use', 'grid_balance']:
        plot_heatmap(
            results_1,
            x='battery_capacity',
            y='system_capacity_kWp',
            value=metric,
            title=f'{metric.replace("_", " ").capitalize()} for Production {system1}'
        )
    for metric in ['grid_independence', 'self_use', 'grid_balance']:
        plot_heatmap(
            results_2,
            x='battery_capacity',
            y='system_capacity_kWp',
            value=metric,
            title=f'{metric.replace("_", " ").capitalize()} for {system2}'
        )
    merged_results = results_1.merge(
        results_2,
        on=['battery_capacity', 'system_capacity_kWp'],
        suffixes=('_1', '_2')
    )

    # Calculate difference metrics
    merged_results['grid_independence_diff'] = merged_results['grid_independence_1'] - merged_results['grid_independence_2']
    merged_results['self_use_diff'] = merged_results['self_use_1'] - merged_results['self_use_2']
    merged_results['grid_provision_diff'] = merged_results['grid_balance_1'] - merged_results['grid_balance_2']

    # Plotting difference metrics 
    plot_heatmap(
        merged_results,
        x='battery_capacity',
        y='system_capacity_kWp',
        value='grid_independence_diff',
        title=f'Grid Independence Difference ({system1} - {system2})'
    )

    plot_heatmap(
        merged_results,
        x='battery_capacity',
        y='system_capacity_kWp',
        value='self_use_diff',
        title=f'Self-Use Difference ({system1} - {system2})'
    )

    plot_heatmap(
        merged_results,
        x='battery_capacity',
        y='system_capacity_kWp',
        value='grid_provision_diff',
        title=f'Grid provision Difference ({system1} - {system2})'
    )
    # Finding closest matches if they exist
    closest_matches = []
    # Euclidian distance, can be exchanged with current Manhattan in code
    def calculate_tuple_distance(tuple1, tuple2):
        return np.sqrt((tuple1[0] - tuple2[0]) ** 2 + (tuple1[1] - tuple2[1]) ** 2)
    # Calculates closest matches for all metric
    for metric in ['grid_independence', 'self_use', 'grid_balance']:
        for idx, vert_row in results_1.iterrows():
            potential_matches = []

            vert_tuple = (vert_row['battery_capacity'], vert_row['system_capacity_kWp'])
            
            for _, south_row in results_2.iterrows():
                if south_row['battery_capacity'] >= vert_row['battery_capacity'] and south_row['system_capacity_kWp'] >= vert_row['system_capacity_kWp']:
                    self_use_diff = round(vert_row[metric] - south_row[metric], 2)
                    if self_use_diff <= 0:
                        south_tuple = (south_row['battery_capacity'], south_row['system_capacity_kWp'])
                        potential_matches.append(south_tuple)
            if not potential_matches:
                closest_matches.append({
                    'vertical_battery_capacity': vert_row['battery_capacity'],
                    'vertical_system_capacity_kWp': vert_row['system_capacity_kWp'],
                    'south_battery_capacity': np.nan,
                    'south_system_capacity_kWp': np.nan,
                    'tuple_distance': np.nan
                })
            else:
                closest_match = min(
                    potential_matches,
                    key=lambda match: (calculate_tuple_distance(vert_tuple, match), -match[1], -match[0])
                )
                
                # Calculate the Euclidean distance for the closest match
                distance = abs(vert_tuple[0]-closest_match[0]+vert_tuple[1]-closest_match[1])
                battery_distance = abs(vert_tuple[0]-closest_match[0])
                system_distance = abs(vert_tuple[1]-closest_match[1])
                
                closest_matches.append({
                    'metric': metric,
                    'vertical_battery_capacity': vert_row['battery_capacity'],
                    'vertical_system_capacity_kWp': vert_row['system_capacity_kWp'],
                    'south_battery_capacity': closest_match[0],
                    'south_system_capacity_kWp': closest_match[1],
                    'tuple_distance': distance,
                    'battery_distance': battery_distance,
                    'system_distance': system_distance,
                })

    # Convert the results to a DataFrame for easier analysis
    closest_matches_df = pd.DataFrame(closest_matches)
    metrics = ['self_use', 'grid_independence', 'grid_balance']

    # Creates heatmaps for matches
    for metric in metrics:
        filtered_data = closest_matches_df[closest_matches_df['metric'] == metric]
        # Create the heatmap for the current metric
        heatmap_data = filtered_data.pivot_table(
            index='vertical_system_capacity_kWp',
            columns='vertical_battery_capacity',
            values='tuple_distance'
        )
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".1f", cbar_kws={'label': 'Euclidean Distance'})
        plt.xlabel("System Capacity (kWp) - Vertical")
        plt.ylabel("Battery Capacity (kWh) - Vertical")
        plt.title(f"Manhattan Distance for {metric.capitalize()} Matches {system1} - {system2}")
        plt.gca().invert_yaxis()
        plt.tight_layout()


    # Creates price increase metric, using current(2024/2023) module and system prices
    closest_matches_df['price_increase'] = (
        closest_matches_df['battery_distance'] * 500 + 
        closest_matches_df['system_distance'] * 190
    ) / 1000

    # Iterate through each metric to create cost heatmap
    for metric in metrics:
        filtered_data = closest_matches_df[closest_matches_df['metric'] == metric]

        heatmap_data = filtered_data.pivot_table(
            index='vertical_system_capacity_kWp',
            columns='vertical_battery_capacity',
            values='price_increase'
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".1f", cbar_kws={'label': 'price_increase in thousands'})
        plt.xlabel("Battery Capacity (kWh)")
        plt.ylabel("System Capacity (kWp)")
        plt.title(f"Price difference for equivalent effect for {metric} and systems {system1} - {system2}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
def yearly_savings(system1, system2):
    """Shows yearly savings/losses for two systems using average energy prices and feed-in tarrif in Germany."""
    results_system1 = all_results[all_results['SystemType'] ==system1]
    results_system2 = all_results[all_results['SystemType'] == system2]

    merged_results = pd.merge(
        results_system1,
        results_system2,
        on=['battery_capacity', 'system_capacity_kWp'],
        suffixes=('_system1', '_system2')
    )
    # Calculate the money saved/gained per year
    merged_results['saved_gained_money'] = (
        (merged_results['savings_system1'] - merged_results['savings_system2']) * 0.3 +
        (merged_results['surplus_system1'] - merged_results['surplus_system2']) * 0.12
    )
    # Create a pivot table for the heatmap
    heatmap_data = merged_results.pivot_table(
        index='system_capacity_kWp',
        columns='battery_capacity',
        values='saved_gained_money'
    )
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".2f", cbar_kws={'label': 'Saved/Gained Money (€)'})
    plt.xlabel("Battery Capacity (kWh)")
    plt.ylabel("System Capacity (kWp)")
    plt.title("Saved/Gained Money Between Two Systems")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.show()
    merged_results.sum()