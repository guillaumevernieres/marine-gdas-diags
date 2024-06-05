import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import tarfile
import re
import gzip
import shutil
from datetime import datetime
import pandas as pd


def gunzip_file(file_path):
    # Define the name of the output file
    output_path = file_path.rstrip('.gz')

    # Decompress the file
    with gzip.open(file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Replace the original file with the decompressed file
    os.remove(file_path)
    os.rename(output_path, file_path.rstrip('.gz'))

def extract_files_matching_pattern(tar_path, pattern, extract_path='.'):
    # Open the tar file
    with tarfile.open(tar_path) as tar:
        # Iterate through each member of the tar file
        for member in tar.getmembers():
            # Check if the member name matches the pattern
            if re.search(pattern, member.name):
                print(f'Extracting {member.name}')
                tar.extract(member, path=extract_path)

# Function to calculate RMSE
def calculate_rmse(data):
    return np.sqrt(np.mean(np.square(data)))

def gsi_stats(file_path):
    # Open the NetCDF file
    ds = nc.Dataset(file_path, 'r')

    # Get the date
    date_time_str = ds.getncattr('date_time')
    date_time_format = datetime.strptime(str(date_time_str), '%Y%m%d%H')
    formatted_date = date_time_format.strftime('%Y-%m-%d %H:%M:%S')
    print(f'Formatted Date: {formatted_date}')

    # Read the required variables
    chaninfoidx = ds.variables['chaninfoidx'][:]
    obs_minus_forecast_adjusted = ds.variables['Obs_Minus_Forecast_unadjusted'][:]
    qc_flag = ds.variables['QC_Flag'][:]
    latitude = ds.variables['Latitude'][:]
    channel_index = ds.variables['Channel_Index'][:]

    # Get the number of channel
    n_channels = len(chaninfoidx)

    # Plot histograms for each channel
    mean_value = []
    rmse_value = []
    nobs = []
    for i, idx in enumerate(chaninfoidx):
        # Filter the data for the current channel and remove extreme values
        channel_data = obs_minus_forecast_adjusted[(channel_index == idx) &
                                                   (latitude < 90) &
                                                   (qc_flag == 0)]
        channel_data = channel_data[np.abs(channel_data) < 999]

        # Calculate stats
        mean_value.append(np.mean(channel_data))
        rmse_value.append(calculate_rmse(channel_data))
        nobs.append(len(channel_data))

    print(rmse_value)
    print('-----------------------')
    return nobs, mean_value, rmse_value, date_time_format

def create_experiment_dict(experiment_name, datetime_list, channel_stats):
    return {
        experiment_name: {
            "date": datetime_list,
            "stats": channel_stats
        }
    }

def create_channel_stats(rmse_list, bias_list, nobs):
    return {
        "rmse": rmse_list,
        "bias": bias_list,
        "nobs": nobs
    }

def plot_ts(ax, stats_dict, channel, exp, stat_type, color):
    # Sort the DataFrame by date to ensure monotonic time data
    print('******************* ', stats_dict['stats'][channel]['nobs'])
    df = pd.DataFrame({
        'date': stats_dict['date'],
        stat_type.lower(): stats_dict['stats'][channel][stat_type.lower()],
        'nobs': stats_dict['stats'][channel]['nobs']
    })
    df.sort_values(by='date', inplace=True)
    ax.plot(df['date'], df[stat_type.lower()], marker='.', linestyle='-', linewidth=3, label=exp, color=color)
    ax.set_title(channel, fontsize=18, fontweight='bold')
    ax.grid(True)

    axt = ax.twinx()
    print('&&&&&&&&&&&&&&&&& ',df['nobs'])
    axt.plot(df['date'], df['nobs'], marker='', linestyle='--', alpha=0.5, label=exp, color=color)

exp_dirs = ['../C02/COMROOT/C02/gdas.202107??/??/analysis/atmos/gdas.t??z.radstat',
            '../HR_3_5_atm_model_atm_da_arch/gdas.t??z.radstat.*',
            '../HR_3_5_s2s_model_atm_da/gdas.t??z.radstat.*']
exp_names = ['atmos-ocean', 'atmos','atmos-ocean-nosoca']
exp_colors = ['lightsalmon', 'lightgreen','lightsteelblue']
#inst = 'avhrr_metop-a'
inst = 'iasi_metop-b'
stat_type = 'RMSE'  #'BIAS'


fig, axs = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
fig.suptitle(f'{stat_type} {inst}', fontsize=18, fontweight='bold')
for exp_dir, exp_name, exp_color in zip(exp_dirs, exp_names, exp_colors):
    print(f'--------------------------------- {exp_name}')
    print(f'--------------------------------- {exp_dir}')
    channels = {'channel 0': create_channel_stats([], [], []),
                'channel 1': create_channel_stats([], [], []),
                'channel 2': create_channel_stats([], [], [])}

    exp_dict = create_experiment_dict(exp_name, [], channels)

    # Get list of NetCDF files in the input directory
    radstat_files = glob(exp_dir)
    radstat_files.sort()
    print(radstat_files)
    scratch_dir = './scratch'
    exp_mean = []
    exp_rmse = []
    exp_date = []
    for radstat_file in radstat_files[0:]:
        os.makedirs(scratch_dir, exist_ok=True)
        extract_files_matching_pattern(radstat_file, inst, extract_path=scratch_dir)
        gsi_stat_file = glob(os.path.join(scratch_dir, '*ges*'))[0]
        print(gsi_stat_file)
        gunzip_file(gsi_stat_file)
        gsi_stat_file = glob(os.path.join(scratch_dir, '*ges*'))[0]
        tmp_nobs, tmp_mean, tmp_rmse, tmp_date = gsi_stats(gsi_stat_file)

        exp_dict[exp_name]['date'].append(tmp_date)

        exp_dict[exp_name]['stats']['channel 0']['nobs'].append(tmp_nobs[0])
        exp_dict[exp_name]['stats']['channel 1']['nobs'].append(tmp_nobs[1])
        exp_dict[exp_name]['stats']['channel 2']['nobs'].append(tmp_nobs[2])

        exp_dict[exp_name]['stats']['channel 0']['rmse'].append(tmp_rmse[0])
        exp_dict[exp_name]['stats']['channel 1']['rmse'].append(tmp_rmse[1])
        exp_dict[exp_name]['stats']['channel 2']['rmse'].append(tmp_rmse[2])

        exp_dict[exp_name]['stats']['channel 0']['bias'].append(tmp_mean[0])
        exp_dict[exp_name]['stats']['channel 1']['bias'].append(tmp_mean[1])
        exp_dict[exp_name]['stats']['channel 2']['bias'].append(tmp_mean[2])
        os.remove(gsi_stat_file)

    plot_ts(axs[0], exp_dict[exp_name], 'channel 0', exp_name, stat_type, exp_color)
    plot_ts(axs[1], exp_dict[exp_name], 'channel 1', exp_name, stat_type, exp_color)
    plot_ts(axs[2], exp_dict[exp_name], 'channel 2', exp_name, stat_type, exp_color)

axs[0].legend()

# Format the x-axis to show datetime labels clearly
plt.gcf().autofmt_xdate()

# Save and/or show the plot
plt.savefig(f'{inst}_{stat_type}.png')
