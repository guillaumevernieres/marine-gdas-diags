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
                tar.extract(member, path=extract_path)

# Function to calculate RMSE
def calculate_rmse(data):
    return np.sqrt(np.mean(np.square(data)))

def gsi_stats(file_path, channel_id, bias_corrected=True):
    # Open the NetCDF file
    ds = nc.Dataset(file_path, 'r')

    # Get the date
    date_time_str = ds.getncattr('date_time')
    date_time_format = datetime.strptime(str(date_time_str), '%Y%m%d%H')
    formatted_date = date_time_format.strftime('%Y-%m-%d %H:%M:%S')
    print(f'Formatted Date: {formatted_date}')

    # Read the required variables
    chaninfoidx = ds.variables['chaninfoidx'][:]
    omb_var = 'Obs_Minus_Forecast_unadjusted'
    if bias_corrected:
        omb_var = 'Obs_Minus_Forecast_adjusted'
    obs_minus_forecast_adjusted = ds.variables[omb_var][:]
    qc_flag = ds.variables['QC_Flag'][:]
    latitude = ds.variables['Latitude'][:]
    channel_index = ds.variables['Channel_Index'][:]

    # Get the number of channel
    n_channels = len(chaninfoidx)

    # Plot histograms for each channel
    mean_value = []
    rmse_value = []
    nobs = []

    # Filter the data for the current channel and remove extreme values
    channel_data = obs_minus_forecast_adjusted[(channel_index == channel_id) &
                                               (latitude < 90) &
                                               (qc_flag == 0)]
    channel_data = channel_data[np.abs(channel_data) < 999]

    # Calculate stats
    mean_value.append(np.mean(channel_data))
    rmse_value.append(calculate_rmse(channel_data))
    nobs.append(len(channel_data))

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
    axt.plot(df['date'], df['nobs'], marker='', linestyle='--', alpha=0.5, label=exp, color=color)

def plot_exps(exp_dirs, exp_names, exp_colors, inst, channel_id, stat_type, bias_corrected):
    fig, axs = plt.subplots(1, 1, figsize=(15, 10), sharex=True)
    bias_str = ''
    if bias_corrected:
        bias_str = 'bias corrected'
    fig.suptitle(f'{stat_type} {inst.upper()} {bias_str}', fontsize=18, fontweight='bold')
    for exp_dir, exp_name, exp_color in zip(exp_dirs, exp_names, exp_colors):
        channels ={}
        channels[f'channel {channel_id}'] = create_channel_stats([], [], [])

        exp_dict = create_experiment_dict(exp_name, [], channels)

        # Get list of NetCDF files in the input directory
        radstat_files = glob(exp_dir)
        radstat_files.sort()
        scratch_dir = './scratch'
        exp_mean = []
        exp_rmse = []
        exp_date = []
        for radstat_file in radstat_files[0:]:
            os.makedirs(scratch_dir, exist_ok=True)
            extract_files_matching_pattern(radstat_file, inst, extract_path=scratch_dir)
            gsi_stat_file = glob(os.path.join(scratch_dir, '*ges*'))[0]
            gunzip_file(gsi_stat_file)
            gsi_stat_file = glob(os.path.join(scratch_dir, '*ges*'))[0]
            tmp_nobs, tmp_mean, tmp_rmse, tmp_date = gsi_stats(gsi_stat_file, channel_id, bias_corrected)

            exp_dict[exp_name]['date'].append(tmp_date)
            exp_dict[exp_name]['stats'][f'channel {channel_id}']['nobs'].append(tmp_nobs[0])
            exp_dict[exp_name]['stats'][f'channel {channel_id}']['rmse'].append(tmp_rmse[0])
            exp_dict[exp_name]['stats'][f'channel {channel_id}']['bias'].append(tmp_mean[0])

            os.remove(gsi_stat_file)

        for channel in exp_dict[exp_name]['stats'].keys():
            print(channel)

        plot_ts(axs, exp_dict[exp_name], channel, exp_name, stat_type, exp_color)

    axs.legend()

    # Format the x-axis to show datetime labels clearly
    plt.gcf().autofmt_xdate()

    # Save and/or show the plot
    fname = f'{inst}_channel{channel_id}_{stat_type}.png'
    if bias_corrected:
        fname = f'{inst}_channel{channel_id}_{stat_type}_bias_corrected.png'
    plt.savefig(fname)

exp_dirs = ['../C02/COMROOT/C02/gdas.202107??/??/analysis/atmos/gdas.t??z.radstat',
            '../HR_3_5_atm_model_atm_da_arch/gdas.t??z.radstat.*',
            '../HR_3_5_s2s_model_atm_da/gdas.t??z.radstat.*']
exp_names = ['atmos-ocean', 'atmos','atmos-ocean-nosoca']
exp_colors = ['lightsalmon', 'lightgreen','lightsteelblue']
inst = 'avhrr_metop-b'
#bias_corrected = True
for inst in ['avhrr_metop-a', 'avhrr_metop-b']:
    for bias_corrected in [True, False]:
        for stat_type in ['RMSE', 'BIAS']:
            for channel_id in [1, 2, 3]:
                plot_exps(exp_dirs, exp_names, exp_colors, inst, channel_id, stat_type, bias_corrected)
