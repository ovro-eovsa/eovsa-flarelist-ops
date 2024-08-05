##ipython flarelist2sql.py
##=============
from __future__ import print_function
import socket
import requests
from bs4 import BeautifulSoup
import re
import os
import sys
from scipy.signal import find_peaks
import matplotlib
import argparse
import sys

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.dates import AutoDateFormatter, AutoDateLocator, num2date
import matplotlib.cm
# import matplotlib.image as mpimg
from PIL import Image
from datetime import datetime, timedelta
import warnings
from matplotlib import MatplotlibDeprecationWarning
import pandas as pd
import numpy as np
from astropy.time import Time
import mysql.connector
from datetime import datetime
from astropy.io import fits

warnings.filterwarnings('ignore', category=FutureWarning)

# Global settings and initializations
# EO_WIKI_URL = "http://www.ovsa.njit.edu/wiki/index.php/Recent_Flare_List_(2021-)"
EO_WIKI_URLs = [
    "https://www.ovsa.njit.edu/wiki/index.php/2019",
    "https://www.ovsa.njit.edu/wiki/index.php/2020",
    "https://www.ovsa.njit.edu/wiki/index.php/2021",
    "https://www.ovsa.njit.edu/wiki/index.php/2022",
    "https://www.ovsa.njit.edu/wiki/index.php/2023",
    "https://www.ovsa.njit.edu/wiki/index.php/2024"
]


## todo add command line ar for time and frequency input to register a flare event and update the sql.
def create_flare_db_connection():
    return mysql.connector.connect(
        host=os.getenv('FLARE_DB_HOST'),
        database=os.getenv('FLARE_DB_DATABASE'),
        user=os.getenv('FLARE_DB_USER'),
        password=os.getenv('FLARE_DB_PASSWORD')
    )


def create_flare_lc_db_connection():
    return mysql.connector.connect(
        host=os.getenv('FLARE_DB_HOST'),
        database=os.getenv('FLARE_LC_DB_DATABASE'),
        user=os.getenv('FLARE_DB_USER'),
        password=os.getenv('FLARE_DB_PASSWORD')
    )


##============= bad data
bad_data = ['EOVSA_20221013_M1flare.dat',
            'EOVSA_20230427_Cflare.dat',
            'EOVSA_20230429_Cflare.dat',
            'EOVSA_20230508_C9flare.dat',
            'EOVSA_20230520_M56flare.dat']

bad_data_mark = ['2022-10-13 00:10:00',
                 '2023-04-27 01:00:00',
                 '2023-04-29 23:02:00',
                 '2023-05-08 14:19:00',
                 '2023-05-20 14:58:00']

## List of URLs to download files from. this is not needed if running the code on ovsa and pipeline.
datfile_urls = [
    "http://ovsa.njit.edu/events/2019/",
    "http://ovsa.njit.edu/events/2020/",
    "http://ovsa.njit.edu/events/2021/",
    "http://ovsa.njit.edu/events/2022/",
    "http://ovsa.njit.edu/events/2023/",
    "http://ovsa.njit.edu/events/2024/"
]

## plotting config

fontsize_pl = 14.
window_size = 10

# Path to the folder containing the static images

static_img_folder = '/var/www/html/flarelist/static/images/'

# Target height for all logos
FIG_DPI = 150
LOGO_HEIGHT = 60  # Adjust this value as needed


# Load and resize logos
def load_and_resize_logo(path, target_height):
    img = Image.open(path)
    aspect_ratio = img.width / img.height
    new_height = target_height
    new_width = int(aspect_ratio * new_height)
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return np.array(resized_img)


def add_logos_horizontally(fig, dpi, logos, gap=4, right_offset=200, top_offset=30):
    """
    Adds pre-loaded and resized logos horizontally to a matplotlib figure.

    Parameters:
    - fig: Matplotlib figure object to which the logos will be added.
    - logos: List of numpy array representations of logos. Assumes logos are already resized.
    - gap: Pixels between logos. Default is 10 pixels.
    - right_offset: Pixels from the right edge of the figure to start placing logos. Default is 520 pixels.
    - top_offset: Pixels from the top edge of the figure to place the logos. Default is 80 pixels.
    """
    # Get figure dimensions
    fig_width, fig_height = fig.get_size_inches() * dpi

    # Calculate total width needed for all logos and gaps
    total_width = sum(logo.shape[1] for logo in logos) + gap * (len(logos) - 1)

    # Calculate starting positions
    x_offset = fig_width - total_width - right_offset
    y_offset = fig_height - top_offset  # Adjusted to align logos by their top edge

    # Place logos on the figure
    for logo in logos:
        fig.figimage(logo, xo=x_offset, yo=y_offset - logo.shape[0],
                     zorder=1000)  # Adjust y_offset for each logo based on its height
        x_offset += logo.shape[1] + gap  # Move right for the next logo


# Load the logo images
try:
    nsf_logo = load_and_resize_logo(os.path.join(static_img_folder, 'NSF_logo.png'), LOGO_HEIGHT)
    njit_logo = load_and_resize_logo(os.path.join(static_img_folder, 'njit-logo.png'), LOGO_HEIGHT)
    eovsa_logo = load_and_resize_logo(os.path.join(static_img_folder, 'eovsa_logo.png'), LOGO_HEIGHT)
    logos = [eovsa_logo, nsf_logo, njit_logo]
except:
    logos = ''


def fetch_flare_data_from_wiki(eo_wiki_urls, given_date_strp, outcsvfile):
    """
    Fetches flare data from the given wiki URL and saves it to a CSV file in the specified directory.

    Parameters:
    - eo_wiki_url: URL of the EO wiki page to fetch data from.
    - given_date_strp: A tuple of datetime objects specifying the start and end dates to filter the data.
    - outcsvfile: path to the resulting CSV file.

    Returns:
    - None
    """
    date_data = []
    time_ut_data = []
    flare_class_data = []
    depec_file = []
    depec_img = []

    for eo_wiki_url in eo_wiki_urls:
        response = requests.get(eo_wiki_url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content of the page using BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")
            tables = soup.find_all("table", {"class": "wikitable"})

            for table in tables:
                for row in table.find_all("tr"):
                    cells = row.find_all("td")

                    if len(cells) >= 3:
                        date = cells[0].text.strip()
                        time_ut = cells[1].text.strip()
                        flare_class = cells[2].text.strip()
                        datetime_strp = datetime.strptime(date + ' ' + time_ut, '%Y-%m-%d %H:%M')

                        if given_date_strp[0] <= datetime_strp <= given_date_strp[1]:
                            date_data.append(date)
                            time_ut_data.append(time_ut)
                            flare_class_data.append(flare_class)

                            depec_file_tmp = ''
                            for cell in cells:
                                link_cell = cell.find('a', class_='external text', href=True, rel='nofollow')
                                if link_cell:
                                    url = link_cell['href']
                                    if url.endswith(".dat") or url.endswith(".fits"):
                                        depec_file_tmp = url.split('/')[-1]
                                        break
                            depec_file.append(depec_file_tmp)

                            depec_img_tmp = ''
                            for cell in cells:
                                if cell.find(class_="thumbimage"):
                                    img_tag = cell.find('img')
                                    if img_tag:
                                        src_attribute = img_tag.get(
                                            'src')  ##'/wiki/images/a/ac/EOVSA_20240212_C5flare.png'
                                        if src_attribute:
                                            depec_img_tmp = src_attribute.split('/')[-1]
                            depec_img.append(depec_img_tmp)
        else:
            print("Failed to retrieve the webpage. Status code:", response.status_code)

    data = {
        "ID": np.arange(len(date_data)) + 1,
        "Date": date_data,
        "Time (UT)": time_ut_data,
        "Flare Class": flare_class_data,
        "depec_file": depec_file,
        "depec_img": depec_img
    }

    df = pd.DataFrame(data)
    df.to_csv(outcsvfile, index=False)
    print(f"Date and Time (UT) data saved to {outcsvfile}")


def get_flare_info_from_GOES(tpeak_str):
    # This function should return a tuple with GOES class, start, peak, end times, and optionally X and Y coordinates.
    # Example return format: ('M1.0', '2024-02-23 21:00:00', '2024-02-23 22:00:00', '2024-02-23 23:00:00', None, None)
    from astropy.time import Time
    from sunpy.net import Fido
    from sunpy.net import attrs as a
    import numpy as np

    tpeak = Time(tpeak_str, format='isot').mjd * 24.  ##in hours

    tstart = tpeak - 1.
    tend = tpeak + 1.

    tstart_str = ((Time(tstart / 24., format='mjd').isot).replace('-', '/')).replace('T', ' ')
    tend_str = ((Time(tend / 24., format='mjd').isot).replace('-', '/')).replace('T', ' ')
    # tstart_str = "2019/04/15 12:00:00"
    # tend_str = "2019/04/16 12:00:00"
    # event_type = "FL"
    try:
        result = Fido.search(a.Time(tstart_str, tend_str),
                             # a.hek.EventType(event_type),
                             # a.hek.FL.GOESCls > "M1.0",
                             a.hek.OBS.Observatory == "GOES")

        hek_results = result["hek"]
        #print(hek_results.colnames[::2])
        #print(result.show("hpc_bbox", "refs"))

        filtered_results = hek_results["fl_goescls", "event_starttime", "event_peaktime",
        "event_endtime", "ar_noaanum", "hgc_x", "hgc_y"]

        GOES_tpeak = hek_results["event_peaktime"]
        GOES_tpeak_mjd = GOES_tpeak.mjd * 24.

        if len(GOES_tpeak) == 1:
            ind = 0
        if len(GOES_tpeak) > 1:
            ind = np.argmin(abs(GOES_tpeak_mjd - tpeak))
        if len(GOES_tpeak) < 1:
            print("ERRORs: No flares detected")
            return

        GOES_class = (hek_results["fl_goescls"])[ind]
        GOES_tstart = (hek_results["event_starttime"])[ind].iso
        GOES_tpeak = (hek_results["event_peaktime"])[ind].iso
        GOES_tend = (hek_results["event_endtime"])[ind].iso
        GOES_hgc_x = (hek_results["hgc_x"])[ind]
        GOES_hgc_y = (hek_results["hgc_y"])[ind]

    except:
        GOES_class = '?'
        GOES_tstart = Time((tpeak-0.0)/24., format='mjd').iso
        GOES_tpeak = Time(tpeak/24., format='mjd').iso
        GOES_tend = Time((tpeak+0.0)/24., format='mjd').iso
        GOES_hgc_x = 0
        GOES_hgc_y = 0

    print(f"GOES Class {GOES_class} peaks on {GOES_tpeak} / EO radio on {tpeak_str}")
    return GOES_class, GOES_tstart, GOES_tpeak, GOES_tend, GOES_hgc_x, GOES_hgc_y


def download_datfiles_from_url(url, download_directory, timerange):
    # Send an HTTP GET request to the webpage
    response = requests.get(url)
    file_name_download = []

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all links on the page
        links = soup.find_all('a')

        # Loop through the links and download files
        for link in links:
            href = link.get('href')
            if href and not href.startswith('#'):
                # Ensure that the link is an absolute URL
                if not href.startswith('http'):
                    href = url + href

                # Get the file name from the URL
                file_name = os.path.basename(href)
                match = None

                if file_name.split('.')[-1] == 'dat':
                    # print(file_name)
                    match = re.match(r'EOVSA_(\d{4})(\d{2})(\d{2})', file_name)
                    if match == None:
                        match = re.match(r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', file_name)
                if file_name.split('.')[-1] == 'fits':
                    match = re.match(r'eovsa.spec.flare_id_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', file_name)

                # Check if the file name matches the given date range pattern
                if match:
                    file_date = match.group(1) + '-' + match.group(2) + '-' + match.group(3)

                    # Check if the file date is within the given date range
                    if timerange[0].strftime('%Y-%m-%d') <= file_date <= (
                            timerange[1] + timedelta(days=1)).strftime('%Y-%m-%d'):
                        # Download the file
                        if os.path.exists(os.path.join(download_directory, file_name)):
                            print(f"Exist: {file_name}")
                            continue  # Skip to the next iteration

                        try:
                            with open(os.path.join(download_directory, file_name), 'wb') as file:
                                file.write(requests.get(href).content)
                            print(f"Downloaded: {file_name}")
                            file_name_download.append(file_name)
                        except Exception as e:
                            print(f"Failed to download: {file_name} - {str(e)}")
    else:
        print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
    return file_name_download


def rd_datfile(file):
    ''' Read EOVSA binary spectrogram file and return a dictionary with times
        in Julian Date, frequencies in GHz, and cross-power data in sfu.

        Return Keys:
          'time'     Numpy array of nt times in JD format
          'fghz'     Numpy array of nf frequencies in GHz
          'data'     Numpy array of size [nf, nt] containing cross-power data

        Returns empty dictionary ({}) if file size is not compatible with inferred dimensions
    '''
    import struct
    import numpy as np
    def dims(file):
        # Determine time and frequency dimensions (assumes the file has fewer than 10000 times)
        f = open(file, 'rb')
        tmp = f.read(83608)  # max 10000 times and 451 frequencies
        f.close()
        nbytes = len(tmp)
        tdat = np.array(struct.unpack(str(int(nbytes / 8)) + 'd', tmp[:nbytes]))
        nt = np.where(tdat < 2400000.)[0]
        nf = np.where(np.logical_or(tdat[nt[0]:] > 18, tdat[nt[0]:] < 1))[0]
        return nt[0], nf[0]

    nt, nf = dims(file)
    f = open(file, 'rb')
    tmp = f.read(nt * 8)
    times = struct.unpack(str(nt) + 'd', tmp)
    tmp = f.read(nf * 8)
    fghz = struct.unpack(str(nf) + 'd', tmp)
    tmp = f.read()
    f.close()
    if len(tmp) != nf * nt * 4:
        print('File size is incorrect for nt=', nt, 'and nf=', nf)
        return {}
    data = np.array(struct.unpack(str(nt * nf) + 'f', tmp)).reshape(nf, nt)
    return {'time': times, 'fghz': fghz, 'data': data}


def spec_rebin(time, freq, spec, t_step=1, f_step=1, do_mean=True):
    """
    Rebin a spectrogram array to a new resolution in time and frequency.
    """
    import numpy as np
    tnum_steps = len(time) // t_step + (1 if len(time) % t_step != 0 else 0)
    fnum_steps = len(freq) // f_step + (1 if len(freq) % f_step != 0 else 0)

    time_new = np.array([time[i * t_step] for i in range(tnum_steps)])
    freq_new = np.array([freq[i * f_step] for i in range(fnum_steps)])

    spec_new = np.zeros((fnum_steps, tnum_steps))

    # Rebin the spectrogram
    if do_mean:
        for i in range(fnum_steps):
            for j in range(tnum_steps):
                spec_slice = spec[i * f_step:min((i + 1) * f_step, len(freq)),
                             j * t_step:min((j + 1) * t_step, len(time))]
                spec_new[i, j] = np.mean(spec_slice)
    else:
        for i in range(fnum_steps):
            for j in range(tnum_steps):
                spec_new[i, j] = spec[i * f_step, j * t_step]

    return time_new, freq_new, spec_new


def moving_average(data, window_size):
    # Create a convolution kernel for the moving average
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Fetch flarelist data from wiki and write mySQL.',
                                     epilog='Example usage:\n'
                                            '  python flarelist2sql.py --timerange "2024-02-23 22:00:00" "2024-03-05 00:00:00" --do_manu 1\n'
                                            '  python flarelist2sql.py -t "2024-02-23 22:00:00" "2024-03-05 00:00:00"\n'
                                            '  python flarelist2sql.py\n\n'
                                            'The default time range is from 13:00 UT a week ago to now. '
                                            'The default for do_manu is 0.',
                                     formatter_class=argparse.RawTextHelpFormatter)  # Use RawTextHelpFormatter for better formatting of the epilog

    parser.add_argument('--timerange', '-t', type=str, nargs=2,
                        default=[(datetime.utcnow() - timedelta(days=7)).strftime('%Y-%m-%d 13:00:00'),
                                 datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')],
                        help='Time range for the analysis, formatted as "YYYY-MM-DD HH:MM:SS YYYY-MM-DD HH:MM:SS".\nDefaults to 7 days ago from 13:00 UT to now.')

    parser.add_argument('--do_manu', type=int, default=0,
                        help='Manually determine the start/end time of the radio burst by clicking on the Dspec.\nDefault: 0.')

    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()

    timerange = args.timerange
    do_manu = args.do_manu

    # Convert timerange strings to datetime objects for internal use
    timerange_strp = [datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in timerange]

    print(f"Fetching data for time range {timerange[0]} to {timerange[1]} with manual mode set to {do_manu}")

    hostname = socket.gethostname()

    timenow = datetime.now().strftime('%Y%m%dT%H%M%S')
    on_server = 0
    if hostname == "ovsa":
        work_dir = os.path.join(os.getenv('HOME'), 'workdir', 'eo_flarelist_update', timenow)
        on_server = 1
    elif hostname == "pipeline":
        work_dir = os.path.join('/data1/workdir', 'eo_flarelist_update', timenow)
        on_server = 1
    else:
        work_dir = os.path.join('.', 'eo_flarelist_update', timenow)
    os.makedirs(work_dir, exist_ok=True)

    print("##=============Step 1: capture the radio peak times from flare list wiki webpage")
    init_csv = os.path.join(work_dir, "get_time_from_wiki_given_date.csv")
    fetch_flare_data_from_wiki(EO_WIKI_URLs, timerange_strp, init_csv)

    print("##=============Step 2: reformate the date and time")
    # Read the initial CSV file
    df = pd.read_csv(init_csv)

    # Prepare lists to collect the updated data
    new_data = {
        "ID": [],
        "Flare_ID": [],
        "Date": [],
        "Time (UT)": [],
        "flare_class": [],
        "GOES_flare_class": [],
        "GOES_tstart": [],
        "GOES_tpeak": [],
        "GOES_tend": [],
        # "GOES_hgc_x": [],
        # "GOES_hgc_y": [],
        "depec_file": [],
        "depec_img": []
    }

    # Process each row in the DataFrame
    for index, row in df.iterrows():
        date = row["Date"]
        time = f"{row['Time (UT)']}:00"  # Directly add ':00' to the time string
        eotime_flare_wiki = f"{date.replace('/', '-')}T{time}"

        GOES_class_tp, GOES_tstart_tp, GOES_tpeak_tp, GOES_tend_tp, GOES_hgc_x_tp, GOES_hgc_y_tp = get_flare_info_from_GOES(
            tpeak_str=eotime_flare_wiki)

        flare_id_tp = eotime_flare_wiki.replace('-', '').replace('T', '').replace(':', '')

        # Update the new_data dictionary
        new_data["ID"].append(index + 1)
        new_data["Flare_ID"].append(flare_id_tp)
        new_data["Date"].append(date)
        new_data["Time (UT)"].append(time)
        new_data["flare_class"].append(row["Flare Class"])
        new_data["GOES_flare_class"].append(GOES_class_tp)
        new_data["GOES_tstart"].append(GOES_tstart_tp.split('.')[0])
        new_data["GOES_tpeak"].append(GOES_tpeak_tp.split('.')[0])
        new_data["GOES_tend"].append(GOES_tend_tp.split('.')[0])
        # new_data["GOES_hgc_x"].append(GOES_hgc_x_tp)
        # new_data["GOES_hgc_y"].append(GOES_hgc_y_tp)
        new_data["depec_file"].append(str(row["depec_file"]))
        new_data["depec_img"].append(str(row["depec_img"]))

    # Create a new DataFrame from the collected data
    new_df = pd.DataFrame(new_data)

    # Save the updated DataFrame to a new CSV file.
    updated_csv = init_csv.replace('.csv', '_updated.csv')
    new_df.to_csv(updated_csv, index=False)

    print("Times data saved to", init_csv)

    ##=============Step 3: download the spectrum data from flare list wiki webpage=============
    print("##=============Step 3: download the spectrum data from flare list wiki webpage")

    if on_server == 1:
        spec_data_dir = "/common/webplots/events/"  # YYYY/
        print("Spec data in ", spec_data_dir)
    else:
        # Directory to store downloaded files
        spec_data_dir = os.path.join(work_dir, 'spec_data/')
        os.makedirs(spec_data_dir, exist_ok=True)
        # # Date range to filter files (e.g., 'yyyy-mm-dd')
        # given_date = ['2023-09-01', '2023-09-10']
        # Iterate through the list of URLs and download files from each
        for url in datfile_urls:
            file_name_download = download_datfiles_from_url(url, spec_data_dir, timerange)

    ##=============Step 4: plot the spectrum  and determine the start and end times of radio flux profiles=============
    ##=============Step 4: read and plot the spectrum data=============

    print("##=============Step 4: read and plot the spectrum data")

    ##=============
    # Create a function to handle mouse click events
    def onclick(event):
        global click_count, x1, y1, x2, y2
        if event.dblclick:
            if click_count == 0:
                x1, y1 = event.xdata, event.ydata
                click_count += 1
                print(f"First click: ({x1:.2f}, {y1:.2f})")
            elif click_count == 1:
                x2, y2 = event.xdata, event.ydata
                click_count += 1
                print(f"Second click: ({x2:.2f}, {y2:.2f})")
                fig.canvas.mpl_disconnect(cid)  # Disconnect the click event
                plt.close(fig)  # Close the figure
        else:
            print("Please double-click to select two positions.")

    df = pd.read_csv(updated_csv)
    flare_id = df['Flare_ID']
    depec_file = df['depec_file']

    if on_server == 1:
        files_wiki = [spec_data_dir + str(flare_id[i])[0:4] + "/" + str(file_name) for i, file_name in
                      enumerate(depec_file)]
    else:
        files_wiki = [os.path.join(spec_data_dir, f'{str(file_name)}') for file_name in depec_file]

    spec_img_dir = os.path.join(work_dir, 'spec_img/')
    os.makedirs(spec_img_dir, exist_ok=True)

    ##=============
    tpk_spec_wiki, tst_mad_spec_wiki, ted_mad_spec_wiki = [], [], []
    tst_thrd_spec_wiki, ted_thrd_spec_wiki = [], []
    tst_manu_spec_wiki, ted_manu_spec_wiki = [], []
    depec_file = []

    ##=============
    for ww, file_wiki in enumerate(files_wiki):  # len(files_wiki)

        filename1 = os.path.basename(file_wiki)
        depec_file.append(filename1)
        print("Spec data: ", filename1)

        try:
            if filename1.split('.')[-1] == 'dat':
                filename = filename1.split('.dat')[0]
                data1 = rd_datfile(file_wiki)
                spec = np.array(data1['data'])
                fghz = np.array(data1['fghz'])
                time1 = data1['time']
            if filename1.split('.')[-1] == 'fits':
                filename = filename1.split('.fits')[0]
                eospecfits = fits.open(file_wiki)
                spec = eospecfits[0].data  # [freq, time]
                fghz = np.array(eospecfits[1].data['FGHZ'])  # in GHz
                time1 = np.array(eospecfits[2].data['TIME'])  # in jd format

        except Exception as e:
            temp = datetime.strptime(str(flare_id[ww]), "%Y%m%d%H%M%S")
            temp_st = (temp - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S")
            temp_ed = (temp + timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S")

            tpk_spec_wiki.append(temp.strftime("%Y-%m-%d %H:%M:%S"))
            tst_mad_spec_wiki.append(temp_st)
            ted_mad_spec_wiki.append(temp_ed)
            tst_thrd_spec_wiki.append(temp_st)
            ted_thrd_spec_wiki.append(temp_ed)
            tst_manu_spec_wiki.append(temp_st)
            ted_manu_spec_wiki.append(temp_ed)
            print('no data for:', file_wiki)
            continue

        time_spec = Time(time1, format='jd')
        time_str = time_spec.isot

        ##=============try MAD method
        tpk_tot_ind = []

        tst_tot_ind = []
        ted_tot_ind = []
        mad_threshd = []

        tst_tot_ind_thrd = []
        ted_tot_ind_thrd = []

        spec = np.nan_to_num(spec, nan=0.0)
        spec[spec < 0] = 0.01
        spec_median = np.median(spec, axis=1, keepdims=True)
        spec_abs_deviations = np.abs(spec - spec_median)
        spec_mad = np.median(spec_abs_deviations, axis=1, keepdims=True)
        mad_threshd = 3.0 * spec_mad

        outliers = spec_abs_deviations > mad_threshd

        for ff in range(len(fghz)):
            good_channel = False
            flux_array = spec[ff, :]

            ##=============try MAD method
            outlier_ind = np.where(outliers[ff])[0]
            if len(outlier_ind) > 0:
                tst_tot_ind.append(outlier_ind[0])
                ted_tot_ind.append(outlier_ind[-1])

            ##=============try to set threshold
            y = moving_average(flux_array, window_size) + 0.001  ##flux_array
            peaks, _ = find_peaks(y, height=1.)
            noise_thrd_st = np.mean(y[0:5])
            noise_thrd_ed = np.mean(y[-5:])

            if noise_thrd_st == 0:
                noise_thrd_st = 0.005 * np.max(y)
            if noise_thrd_ed == 0:
                noise_thrd_ed = 0.01 * np.max(y)
            if np.max(y) / noise_thrd_ed > 100:
                noise_thrd_ed = 0.02 * np.max(y)
                # noise_thrd_st = np.max([np.mean(y[0:10]),0.01*np.max(y)])
            # noise_thrd_ed = np.max([np.mean(y[-10:])*np.median(y[peaks]),0.01*np.max(y)])
            # if ff == 5:
            #     print(np.max(y), np.median(y[peaks]), np.mean(y[0:10]), np.mean(y[-10:]))

            for ind in range(len(y) - 5):
                if y[ind] < y[ind + 1] < y[ind + 2] < y[ind + 3] < y[ind + 4] < y[ind + 5]:
                    if y[ind + 5] >= 2 * y[ind]:
                        if all(y[i] > noise_thrd_st for i in range(ind, ind + 6)):
                            tst_tot_ind_thrd.append(ind)
                            break
            ind_tmp = np.argmax(flux_array) - 30
            for ind in range(len(y) - 5):
                if y[ind] > y[ind + 1] > y[ind + 2] > y[ind + 3] > y[ind + 4] > y[ind + 5]:
                    if y[ind + 3] <= 2 * y[ind]:
                        if all(abs(y[i]) > noise_thrd_ed for i in range(ind, ind + 6)):
                            ind_tmp = ind + 5
            ted_tot_ind_thrd.append(ind_tmp)

            ##=============tpeak
            tpk_tot_ind.append(np.argmax(flux_array))

        time_st_mad = time_spec[int(np.round(np.median(np.array(tst_tot_ind))))]
        time_ed_mad = time_spec[int(np.round(np.median(np.array(ted_tot_ind))))]

        time_st_thrd = time_spec[int(np.round(np.median(np.array(tst_tot_ind_thrd))))]
        time_ed_thrd = time_spec[int(np.round(np.median(np.array(ted_tot_ind_thrd))))]

        time_st = time_st_thrd
        time_ed = time_ed_thrd

        time_pk = time_spec[int(np.median(np.array(tpk_tot_ind)))]

        time_pk_obj = Time(time_pk, format='jd')

        tpk_spec_wiki.append(time_pk_obj.strftime('%Y-%m-%d %H:%M:%S'))

        tst_mad_spec_wiki.append(Time(time_st_mad, format='jd').strftime('%Y-%m-%d %H:%M:%S'))
        ted_mad_spec_wiki.append(Time(time_ed_mad, format='jd').strftime('%Y-%m-%d %H:%M:%S'))

        tst_thrd_spec_wiki.append(Time(time_st_thrd, format='jd').strftime('%Y-%m-%d %H:%M:%S'))
        ted_thrd_spec_wiki.append(Time(time_ed_thrd, format='jd').strftime('%Y-%m-%d %H:%M:%S'))

        # #####==================================================plot the dynamic spectrum and flux curves
        # tim_plt = time_spec.plot_date
        # freq_plt = fghz
        # spec_plt = spec  # np.nan_to_num(spec, nan=0.0)
        # spec_plt[spec_plt < 0] = 0.01

        # # drange = [1,np.max(spec_plt)]#np.max(spec_plt)
        # # drange = [0.01, 10]  # np.max(spec_plt)
        # drange = [0.01, np.percentile(spec_plt, 97.5)]

        # if do_manu == 1:
        #     #####==================================================
        #     # Initialize global variables
        #     click_count = 0
        #     x1, y1, x2, y2 = 0, 0, 0, 0

        #     # Create a figure and connect the mouse click event handler
        #     fig, ax2 = plt.subplots()
        #     cid = fig.canvas.mpl_connect('button_press_event', onclick)

        #     ##=============plot the flux curves and manully determine the start/end time
        #     freq_pl_tot = [2., 4., 6., 8., 10., 12.]  #
        #     # freq_pl_tot = [3.,5.,7.,9.,11,14.]#

        #     cmap_flux = matplotlib.colormaps.get_cmap("jet")

        #     for ff, freq_pl_temp in enumerate(freq_pl_tot):
        #         freq_pl_ind = np.argmin(np.abs(freq_plt - freq_pl_temp))
        #         ax2.plot(tim_plt, spec_plt[freq_pl_ind, :], label="{:.2f}".format(freq_plt[freq_pl_ind]) + ' GHz',
        #                  c=cmap_flux(ff / (len(freq_pl_tot) - 1)), linewidth=0.8)
        #         plt.hlines(mad_threshd[freq_pl_ind], tim_plt[0], tim_plt[-1], linestyle='--',
        #                    color=cmap_flux(ff / (len(freq_pl_tot) - 1)), linewidth=0.5)

        #     ax2.set_ylabel('Flux [sfu]', fontsize=fontsize_pl + 2)
        #     ax2.set_xlim(tim_plt[0], tim_plt[-1])
        #     # ax2.set_ylim(drange[0], drange[1])
        #     ax2.set_title(f'{filename}')#EOVSA Flare ID: 

        #     locator = AutoDateLocator(minticks=2)
        #     ax2.xaxis.set_major_locator(locator)
        #     formatter = AutoDateFormatter(locator)
        #     formatter.scaled[1 / 24] = '%D %H'
        #     formatter.scaled[1 / (24 * 60)] = '%H:%M'
        #     ax2.xaxis.set_major_formatter(formatter)
        #     ax2.set_autoscale_on(False)

        #     plt.xticks(fontsize=fontsize_pl)
        #     plt.yticks(fontsize=fontsize_pl)

        #     plt.fill_between([time_st.plot_date, time_ed.plot_date], -1e3, 1e4, color='gray', alpha=0.3,
        #                      label='Flare Duration')

        #     plt.legend(prop={'size': 11})

        #     # Show the figure
        #     plt.show()
        #     ##=============
        #     # After closing the figure, check if two clicks were made
        #     if click_count == 2:
        #         print("Clicked positions:", (x1, y1, x2, y2))
        #     else:
        #         x1 = time_st.plot_date
        #         x2 = time_ed.plot_date
        #         print("You need to click twice to select positions.")
        #     try:
        #         tst_manu_spec_wiki.append(Time(x1, format='plot_date').strftime('%Y-%m-%d %H:%M:%S'))
        #         ted_manu_spec_wiki.append(Time(x2, format='plot_date').strftime('%Y-%m-%d %H:%M:%S'))
        #     except ValueError:
        #         print(f"The input x1/x2 does not match the plot_date format.")
        #         tst_manu_spec_wiki.append(Time(time_st, format='jd').strftime('%Y-%m-%d %H:%M:%S'))
        #         ted_manu_spec_wiki.append(Time(time_ed, format='jd').strftime('%Y-%m-%d %H:%M:%S'))
        # else:
        #     tst_manu_spec_wiki.append('')
        #     ted_manu_spec_wiki.append('')

        # #####==================================================plot 1-1 spectrum
        # fig = plt.figure(figsize=(10, 8))
        # ax1 = fig.add_axes([0.1, 0.55, 0.75, 0.36])

        # ph1 = ax1.pcolormesh(tim_plt, freq_plt, spec_plt, norm=mcolors.LogNorm(vmin=drange[0], vmax=drange[1]),
        #                      cmap='viridis')  #

        # ax1.set_title(f'{filename}', fontsize=fontsize_pl + 2)
        # ax1.set_ylabel('Frequency [GHz]', fontsize=fontsize_pl + 2)

        # ax1.set_xlim(tim_plt[0], tim_plt[-1])
        # # ax1.set_ylim(freq_plt[fidx[0]], freq_plt[fidx[-1]])

        # locator = AutoDateLocator(minticks=2)
        # ax1.xaxis.set_major_locator(locator)
        # formatter = AutoDateFormatter(locator)
        # formatter.scaled[1 / 24] = '%D %H'
        # formatter.scaled[1 / (24 * 60)] = '%H:%M'
        # ax1.xaxis.set_major_formatter(formatter)
        # ax1.set_autoscale_on(False)

        # plt.xticks(fontsize=fontsize_pl)
        # plt.yticks(fontsize=fontsize_pl)

        # cax = fig.add_axes([0.86, 0.55, 0.012, 0.36])
        # cbar = plt.colorbar(ph1, ax=ax1, cax=cax)  # shrink=0.8, pad=0.05
        # cbar.set_label('Flux  (sfu)', fontsize=fontsize_pl)

        # #####==================================================plot 1-2 flux curve
        # ax2 = fig.add_axes([0.1, 0.1, 0.75, 0.36])

        # freq_pl_tot = [2., 4., 6., 8., 10., 12.]  #
        # # freq_pl_tot=[3.,5.,7.,9.,11,14.]#

        # cmap_flux = matplotlib.colormaps.get_cmap("jet")

        # for ff, freq_pl_temp in enumerate(freq_pl_tot):
        #     freq_pl_ind = np.argmin(np.abs(freq_plt - freq_pl_temp))
        #     ax2.plot(tim_plt, spec_plt[freq_pl_ind, :], label="{:.2f}".format(freq_plt[freq_pl_ind]) + ' GHz',
        #              c=cmap_flux(ff / (len(freq_pl_tot) - 1)), linewidth=0.8)
        #     plt.hlines(mad_threshd[freq_pl_ind], tim_plt[0], tim_plt[-1], linestyle='--',
        #                color=cmap_flux(ff / (len(freq_pl_tot) - 1)), linewidth=0.5)

        # ax2.set_ylabel('Flux [sfu]', fontsize=fontsize_pl + 2)
        # ax2.set_xlim(tim_plt[0], tim_plt[-1])
        # # ax2.set_ylim(drange[0], drange[1])

        # locator = AutoDateLocator(minticks=2)
        # ax2.xaxis.set_major_locator(locator)
        # formatter = AutoDateFormatter(locator)
        # formatter.scaled[1 / 24] = '%D %H'
        # formatter.scaled[1 / (24 * 60)] = '%H:%M'
        # ax2.xaxis.set_major_formatter(formatter)
        # ax2.set_autoscale_on(False)

        # plt.xticks(fontsize=fontsize_pl)
        # plt.yticks(fontsize=fontsize_pl)

        # plt.fill_between([time_st.plot_date, time_ed.plot_date], -1e3, 1e4, color='gray', alpha=0.3,
        #                  label='Flare Duration')
        # if do_manu == 1:
        #     plt.fill_between([x1, x2], -1e3, 1e4, color='red', alpha=0.1, label='st/ed manu')

        # plt.legend(prop={'size': 9})

        # ## add logos
        # try:
        #     add_logos_horizontally(fig, FIG_DPI, logos, gap=4, right_offset=200, top_offset=30)
        # except:
        #     print('Failed to add logos. Proceed')
        # #####==================================================
        # figname = os.path.join(spec_data_dir, time_pk_obj.strftime('%Y'), f"{filename}.png")
        # fig.savefig(figname, dpi=FIG_DPI, transparent=False)
        # plt.close()
        # print(f'Write spectrogram to {figname}')

    data_csv = {
        "ID": np.arange(len(flare_id)) + 1,
        "Flare_ID": flare_id,
        'EO_tpeak': tpk_spec_wiki,
        'EO_tstart_thrd': tst_thrd_spec_wiki,
        'EO_tend_thrd': ted_thrd_spec_wiki,
        # 'EO_tstart_manu': tst_manu_spec_wiki,
        # 'EO_tend_manu': ted_manu_spec_wiki,
        'EO_tstart_mad': tst_mad_spec_wiki,
        'EO_tend_mad': ted_mad_spec_wiki,
        'depec_file': df['depec_file'],
        'depec_img': df['depec_img']
    }

    df = pd.DataFrame(data_csv)
    csv_file_trange = os.path.join(work_dir, 'get_spec_tst_ted_from_wiki_given_date.csv')
    df.to_csv(csv_file_trange, index=False)

    print("Step 4: Times data saved to get_spec_tst_ted_from_wiki_given_date.csv")

    print("##=============step 5: combine two files")

    # Load data
    df_time = pd.read_csv(updated_csv)
    df_tst_ted = pd.read_csv(csv_file_trange)

    # Convert dates and times from df_time to datetime for efficient handling
    df_time['DateTime'] = pd.to_datetime(df_time['Date'] + ' ' + df_time['Time (UT)'], format='%Y/%m/%d %H:%M:%S')

    # Similarly, convert EO_tpeak in df_tst_ted to datetime
    df_tst_ted['EO_tpeak_dt'] = pd.to_datetime(df_tst_ted['EO_tpeak'], format='%Y/%m/%d %H:%M:%S')

    # Assuming we need to find the closest EO_tpeak_dt for each DateTime in df_time
    # Initialize columns in df_time for the closest matches
    df_time['EO_tstart'] = ""
    df_time['EO_tpeak'] = ""
    df_time['EO_tend'] = ""
    df_time['depec_file'] = ""
    df_time['depec_img'] = ""

    # Convert EO_tpeak_dt to the same MJD format for comparison
    df_tst_ted['EO_tpeak_mjd'] = [Time(row, format='datetime').mjd for row in df_tst_ted['EO_tpeak_dt']]

    # For each row in df_time, find the closest date in df_tst_ted
    for index, row in df_time.iterrows():
        # Compute the absolute difference in times between the current row in df_time and all rows in df_tst_ted
        time_diff = np.abs(df_tst_ted['EO_tpeak_dt'] - row['DateTime'])

        # Find the index of the minimum time difference
        closest_index = time_diff.idxmin()

        # Assign the matched values from df_tst_ted to df_time
        df_time.at[index, 'EO_tstart'] = df_tst_ted.at[closest_index, 'EO_tstart_thrd']
        df_time.at[index, 'EO_tpeak'] = df_tst_ted.at[closest_index, 'EO_tpeak']
        df_time.at[index, 'EO_tend'] = df_tst_ted.at[closest_index, 'EO_tend_thrd']
        df_time.at[index, 'depec_file'] = df_tst_ted.at[closest_index, 'depec_file']
        df_time.at[index, 'depec_img'] = df_tst_ted.at[closest_index, 'depec_img']

    # Drop the temporary columns
    df_time.drop(['GOES_flare_class', 'DateTime'], axis=1, inplace=True)

    # Save the combined data to CSV
    csv_file_comb = os.path.join(work_dir, 'get_info_from_wiki_given_date_sub.csv')
    df_time.to_csv(csv_file_comb, index=False)

    print(f"step 5: Times data saved to {csv_file_comb}")

    final_csv_file = os.path.join(work_dir, 'EOVSA_flare_list_from_wiki_sub.csv')
    os.rename(csv_file_comb, final_csv_file)
    print(f"Renamed to {final_csv_file}")

    ##=============step 6: write to MySQL 'EOVSA_flare_list_wiki_db'=============
    print("##=============step 6: write flare info to MySQL 'EOVSA_flare_list_wiki_db'")

    ##=============get flare_id_exist from mySQL

    connection = create_flare_db_connection()

    cursor = connection.cursor()
    cursor.execute("SELECT Flare_ID FROM EOVSA_flare_list_wiki_tb")
    flare_id_exist = cursor.fetchall()

    cursor.close()
    connection.close()

    ##=============

    # Connect to the database and get a cursor to access it:
    cnxn = create_flare_db_connection()

    cursor = cnxn.cursor()
    table = 'EOVSA_flare_list_wiki_tb'

    # Write to the database (add records)
    # Assume a database that mirrors the .csv file (first two lines below):
    #    Flare_ID,Date,Time,flare_class,EO_tstart,EO_tpeak,EO_tend,EO_xcen,EO_ycen
    #    20190415193100,2019-04-15,19:31:00,B3.3,2019-04-15 19:30:04,2019-04-15 19:32:21,2019-04-15 19:33:10,519.1,152.3
    # The Flare_ID is automatic (just incremented from 1), so is not explicitly written.  Also, separating Date and Time doesn't make sense, so combine into a single Date:

    columns = ['Flare_ID', 'Flare_class', 'EO_tstart', 'EO_tpeak', 'EO_tend', 'depec_file', 'depec_img']

    values = []

    #####==================================================

    df = pd.read_csv(final_csv_file)
    flare_id = df['Flare_ID']
    dates = df['Date']
    times = df['Time (UT)']
    EO_tstart = df['EO_tstart']
    EO_tpeak = df['EO_tpeak']
    EO_tend = df['EO_tend']
    GOES_class = df['flare_class']
    depec_file = df['depec_file']
    depec_img = df['depec_img']

    ##=============
    for i in range(len(flare_id)):
        if not any(int(flare_id[i]) == existing_flare_id[0] for existing_flare_id in flare_id_exist):
            date = Time(dates[i] + ' ' + times[i]).jd
            # newlist = [int(flare_id[i]), GOES_class[i], Time(EO_tstart[i]).jd, Time(EO_tpeak[i]).jd, Time(EO_tend[i]).jd, EO_xcen[i], EO_ycen[i], depec_file[i]]
            newlist = [int(flare_id[i]), GOES_class[i], Time(EO_tstart[i]).jd, Time(EO_tpeak[i]).jd,
                       Time(EO_tend[i]).jd,
                       str(depec_file[i]), str(depec_img[i])]

            values.append(newlist)
            print("EOVSA_flare_list_wiki_tb Update for ", int(flare_id[i]))

    values = [[None if pd.isna(val) else val for val in sublist] for sublist in values]

    put_query = 'insert ignore into ' + table + ' (' + ','.join(columns) + ') values (' + ('%s,' * len(columns))[
                                                                                          :-1] + ')'

    cursor.executemany(put_query, values)
    cnxn.commit()  # Important!  The record will be deleted if you do not "commit" after a transaction

    ##=============step 7: write to MySQL 'EOVSA_lightcurve_QL_db'=============
    print("##=============step 7: write flare light curves to MySQL 'EOVSA_lightcurve_QL_db'")

    ##=============get flare_id_exist from mySQL

    connection = create_flare_lc_db_connection()

    cursor = connection.cursor()
    cursor.execute("SELECT DISTINCT Flare_ID FROM freq_QL")
    flare_id_exist = cursor.fetchall()

    cursor.close()
    connection.close()

    # Connect to the database and get a cursor to access it:
    cnxn = create_flare_lc_db_connection()

    #####==================================================data preparation

    df = pd.read_csv(final_csv_file)
    flare_id_tot = df['Flare_ID']
    depec_file_tot = df['depec_file']
    EO_tpeak_tot = df['EO_tpeak']

    for i, date_str in enumerate(EO_tpeak_tot):
        datetp = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')

        if not any(int(flare_id_tot[i]) == existing_flare_id[0] for existing_flare_id in flare_id_exist):
            # if given_date[0] <= datetp <= given_date[1]:  # Check if the date is within the given range
            print("Flare_ID", int(flare_id_tot[i]))
            Flare_ID = int(flare_id_tot[i])
            depec_file = str(depec_file_tot[i])

            dsfile_path = '/common/webplots/events/' + str(Flare_ID)[0:4] + '/' + depec_file
            try:
                if dsfile_path.split('.')[-1] == 'dat':
                    data1 = rd_datfile(dsfile_path)
                    time = data1['time']  ##in jd
                    freq = np.array(data1['fghz'])
                    spec = np.array(data1['data'])

                if dsfile_path.split('.')[-1] == 'fits':
                    eospecfits = fits.open(dsfile_path)
                    time = np.array(eospecfits[2].data['TIME'])
                    freq = np.array(eospecfits[1].data['FGHZ'])
                    spec = eospecfits[0].data

                time_new, freq_new, spec_new = spec_rebin(time, freq, spec, t_step=1, f_step=1, do_mean=False)

                freq_plt = [3, 7, 11, 15]
                freq_QL = np.zeros(len(freq_plt))
                spec_QL = np.zeros((len(freq_plt), len(time_new)))

                for ff, fghz in enumerate(freq_plt):
                    ind = np.argmin(np.abs(freq_new - fghz))
                    # print(ind, fghz, freq_new[ind])
                    freq_QL[ff] = freq_new[ind]
                    spec_QL[ff, :] = spec_new[ind, :]

                time_QL = time_new
            except Exception as e:
                print(f"Errors of reading data {dsfile_path} - flux set to be 0")
                date_list = []
                for mm in range(20):
                    date_list.append(datetp + timedelta(seconds=12 * mm))
                time_QL = [date_obj.toordinal() + 1721425.5 for date_obj in date_list]
                freq_QL = [3, 7, 11, 15]
                spec_QL = np.zeros((len(freq_QL), len(time_QL))) + 1e-3

            cursor = cnxn.cursor()
            select_query = "SELECT * FROM Flare_IDs WHERE Flare_ID = %s"
            cursor.execute(select_query, (Flare_ID,))
            existing_records = cursor.fetchall()
            if existing_records:
                print(f"{Flare_ID} exist then jump to next ID ")
                continue
            cursor.close()

            #####==================================================
            cursor = cnxn.cursor()
            insert_query = "INSERT INTO Flare_IDs (Flare_ID, `Index`) VALUES (%s, %s)"
            cursor.execute(insert_query, (Flare_ID, i + 1))
            cnxn.commit()
            cursor.close()

            #####=============
            cursor = cnxn.cursor()
            for index, value in enumerate(time_QL):
                jd_time = value  # Time(value).jd
                cursor.execute("INSERT INTO time_QL VALUES (%s, %s, %s)", (Flare_ID, index, jd_time))
            cnxn.commit()
            cursor.close()

            #####=============
            cursor = cnxn.cursor()
            for index, value in enumerate(freq_QL):
                cursor.execute("INSERT INTO freq_QL VALUES (%s, %s, %s)", (Flare_ID, index, round(value, 3)))
            cnxn.commit()
            cursor.close()

            #####=============
            cursor = cnxn.cursor()
            for ff, row in enumerate(spec_QL):
                # print(ff, len(row))
                for tt, value in enumerate(row):
                    value = round(value, 3) if not np.isnan(value) else None  # Replace nan with None
                    cursor.execute("INSERT INTO flux_QL VALUES (%s, %s, %s, %s)", (Flare_ID, ff, tt, value))
            cnxn.commit()
            cursor.close()

    cnxn.close()

    print("Success!")


if __name__ == "__main__":
    main()
