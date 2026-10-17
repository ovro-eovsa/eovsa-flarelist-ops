##ipython flarelist2sql.py
##=========================
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

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.dates import AutoDateFormatter, AutoDateLocator, num2date
import matplotlib.cm
# import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime, timedelta
from astropy.time import Time
from astropy.io import fits
import mysql.connector

import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings('ignore', category=FutureWarning)

current_year = datetime.now().year

##=========================
# Global settings and initializations
# EO_WIKI_URL = "http://www.ovsa.njit.edu/wiki/index.php/Recent_Flare_List_(2021-)"
wiki_base_url = "https://www.ovsa.njit.edu/wiki/index.php/"
EO_WIKI_URLs = [f"{wiki_base_url}{year}" for year in range(2017, current_year + 1)]

##=========================dat data
## List of URLs to download files from. this is not needed if running the code on ovsa and pipeline.
datfile_base_url = "http://ovsa.njit.edu/events/"
datfile_urls = [f"{datfile_base_url}{year}/" for year in range(2017, current_year + 1)]

## plotting config
fontsize_pl = 14.
window_size = 10


##=========================connect to database
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


def check_url_exists(url, timeout=5):
    try:
        response = requests.head(url, allow_redirects=True, timeout=timeout)
        if response.status_code == 200:
            return True
        if url.endswith('/'):
            response = requests.get(url, allow_redirects=True, timeout=timeout)
            return response.status_code == 200
        return False
    except requests.RequestException:
        return False


##=========================add EOVSA log to the spec plotting
# Path to the folder containing the static images
static_img_folder = '/var/www/html/eovsadata/assets/'
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


##=========================
def fetch_flare_data_from_wiki(eo_wiki_urls, given_date_strp, outcsvfile, debug_wiki=False):
    """
    Fetches flare data from the given wiki URLs and saves it to a CSV file in the specified directory.

    Parameters:
    - eo_wiki_urls: List of URLs of the EO wiki pages to fetch data from.
    - given_date_strp: A tuple of datetime objects specifying the start and end dates to filter the data.
    - outcsvfile: Path to the resulting CSV file.

    Returns:
    - None
    """
    # Initializing lists to collect data
    date_data, time_ut_data, flare_class_data, flare_ID_data = [], [], [], []
    depec_imgfile_XP, depec_datafile_XP = [], []
    depec_imgfile_TP, depec_datafile_TP = [], []
    debug_rows = []

    def _normalize_datetime(date_str, time_str):
        date_clean = date_str.strip().replace('/', '-')
        time_clean = time_str.strip()
        time_clean = time_clean.replace('UTC', '').replace('UT', '').strip()
        if re.match(r'^\d{2}:\d{2}$', time_clean):
            time_clean = f"{time_clean}:00"
        for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M'):
            try:
                return datetime.strptime(f"{date_clean} {time_clean}", fmt)
            except ValueError:
                pass
        return None

    def _clean_file_name(name):
        name = requests.utils.unquote(os.path.basename(str(name).split('?')[0]))
        return re.sub(r'^\d+px-', '', name)

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
                        datetime_strp = _normalize_datetime(date, time_ut)
                        if datetime_strp is None:
                            if debug_wiki:
                                debug_rows.append({
                                    'wiki_url': eo_wiki_url,
                                    'date_raw': date,
                                    'time_raw': time_ut,
                                    'flare_class_raw': flare_class,
                                    'reason': 'datetime_parse_failed',
                                    'row_text': row.get_text(' ', strip=True)
                                })
                            continue

                        if given_date_strp[0] <= datetime_strp <= given_date_strp[1]:
                            # Get Date, Time (UT), and Flare Class
                            date_norm = datetime_strp.strftime('%Y-%m-%d')
                            time_norm = datetime_strp.strftime('%H:%M:%S')
                            date_data.append(date_norm)
                            time_ut_data.append(time_norm)
                            flare_class_data.append(flare_class)

                            # Get Flare_ID
                            eotime = f"{date_norm}T{time_norm}"
                            flare_ID_data.append(eotime.replace('-', '').replace('T', '').replace(':', ''))
                            print("Fetched: ", eotime)

                            # Initialize placeholders for spectrogram files
                            imgfile_XP, datafile_XP, imgfile_TP, datafile_TP = '', '', '', ''
                            
                            link_files = []
                            image_files = []

                            for link in row.find_all('a', href=True):
                                href = str(link.get('href', ''))
                                if not href:
                                    continue
                                href_file = _clean_file_name(href)
                                if href_file and '.' in href_file:
                                    link_files.append(href_file)

                                # MediaWiki usually links thumbnails through /wiki/File:<name>.
                                if 'File:' in href:
                                    wiki_file = _clean_file_name(href.split('File:')[-1])
                                    if wiki_file.lower().endswith('.png'):
                                        image_files.append(wiki_file)

                            for img_tag in row.find_all('img'):
                                src_attribute = img_tag.get('src')
                                if src_attribute:
                                    image_files.append(_clean_file_name(src_attribute))

                            for file_name in link_files:
                                file_low = file_name.lower()
                                if file_low.endswith(('.dat', '.fits')):
                                    if ('eovsa.spec_tp' in file_low) and not datafile_TP:
                                        datafile_TP = file_name
                                    if (('eovsa.spec_xp.' in file_low) or ('eovsa.spec.' in file_low)) and not datafile_XP:
                                        datafile_XP = file_name

                            for file_name in image_files:
                                file_low = file_name.lower()
                                if file_low.endswith('.png'):
                                    if ('.spec_tp.' in file_low) and not imgfile_TP:
                                        imgfile_TP = file_name
                                    if (('.spec_xp.' in file_low) or ('.spec.' in file_low)) and not imgfile_XP:
                                        imgfile_XP = file_name

                            print(date, time_ut, flare_class, imgfile_XP, datafile_XP)
                            if debug_wiki and (not datafile_XP or not imgfile_XP):
                                debug_rows.append({
                                    'wiki_url': eo_wiki_url,
                                    'date': date_norm,
                                    'time_ut': time_norm,
                                    'flare_class': flare_class,
                                    'datafile_tp': datafile_TP,
                                    'imgfile_tp': imgfile_TP,
                                    'datafile_xp': datafile_XP,
                                    'imgfile_xp': imgfile_XP,
                                    'link_files': ';'.join(sorted(set(link_files))),
                                    'image_files': ';'.join(sorted(set(image_files))),
                                    'reason': 'missing_xp_or_image',
                                    'row_text': row.get_text(' ', strip=True)
                                })
                            # Append extracted file names to respective lists
                            depec_imgfile_TP.append(imgfile_TP)
                            depec_datafile_TP.append(datafile_TP)
                            depec_imgfile_XP.append(imgfile_XP)
                            depec_datafile_XP.append(datafile_XP)

        else:
            if response.status_code == 404:
                print(f"Wiki page not found (404), skipping: {eo_wiki_url}")
            else:
                print(f"Failed to retrieve the webpage {eo_wiki_url}. Status code: {response.status_code}")

    # Reformatting and saving collected data into a DataFrame
    data = {
        "ID": np.arange(len(date_data)) + 1,
        "Flare_ID": flare_ID_data,
        "Date": date_data,
        "Time (UT)": time_ut_data,
        "Flare_Class": flare_class_data,
        "depec_imgfile_TP": depec_imgfile_TP,
        "depec_datafile_TP": depec_datafile_TP,
        "depec_imgfile_XP": depec_imgfile_XP,
        "depec_datafile_XP": depec_datafile_XP
    }
    df = pd.DataFrame(data)
    df.to_csv(outcsvfile, index=False)
    print(f"Data saved to {outcsvfile}")
    if debug_wiki and debug_rows:
        debug_file = os.path.splitext(outcsvfile)[0] + '_wiki_debug.csv'
        pd.DataFrame(debug_rows).to_csv(debug_file, index=False)
        print(f"Debug wiki parse data saved to {debug_file}")
    return

##=========================
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



##=========================
def add_GOES_to_flarelist(input_csv, output_csv):
    """
    Add GOES_class, GOES_tstart, GOES_tpeak, GOES_tend, GOES_hgc_x, GOES_hgc_y
    to the input flarelist csv file
    """
    df = pd.read_csv(input_csv)

    GOES_flare_class = []
    GOES_tstart = []
    GOES_tpeak = []
    GOES_tend = []
    GOES_hgc_x = []
    GOES_hgc_y = []
    for index, row in df.iterrows():
        eotime_flare_wiki = row["Date"] + 'T' + row["Time (UT)"]
        GOES_class_tp, GOES_tstart_tp, GOES_tpeak_tp, GOES_tend_tp, GOES_hgc_x_tp, GOES_hgc_y_tp = get_flare_info_from_GOES(tpeak_str=eotime_flare_wiki)
        GOES_flare_class.append(GOES_class_tp)
        GOES_tstart.append(GOES_tstart_tp.split('.')[0])
        GOES_tpeak.append(GOES_tpeak_tp.split('.')[0])
        GOES_tend.append(GOES_tend_tp.split('.')[0])
        GOES_hgc_x.append(GOES_hgc_x_tp)
        GOES_hgc_y.append(GOES_hgc_y_tp)
    # Reformatting and saving collected data into a DataFrame
    data = {
        "ID": df['ID'],
        "Flare_ID": df['Flare_ID'],
        "Date": df['Date'],
        "Time (UT)": df['Time (UT)'],
        "Flare_Class": df['Flare_Class'],
        "GOES_flare_class": GOES_flare_class,
        "GOES_tstart": GOES_tstart,
        "GOES_tpeak": GOES_tpeak,
        "GOES_tend": GOES_tend,
        # "GOES_hgc_x": GOES_hgc_x,
        # "GOES_hgc_y": GOES_hgc_y,
        "depec_imgfile_TP": df['depec_imgfile_TP'],
        "depec_datafile_TP": df['depec_datafile_TP'],
        "depec_imgfile_XP": df['depec_imgfile_XP'],
        "depec_datafile_XP": df['depec_datafile_XP']
    }
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"<add_GOES_to_flarelist> data saved to {output_csv}")
    return


##=========================
def download_datfiles_from_url(url, download_directory, timerange_strp):
    """
    Downloads .dat and .fits EOVSA spec files from a given URL within a specified date range.
    """
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
                # Match different file naming conventions
                if file_name.split('.')[-1] == 'dat':
                    match = re.match(r'EOVSA_(\d{4})(\d{2})(\d{2})', file_name)
                    if not match:
                        match = re.match(r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', file_name)
                elif file_name.split('.')[-1] == 'fits':
                    match = re.match(r'eovsa.spec.flare_id_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', file_name)
                    if not match:
                        # Match eovsa.spec_xp.flare_id_ with either 12 or 14 digits after
                        match = re.match(r'eovsa\.spec_xp\.flare_id_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})?', file_name)

                # Check if the file name matches the given date range pattern
                if match:
                    file_date = match.group(1) + '-' + match.group(2) + '-' + match.group(3)
                    # Check if the file date is within the given date range
                    if timerange_strp[0].strftime('%Y-%m-%d') <= file_date <= (
                            timerange_strp[1] + timedelta(days=1)).strftime('%Y-%m-%d'):
                        # Download the file if it does not already exist
                        if os.path.exists(os.path.join(download_directory, file_name)):
                            print(f"Exist: {file_name}")
                            continue  # Skip to the next iteration
                        try:
                            # Download and save the file
                            with open(os.path.join(download_directory, file_name), 'wb') as file:
                                file.write(requests.get(href).content)
                            print(f"Downloaded: {file_name}")
                            file_name_download.append(file_name)
                        except Exception as e:
                            print(f"Failed to download: {file_name} - {str(e)}")
    else:
        print(f"Failed to retrieve the webpage {url}. Status code: {response.status_code}")
    return file_name_download


##=========================
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
        f = open(file,'rb')
        tmp = f.read(83608)  # max 10000 times and 451 frequencies
        f.close()
        nbytes = len(tmp)
        tdat = np.array(struct.unpack(str(int(nbytes/8))+'d',tmp[:nbytes]))
        nt = np.where(tdat < 2400000.)[0]
        nf = np.where(np.logical_or(tdat[nt[0]:] > 18, tdat[nt[0]:] < 1))[0]
        return nt[0], nf[0]
    nt, nf = dims(file)
    f = open(file,'rb')
    tmp = f.read(nt*8)
    times = struct.unpack(str(nt)+'d',tmp)
    tmp = f.read(nf*8)
    fghz = struct.unpack(str(nf)+'d',tmp)
    tmp = f.read()
    f.close()
    if len(tmp) != nf*nt*4:
        print('File size is incorrect for nt=',nt,'and nf=',nf)
        return {}
    data = np.array(struct.unpack(str(nt*nf)+'f',tmp)).reshape(nf,nt)
    return {'time':times, 'fghz':fghz, 'data':data}


##=========================
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


##=========================
def moving_average(data, window_size):
    # Create a convolution kernel for the moving average
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')


def despike_small_blobs(data, win=11, n_sigmas=4, max_blob_size=8):
    """Replace small, isolated high-value blobs while preserving coherent bursts."""
    from scipy.ndimage import label, median_filter

    data = np.asarray(data, dtype=float)
    med = median_filter(data, size=(win, win), mode='reflect')
    mad = 1.4826 * median_filter(np.abs(data - med), size=(win, win), mode='reflect')
    positive_mad = mad[mad > 0]
    mad[mad == 0] = np.nanmedian(positive_mad) if positive_mad.size else 1.0

    candidate_mask = (data - med) / mad > n_sigmas
    labeled, nlab = label(candidate_mask)
    if nlab == 0:
        return data

    blob_sizes = np.bincount(labeled.ravel())
    small_blob_ids = np.where(blob_sizes < max_blob_size)[0]
    small_blob_ids = small_blob_ids[small_blob_ids != 0]
    return np.where(np.isin(labeled, small_blob_ids), med, data)


def hampel_2d_vertical(data, window_size=15, n_sigmas=3):
    from scipy.ndimage import median_filter

    data = np.asarray(data, dtype=float)
    med = median_filter(data, size=(window_size, 1), mode='reflect')
    mad = 1.4826 * median_filter(np.abs(data - med), size=(window_size, 1), mode='reflect')
    mask = (mad > 0) & (np.abs(data - med) > n_sigmas * mad)
    return np.where(mask, med, data)


def hampel_2d(data, win_vert=15, win_horiz=15, n_sigmas=3):
    cleaned = hampel_2d_vertical(data, window_size=win_vert, n_sigmas=n_sigmas)
    cleaned = hampel_2d_vertical(cleaned.T, window_size=win_horiz, n_sigmas=n_sigmas).T
    return despike_small_blobs(cleaned)


##=========================
def parse_arguments():
    parser = argparse.ArgumentParser(description='Fetch EOVSA flare data from wiki and write MySQL tables for eovsadata.',
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
                        help='Time range for the analysis, formatted as "YYYY-MM-DD HH:MM:SS YYYY-MM-DD HH:MM:SS".\n'
                             'Backslashes (e.g., from cron-escaped date output) are tolerated.\n'
                             'Defaults to 7 days ago from 13:00 UT to now.')

    parser.add_argument('--do_manu', type=int, default=0,
                        help='Manually determine the start/end time of the radio burst by clicking on the Dspec.\nDefault: 0.')
    parser.add_argument('--debug_wiki', type=int, default=0,
                        help='Enable verbose wiki parsing diagnostics and write *_wiki_debug.csv.\nDefault: 0.')

    return parser.parse_args()


##=========================
def main():
    # Parse command-line arguments
    args = parse_arguments()

    timerange = args.timerange
    do_manu = args.do_manu
    debug_wiki = bool(args.debug_wiki)

    def _parse_timerange_datetime(value: str) -> datetime:
        value = value.strip()
        # Some schedulers (e.g., cron) require escaping '%' as '\\%'. Depending on how the command is invoked,
        # those backslashes can survive into the expanded `date` output, producing strings like:
        #   "\\2026-\\01-\\05 \\15:\\46:\\06"
        # Strip backslashes so the canonical format parses cleanly.
        if "\\" in value:
            value = value.replace("\\", "")

        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                pass
        raise ValueError(
            f'Invalid datetime value: {value!r}. Expected "YYYY-MM-DD HH:MM:SS" (or "YYYY-MM-DDTHH:MM:SS").'
        )

    # Convert timerange strings to datetime objects for internal use
    timerange_strp = [_parse_timerange_datetime(date_str) for date_str in timerange]

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



    ##=========================Step 1: capture the radio peak times from flare list wiki webpage=========================
    print("##=============Step 1: capture the radio peak times from flare list wiki webpage")
    init_csv = os.path.join(work_dir, "get_time_from_wiki_given_date.csv")
    fetch_flare_data_from_wiki(EO_WIKI_URLs, timerange_strp, init_csv, debug_wiki=debug_wiki)


    # ##=========================Step 2: add GOES infor to flarelist=========================
    print("##=============Step 2: add GOES infor to flarelist")
    # add_GOES_to_flarelist(init_csv, init_csv)

    updated_csv = init_csv



    ##=========================Step 3: get the spectrum data from server or EOVSA web=========================
    print("##=============Step 3: get the spectrum data from server or EOVSA web")
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
            file_name_download = download_datfiles_from_url(url, spec_data_dir, timerange_strp)
    # spec.fits will be read from spec_data_dir






    ##=========================Step 4: plot the spectrum  and determine the start and end times of radio flux profiles=========================
    print("##=============Step 4: read, plot the spectrum data")

    ##=========================
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

    ##=========================
    df = pd.read_csv(updated_csv)
    flare_id = df['Flare_ID']
    depec_file = df['depec_datafile_XP']##will use XP fits file to determine the st/ed time

    if on_server == 1:
        files_wiki = [spec_data_dir + str(flare_id[i])[0:4] + "/" + str(file_name) for i, file_name in
                      enumerate(depec_file)]
        files_wiki_tp = [spec_data_dir + str(flare_id[i])[0:4] + "/" + str(file_name)
                         if pd.notna(file_name) and str(file_name).strip() else ''
                         for i, file_name in enumerate(df['depec_datafile_TP'])]
    else:
        files_wiki = [os.path.join(spec_data_dir, f'{str(file_name)}') for file_name in depec_file]
        files_wiki_tp = [os.path.join(spec_data_dir, str(file_name))
                         if pd.notna(file_name) and str(file_name).strip() else ''
                         for file_name in df['depec_datafile_TP']]

    spec_img_dir = os.path.join(work_dir, 'spec_img/')
    os.makedirs(spec_img_dir, exist_ok=True)


    ##=========================
    tpk_spec_wiki, tst_mad_spec_wiki, ted_mad_spec_wiki = [], [], []
    tst_thrd_spec_wiki, ted_thrd_spec_wiki = [], []
    tst_manu_spec_wiki, ted_manu_spec_wiki = [], []
    Fpk_XP_3GHz, Fpk_XP_7GHz, Fpk_XP_11GHz, Fpk_XP_15GHz = [], [], [], []
    Fpk_XP_totGHz, freq_at_Fpk_XP = [], []
    Fpk_TP, pk_freq_GHz = [], []
    has_ql_movie, has_fits = [], []
    depec_file = []
    url_exists_cache = {}

    def get_flux_pk(flux_tmp):
        """Return a sustained peak instead of a potentially bad single sample."""
        from scipy.signal import medfilt

        flux_tmp = np.asarray(flux_tmp, dtype=float)
        finite_flux = flux_tmp[np.isfinite(flux_tmp)]
        if finite_flux.size == 0:
            return np.nan

        kernel_size = min(5, finite_flux.size)
        if kernel_size % 2 == 0:
            kernel_size -= 1
        if kernel_size < 3:
            return float(np.nanmedian(finite_flux))

        flux_tmp_smoothed = medfilt(flux_tmp, kernel_size=kernel_size)
        return float(np.nanmax(flux_tmp_smoothed))

    def find_horizontal_rfi(spec_data, freq_data, ratio_threshold=10.0,
                            occupancy_threshold=0.20):
        """Identify persistent horizontal narrow-band RFI channels."""
        from scipy.ndimage import binary_dilation, median_filter

        spec_data = np.asarray(spec_data, dtype=float)
        freq_data = np.asarray(freq_data, dtype=float)
        positive = np.where(
            np.isfinite(spec_data) & (spec_data > 0), spec_data, np.nan
        )
        with np.errstate(all='ignore'):
            channel_p95 = np.nanpercentile(positive, 95, axis=1)
        finite = np.isfinite(channel_p95)
        fill = float(np.nanmedian(channel_p95[finite])) if finite.any() else 0.0
        channel_p95 = np.nan_to_num(
            channel_p95, nan=fill, posinf=fill, neginf=fill
        )
        local_p95 = median_filter(channel_p95, size=31, mode='reflect')
        p95_ratio = channel_p95 / np.maximum(local_p95, 0.05)

        data = np.nan_to_num(
            spec_data, nan=0.0, posinf=0.0, neginf=0.0
        )
        local_spectrum = median_filter(data, size=(31, 1), mode='reflect')
        contrast = np.zeros_like(data)
        np.divide(
            data, local_spectrum, out=contrast, where=local_spectrum > 0.05
        )
        occupancy = np.mean(contrast > ratio_threshold, axis=1)
        mask = (
            (freq_data >= 2.0) & (freq_data <= 15.0)
            & (p95_ratio > ratio_threshold)
            & (occupancy > occupancy_threshold)
        )
        return binary_dilation(mask, iterations=1)

    def get_robust_spectrum_peaks(spec_data, freq_data):
        """Calculate sustained XP/TP peaks after excluding horizontal RFI."""
        from scipy.ndimage import median_filter

        spec_data = np.asarray(spec_data, dtype=float)
        freq_data = np.asarray(freq_data, dtype=float).reshape(-1)
        if spec_data.ndim != 2 or spec_data.shape[0] != freq_data.size:
            raise ValueError(
                f"spectrum/frequency shape mismatch: {spec_data.shape} "
                f"versus {freq_data.size}"
            )

        data = np.nan_to_num(
            spec_data, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(float, copy=False)
        data[data < 0] = 0.01
        rfi_mask = find_horizontal_rfi(data, freq_data)

        t_step, f_step = 5, 10
        nf, nt = data.shape
        pad_f = (-nf) % f_step
        pad_t = (-nt) % t_step
        padded = np.pad(
            data, ((0, pad_f), (0, pad_t)), constant_values=np.nan
        )
        with np.errstate(all='ignore'):
            rebinned = np.nanmedian(
                padded.reshape(
                    padded.shape[0] // f_step, f_step,
                    padded.shape[1] // t_step, t_step
                ),
                axis=(1, 3),
            )
        freq_padded = np.pad(
            freq_data, (0, pad_f), constant_values=np.nan
        )
        freq_rebin = np.nanmedian(
            freq_padded.reshape(-1, f_step), axis=1
        )

        # Exclude a complete frequency block if any channel in it is RFI.
        # Partially removing channels could bias a block median upward.
        mask_padded = np.pad(
            rfi_mask.astype(bool), (0, pad_f), constant_values=False
        )
        bad_frequency_blocks = mask_padded.reshape(-1, f_step).any(axis=1)
        rebinned[bad_frequency_blocks, :] = np.nan

        peak_candidates = []
        peaks_by_target = {}
        for freq_target in range(2, 16):
            ind = int(np.nanargmin(np.abs(freq_rebin - float(freq_target))))
            curve = rebinned[ind, :]
            if not np.isfinite(curve).any():
                continue
            finite_curve = curve[np.isfinite(curve)]
            fill = float(np.nanmedian(finite_curve))
            curve = np.nan_to_num(curve, nan=fill, posinf=fill, neginf=fill)
            smooth = median_filter(curve, size=5, mode='nearest')
            peak = float(np.nanmax(smooth))
            if np.isfinite(peak):
                peaks_by_target[freq_target] = peak
                peak_candidates.append((peak, float(freq_rebin[ind])))

        if not peak_candidates:
            raise ValueError(
                "no finite peak candidates between 2 and 15 GHz after RFI masking"
            )

        peak, peak_freq = max(peak_candidates, key=lambda item: item[0])
        return peak, round(peak_freq, 2), peaks_by_target

    ##=========================
    for ww, file_wiki in enumerate(files_wiki):  # len(files_wiki)
        flare_id_str = str(flare_id[ww])
        movie_url = (
            f'https://www.ovsa.njit.edu/SynopticImg/eovsamedia/eovsa-browser/'
            f'{flare_id_str[0:4]}/{flare_id_str[4:6]}/{flare_id_str[6:8]}/'
            f'eovsa.lev1_mbd_12s.flare_id_{flare_id_str}.mp4'
        )
        fits_url = (
            f'https://www.ovsa.njit.edu/fits/flares/'
            f'{flare_id_str[0:4]}/{flare_id_str[4:6]}/{flare_id_str[6:8]}/{flare_id_str}/'
        )

        if movie_url not in url_exists_cache:
            url_exists_cache[movie_url] = int(check_url_exists(movie_url))
        if fits_url not in url_exists_cache:
            url_exists_cache[fits_url] = int(check_url_exists(fits_url))
        has_ql_movie.append(url_exists_cache[movie_url])
        has_fits.append(url_exists_cache[fits_url])

        filename1 = os.path.basename(file_wiki)
        depec_file.append(filename1)
        print("To read spec data: ", filename1)

        tp_peak = None
        tp_peak_freq = None
        tp_file = files_wiki_tp[ww]
        if tp_file and os.path.isfile(tp_file):
            try:
                with fits.open(tp_file) as tp_fits:
                    tp_spec = np.asarray(tp_fits[0].data, dtype=float)
                    tp_freq = np.asarray(tp_fits[1].data['FGHZ'], dtype=float)
                tp_peak, tp_peak_freq, _ = get_robust_spectrum_peaks(tp_spec, tp_freq)
            except Exception as e:
                print('no TP peak data for:', tp_file, e)
        elif tp_file:
            print('no TP file for:', tp_file)

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
            print('no XP peak data for:', file_wiki, e)
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

            Fpk_XP_3GHz.append('NA')
            Fpk_XP_7GHz.append('NA')
            Fpk_XP_11GHz.append('NA')
            Fpk_XP_15GHz.append('NA')
            Fpk_XP_totGHz.append('NA')
            freq_at_Fpk_XP.append('NA')
            Fpk_TP.append("{:.1f}".format(tp_peak) if tp_peak is not None else '')
            pk_freq_GHz.append(tp_peak_freq if tp_peak_freq is not None else '')
            print('no data for:', file_wiki)
            continue

        time_spec = Time(time1, format='jd')
        time_str = time_spec.isot


        ##=========================try MAD method
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

            ##=========================try MAD method
            outlier_ind = np.where(outliers[ff])[0]
            if len(outlier_ind) > 0:
                tst_tot_ind.append(outlier_ind[0])
                ted_tot_ind.append(outlier_ind[-1])

            ##=========================try to set threshold
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

            ##=========================tpeak
            tpk_tot_ind.append(np.argmax(flux_array))

        def _safe_median_index(indices, fallback_index, label):
            """Return an in-range median index, using a documented fallback."""
            valid = np.asarray(indices, dtype=float)
            valid = valid[np.isfinite(valid)]
            valid = valid[(valid >= 0) & (valid < len(time_spec))]
            if valid.size:
                index = int(np.round(np.median(valid)))
            else:
                index = int(fallback_index)
                print(
                    f'WARNING: no valid {label} indices for flare {flare_id_str}; '
                    f'using fallback index {index}'
                )
            return int(np.clip(index, 0, len(time_spec) - 1))

        # Some weak or noisy events have no channel satisfying the strict
        # monotonic threshold test. Keep processing them using a two-minute
        # sample window around the median per-channel peak instead of taking
        # the median of an empty list (which produces NaN and crashes).
        peak_index = _safe_median_index(
            tpk_tot_ind, len(time_spec) // 2, 'peak'
        )
        fallback_start = max(0, peak_index - 120)
        fallback_end = min(len(time_spec) - 1, peak_index + 120)

        mad_start_index = _safe_median_index(
            tst_tot_ind, fallback_start, 'MAD start'
        )
        mad_end_index = _safe_median_index(
            ted_tot_ind, fallback_end, 'MAD end'
        )
        threshold_start_index = _safe_median_index(
            tst_tot_ind_thrd, fallback_start, 'threshold start'
        )
        threshold_end_index = _safe_median_index(
            ted_tot_ind_thrd, fallback_end, 'threshold end'
        )

        time_st_mad = time_spec[mad_start_index]
        time_ed_mad = time_spec[mad_end_index]
        time_st_thrd = time_spec[threshold_start_index]
        time_ed_thrd = time_spec[threshold_end_index]

        time_st = time_st_thrd
        time_ed = time_ed_thrd
        time_pk = time_spec[peak_index]

        time_pk_obj = Time(time_pk, format='jd')

        tpk_spec_wiki.append(time_pk_obj.strftime('%Y-%m-%d %H:%M:%S'))

        tst_mad_spec_wiki.append(Time(time_st_mad, format='jd').strftime('%Y-%m-%d %H:%M:%S'))
        ted_mad_spec_wiki.append(Time(time_ed_mad, format='jd').strftime('%Y-%m-%d %H:%M:%S'))

        tst_thrd_spec_wiki.append(Time(time_st_thrd, format='jd').strftime('%Y-%m-%d %H:%M:%S'))
        ted_thrd_spec_wiki.append(Time(time_ed_thrd, format='jd').strftime('%Y-%m-%d %H:%M:%S'))


        # Restore the earlier robust peak calculation: remove isolated spikes,
        # rebin in time/frequency, then compare smoothed peaks from 2-15 GHz.
        xp_peak, xp_peak_freq, xp_peaks_by_target = get_robust_spectrum_peaks(spec, fghz)
        for target, output in (
                (3, Fpk_XP_3GHz), (7, Fpk_XP_7GHz),
                (11, Fpk_XP_11GHz), (15, Fpk_XP_15GHz)):
            value = xp_peaks_by_target.get(target, np.nan)
            output.append("{:.1f}".format(value) if np.isfinite(value) else 'NA')

        Fpk_XP_totGHz.append("{:.1f}".format(xp_peak))
        freq_at_Fpk_XP.append(xp_peak_freq)
        Fpk_TP.append("{:.1f}".format(tp_peak) if tp_peak is not None else '')
        pk_freq_GHz.append(tp_peak_freq if tp_peak_freq is not None else '')


        # #####==================================================plot the dynamic spectrum and flux curves
        # tim_plt = time_spec.plot_date
        # freq_plt = fghz
        # spec_plt = spec  # np.nan_to_num(spec, nan=0.0)
        # spec_plt[spec_plt < 0] = 0.01

        # # drange = [1,np.max(spec_plt)]#np.max(spec_plt)
        # # drange = [0.01, 10]  # np.max(spec_plt)
        # drange = [0.01, np.percentile(spec_plt, 97.5)]

        # #####==================================================
        # if do_manu == 1:
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
        # if on_server == 1:
        #     figname = os.path.join(spec_data_dir, time_pk_obj.strftime('%Y'), f"{filename}.png")
        # else:
        #     figname = os.path.join(spec_img_dir, f"{filename}.png")
        # fig.savefig(figname, dpi=FIG_DPI, transparent=False)
        # plt.close()
        # print(f'Write spectrogram to {figname}')



    # #####==================================================
    data_csv = {
        # "ID": np.arange(len(flare_id)) + 1,
        # "Flare_ID": flare_id,
        'EO_tpeak': tpk_spec_wiki,
        'EO_tstart': tst_thrd_spec_wiki,
        'EO_tend': ted_thrd_spec_wiki,
        # 'EO_tstart_thrd': tst_thrd_spec_wiki,
        # 'EO_tend_thrd': ted_thrd_spec_wiki,
        # # 'EO_tstart_manu': tst_manu_spec_wiki,
        # # 'EO_tend_manu': ted_manu_spec_wiki,
        # 'EO_tstart_mad': tst_mad_spec_wiki,
        # 'EO_tend_mad': ted_mad_spec_wiki,
        # # 'depec_file': depec_file,
        'Fpk_XP_3GHz': Fpk_XP_3GHz,
        'Fpk_XP_7GHz': Fpk_XP_7GHz,
        'Fpk_XP_11GHz': Fpk_XP_11GHz,
        'Fpk_XP_15GHz': Fpk_XP_15GHz,
        'Fpk_XP_totGHz': Fpk_XP_totGHz,
        'freq_at_Fpk_XP': freq_at_Fpk_XP,
        'FPK_TP': Fpk_TP,
        'PK_FREQ_GHZ': pk_freq_GHz,
        'has_ql_movie': has_ql_movie,
        'has_fits': has_fits
    }

    # df = pd.DataFrame(data_csv)
    # csv_file_tsted = os.path.join(work_dir, 'get_spec_tst_ted_from_wiki_given_date.csv')
    # df.to_csv(csv_file_tsted, index=False)
    # print("Step 4: Times data saved to get_spec_tst_ted_from_wiki_given_date.csv")


    # ##=========================step 5: combined two csv files
    print("##=============step 5: combine two csv files")

    # initial csv
    df = pd.read_csv(updated_csv)
    # Create a new DataFrame from data_csv
    new_data_df = pd.DataFrame(data_csv)
    # Check if the lengths match; if not, raise an error
    if len(df) != len(new_data_df):
        raise ValueError("Error: The lengths of the original DataFrame and new data do not match.")
    # Concatenate the new columns to the original DataFrame
    df = pd.concat([df, new_data_df], axis=1)


    final_csv_file = os.path.join(work_dir, 'EOVSA_flare_list_from_wiki_sub.csv')
    df.to_csv(final_csv_file, index=False)
    print(f"Times data saved to {final_csv_file}")



    ##=========================step 6: write to MySQL 'EOVSA_flare_list_wiki_db'=========================
    print("##=============step 6: write flare info to MySQL 'EOVSA_flare_list_wiki_db'")

    ##=========================get flare_id_exist from mySQL

    connection = create_flare_db_connection()

    cursor = connection.cursor()
    cursor.execute("SELECT Flare_ID FROM EOVSA_flare_list_wiki_tb")
    flare_id_exist = cursor.fetchall()
    flare_id_exist_set = {int(row[0]) for row in flare_id_exist}

    cursor.close()
    connection.close()

    ##=========================

    # Connect to the database and get a cursor to access it:
    cnxn = create_flare_db_connection()

    cursor = cnxn.cursor()
    table = 'EOVSA_flare_list_wiki_tb'
    cursor.execute(f"SHOW COLUMNS FROM {table}")
    existing_columns = {row[0] for row in cursor.fetchall()}
    if 'has_ql_movie' not in existing_columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN has_ql_movie TINYINT(1) NOT NULL DEFAULT 0")
    if 'has_fits' not in existing_columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN has_fits TINYINT(1) NOT NULL DEFAULT 0")
    if 'FPK_TP' not in existing_columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN FPK_TP DOUBLE NULL")
    if 'PK_FREQ_GHZ' not in existing_columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN PK_FREQ_GHZ DOUBLE NULL")
    if 'Fpk_XP' not in existing_columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN Fpk_XP DOUBLE NULL")
    if 'freq_at_Fpk_XP' not in existing_columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN freq_at_Fpk_XP DOUBLE NULL")

    # Write to the database (add records)
    # Assume a database that mirrors the .csv file (first two lines below):
    #    Flare_ID,Date,Time,flare_class,EO_tstart,EO_tpeak,EO_tend,EO_xcen,EO_ycen
    #    20190415193100,2019-04-15,19:31:00,B3.3,2019-04-15 19:30:04,2019-04-15 19:32:21,2019-04-15 19:33:10,519.1,152.3
    # The Flare_ID is automatic (just incremented from 1), so is not explicitly written.  Also, separating Date and Time doesn't make sense, so combine into a single Date:

    columns = ['Flare_ID', 'Flare_class', 'EO_tstart', 'EO_tpeak', 'EO_tend', 'depec_file', 'depec_img', \
                'Fpk_XP_3GHz', 'Fpk_XP_7GHz', 'Fpk_XP_11GHz', 'Fpk_XP_15GHz']

    columns = ['Flare_ID', 'Flare_class', 'EO_tstart', 'EO_tpeak', 'EO_tend', \
                'depec_datafile_TP', 'depec_imgfile_TP', 'depec_datafile_XP', 'depec_imgfile_XP', \
                'Fpk_XP_3GHz', 'Fpk_XP_7GHz', 'Fpk_XP_11GHz', 'Fpk_XP_15GHz', \
                'Fpk_XP', 'freq_at_Fpk_XP', 'FPK_TP', 'PK_FREQ_GHZ', \
                'has_ql_movie', 'has_fits']


    values = []

    #####==================================================

    df = pd.read_csv(final_csv_file)
    flare_id = df['Flare_ID']
    dates = df['Date']
    times = df['Time (UT)']
    EO_tstart = df['EO_tstart']
    EO_tpeak = df['EO_tpeak']
    EO_tend = df['EO_tend']
    GOES_class = df['Flare_Class']
    depec_datafile_TP = df['depec_datafile_TP']
    depec_imgfile_TP = df['depec_imgfile_TP']
    depec_datafile_XP = df['depec_datafile_XP']
    depec_imgfile_XP = df['depec_imgfile_XP']
    Fpk_XP_3GHz = df['Fpk_XP_3GHz']
    Fpk_XP_7GHz = df['Fpk_XP_7GHz']
    Fpk_XP_11GHz = df['Fpk_XP_11GHz']
    Fpk_XP_15GHz = df['Fpk_XP_15GHz']
    Fpk_XP = df['Fpk_XP_totGHz']
    freq_at_Fpk_XP = df['freq_at_Fpk_XP']
    Fpk_TP = df['FPK_TP']
    pk_freq_GHz = df['PK_FREQ_GHZ']
    has_ql_movie = df['has_ql_movie'] if 'has_ql_movie' in df.columns else pd.Series([0] * len(df))
    has_fits = df['has_fits'] if 'has_fits' in df.columns else pd.Series([0] * len(df))

    def normalize_flare_class(value, flare_id_value):
        """Return a compact GOES class that fits the existing SQL column."""
        if pd.isna(value):
            return None
        original = str(value).replace('\xa0', ' ').strip()
        match = re.search(r'([ABCMX])\s*(\d+(?:\.\d+)?)', original, re.IGNORECASE)
        if match:
            normalized = match.group(1).upper() + match.group(2)
        elif original in ('', '?'):
            normalized = None if original == '' else '?'
        else:
            normalized = '?'

        # The deployed Flare_class column is short. Do not allow an unusual
        # value to abort all later updates; preserve it as unknown instead.
        if normalized is not None and len(normalized) > 5:
            normalized = '?'
        if normalized != original:
            print(
                f"WARNING: normalized Flare_class for {flare_id_value}: "
                f"{original!r} -> {normalized!r}"
            )
        return normalized

    ##=========================
    for i in range(len(flare_id)):
        flare_id_int = int(flare_id[i])
        flare_class_value = normalize_flare_class(GOES_class[i], flare_id_int)
        payload = [
            flare_id_int, flare_class_value, Time(EO_tstart[i]).jd, Time(EO_tpeak[i]).jd, Time(EO_tend[i]).jd,
            str(depec_datafile_TP[i]), str(depec_imgfile_TP[i]), str(depec_datafile_XP[i]), str(depec_imgfile_XP[i]),
            Fpk_XP_3GHz[i], Fpk_XP_7GHz[i], Fpk_XP_11GHz[i], Fpk_XP_15GHz[i],
            Fpk_XP[i], freq_at_Fpk_XP[i], Fpk_TP[i], pk_freq_GHz[i],
            int(has_ql_movie[i]) if not pd.isna(has_ql_movie[i]) else 0,
            int(has_fits[i]) if not pd.isna(has_fits[i]) else 0
        ]
        payload = [None if pd.isna(val) else val for val in payload]

        if flare_id_int not in flare_id_exist_set:
            values.append(payload)
            print("EOVSA_flare_list_wiki_tb Insert for ", flare_id_int)
            continue

        # Existing flare rows are updated in place to backfill fields that may have
        # been missing in earlier parser runs.
        update_cols = []
        update_vals = []
        for col_name, col_val in zip(columns[1:], payload[1:]):
            if col_val is None:
                continue
            if isinstance(col_val, str) and col_val.strip() == '':
                continue
            update_cols.append(f"{col_name} = %s")
            update_vals.append(col_val)

        if update_cols:
            update_query = f"UPDATE {table} SET " + ', '.join(update_cols) + " WHERE Flare_ID = %s"
            update_vals.append(flare_id_int)
            try:
                cursor.execute(update_query, tuple(update_vals))
                print("EOVSA_flare_list_wiki_tb Refresh for ", flare_id_int)
            except mysql.connector.errors.DataError as exc:
                print(
                    f"ERROR: skipping SQL refresh for {flare_id_int}: {exc}; "
                    f"Flare_class={flare_class_value!r}"
                )

    put_query = 'insert ignore into ' + table + ' (' + ','.join(columns) + ') values (' + ('%s,' * len(columns))[
                                                                                          :-1] + ')'

    cursor.executemany(put_query, values)
    cnxn.commit()  # Important!  The record will be deleted if you do not "commit" after a transaction


    ##=========================step 7: write to MySQL 'EOVSA_lightcurve_QL_db'=========================
    print("##=========================step 7: write flare light curves to MySQL 'EOVSA_lightcurve_QL_db'")

    cnxn = create_flare_lc_db_connection()

    def specdata_prep(dsfile_path):
        try:
            extension = os.path.splitext(dsfile_path)[1].lower()
            if extension == '.dat':
                data1 = rd_datfile(dsfile_path)
                time = np.asarray(data1['time'])
                freq = np.asarray(data1['fghz'])
                spec = np.asarray(data1['data'])
            elif extension == '.fits':
                with fits.open(dsfile_path) as eospecfits:
                    time = np.asarray(eospecfits[2].data['TIME'])
                    freq = np.asarray(eospecfits[1].data['FGHZ'])
                    spec = np.asarray(eospecfits[0].data)
            else:
                raise ValueError(f"unsupported spectrum file type: {extension}")

            if time.size == 0 or freq.size == 0 or spec.size == 0:
                raise ValueError("spectrum contains no data")
            if spec.shape != (freq.size, time.size):
                raise ValueError(
                    f"unexpected spectrum shape {spec.shape}; "
                    f"expected ({freq.size}, {time.size})"
                )

            freq_plt = [3, 7, 11, 15]
            freq_QL = np.zeros(len(freq_plt))
            spec_QL = np.zeros((len(freq_plt), time.size))
            for ff, fghz in enumerate(freq_plt):
                ind = np.argmin(np.abs(freq - fghz))
                freq_QL[ff] = freq[ind]
                spec_QL[ff, :] = spec[ind, :]
            return time, freq_QL, spec_QL
        except Exception as e:
            print(f"Errors of reading data {dsfile_path}: {e}. Existing SQL data will be preserved.")
            return None, None, None

    def replace_lightcurve(cursor, flare_id, product, times, freqs, spec):
        """Replace one TP/XP light curve after its source data has been validated."""
        if product not in ('TP', 'XP'):
            raise ValueError(f"Unsupported light-curve product: {product}")

        cursor.execute(f"DELETE FROM flux_QL_{product} WHERE Flare_ID = %s", (flare_id,))
        cursor.execute(f"DELETE FROM freq_QL_{product} WHERE Flare_ID = %s", (flare_id,))
        cursor.execute(f"DELETE FROM time_QL_{product} WHERE Flare_ID = %s", (flare_id,))

        cursor.executemany(
            f"INSERT INTO time_QL_{product} VALUES (%s, %s, %s)",
            [(flare_id, index, float(value)) for index, value in enumerate(times)]
        )
        cursor.executemany(
            f"INSERT INTO freq_QL_{product} VALUES (%s, %s, %s)",
            [(flare_id, index, round(float(value), 3)) for index, value in enumerate(freqs)]
        )

        flux_rows = []
        for ff, row in enumerate(spec):
            for tt, value in enumerate(row):
                sql_value = round(float(value), 3) if np.isfinite(value) else None
                flux_rows.append((flare_id, ff, tt, sql_value))
        cursor.executemany(
            f"INSERT INTO flux_QL_{product} VALUES (%s, %s, %s, %s)",
            flux_rows
        )

    df = pd.read_csv(final_csv_file)
    flare_id_tot = df['Flare_ID']
    EO_tpeak_tot = df['EO_tpeak']
    depec_file_tp_tot = df['depec_datafile_TP']
    depec_file_xp_tot = df['depec_datafile_XP']

    for i, date_str in enumerate(EO_tpeak_tot):
        datetp = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        Flare_ID = int(flare_id_tot[i])
        print(f"Flare_ID {Flare_ID}: refreshing existing light-curve data")

        dsfile_path_tp = '/common/webplots/events/' + str(Flare_ID)[0:4] + '/' + str(depec_file_tp_tot[i])
        time_QL_tp, freq_QL_tp, spec_QL_tp = specdata_prep(dsfile_path_tp)

        dsfile_path_xp = '/common/webplots/events/' + str(Flare_ID)[0:4] + '/' + str(depec_file_xp_tot[i])
        time_QL_xp, freq_QL_xp, spec_QL_xp = specdata_prep(dsfile_path_xp)

        if time_QL_tp is None and time_QL_xp is None:
            print(f"Flare_ID {Flare_ID}: no valid TP or XP data; keeping existing SQL rows")
            continue

        cursor = cnxn.cursor()
        try:
            cursor.execute("SELECT 1 FROM Flare_IDs WHERE Flare_ID = %s LIMIT 1", (Flare_ID,))
            if cursor.fetchone() is None:
                cursor.execute(
                    "INSERT INTO Flare_IDs (Flare_ID, `Index`) VALUES (%s, %s)",
                    (Flare_ID, i + 1)
                )

            if time_QL_tp is not None:
                replace_lightcurve(cursor, Flare_ID, 'TP', time_QL_tp, freq_QL_tp, spec_QL_tp)
                print(f"Flare_ID {Flare_ID}: TP data overwritten")
            else:
                print(f"Flare_ID {Flare_ID}: TP source unavailable; existing TP data preserved")

            if time_QL_xp is not None:
                replace_lightcurve(cursor, Flare_ID, 'XP', time_QL_xp, freq_QL_xp, spec_QL_xp)
                print(f"Flare_ID {Flare_ID}: XP data overwritten")
            else:
                print(f"Flare_ID {Flare_ID}: XP source unavailable; existing XP data preserved")

            cnxn.commit()
        except Exception:
            cnxn.rollback()
            raise
        finally:
            cursor.close()

    cnxn.close()

    print("Success!")


if __name__ == "__main__":
    main()
