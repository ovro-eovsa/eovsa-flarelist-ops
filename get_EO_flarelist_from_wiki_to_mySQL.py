##ipython get_EO_flarelist_from_wiki_to_mySQL.py
##=========================
from __future__ import print_function
import os
from datetime import datetime, timedelta
given_date = ['2024-02-23 22:00:00', '2024-03-05 00:00:00']##generate the flare list table within this timerange
given_date_strp = [datetime.strptime(given_date[0], '%Y-%m-%d %H:%M:%S'), datetime.strptime(given_date[1], '%Y-%m-%d %H:%M:%S')]
# Get the current time
given_date_strp = [datetime.now() - timedelta(days=10), datetime.now()]

work_dir = '/Users/xychen/Desktop/EOVSA/pipeline/0get_info_from_wiki_given_date/202402/'

on_server = 1
if on_server == 1:
    work_dir = '/data1/xychen/web_run_pipeline/data_run/'
    # move_csv_dir = '/data1/xychen/web_run_pipeline/'

os.makedirs(work_dir, exist_ok=True)

do_manu = 0 ##manually determine the start/end time of the radio burst by clicking on the Dspec


##=========================
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

##=========================Step 1: capture the radio peak times from flare list wiki webpage=========================
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
print("##=========================Step 1: capture the radio peak times from flare list wiki webpage")

url = "http://www.ovsa.njit.edu/wiki/index.php/Recent_Flare_List_(2021-)"
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content of the page using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    tables = soup.find_all("table", {"class": "wikitable"})

    date_data = []
    time_ut_data = []
    flare_class_data = []
    depec_file = []

    for table in tables:
        for row in table.find_all("tr"):
            cells = row.find_all("td")

            if len(cells) >= 3:
                date = cells[0].text.strip()
                time_ut = cells[1].text.strip()
                flare_class = cells[2].text.strip()

                datetime_strp = datetime.strptime(date + ' ' + time_ut, '%Y-%m-%d %H:%M')

                if given_date_strp[0] <= datetime_strp <= given_date_strp[1]:  # Check if the date is within the given range
                    date_data.append(date)
                    time_ut_data.append(time_ut)
                    flare_class_data.append(flare_class)

                    depec_file_tmp = ''
                    for cell in cells:
                        link_cell = cell.find('a', class_='external text', href=True, rel='nofollow')
                        if link_cell:
                            url = link_cell['href']
                            if url.endswith(".dat"):
                                depec_file_tmp = url.split('/')[-1].split('.dat')[0]
                    depec_file.append(depec_file_tmp)
                    print(date, time_ut, flare_class, depec_file_tmp)

                    # for cell in cells:
                    #     if cell.find(class_="thumbimage"):
                    #         img_tag = cell.find('img')
                    #         if img_tag:
                    #             src_attribute = img_tag.get('src')##'/wiki/images/a/ac/EOVSA_20240212_C5flare.png'
                    #             if src_attribute:
                    #                 depec_file.append(src_attribute.split('/')[-1].split('.')[0])
                    #             else:
                    #                 depec_file.append('')

    data = {
    "ID": np.arange(len(date_data))+1, 
    "Date": date_data, 
    "Time (UT)": time_ut_data, 
    "Flare Class": flare_class_data, 
    "depec_file": depec_file
    }

    df = pd.DataFrame(data)    
    df.to_csv(work_dir + "0get_time_from_wiki_given_date.csv", index=False)
    print("Date and Time (UT) data saved to 0get_time_from_wiki_given_date.csv")
else:
    print("Failed to retrieve the webpage. Status code:", response.status_code)





##=========================Step 2: get the start, peak, end time of the flare from GOES=========================
def get_flare_info_from_GOES(tpeak_str):
    # tpeak_str = "2019-04-15T19:30:00"
    # tpeak_str = tpeak_str
    from astropy.time import Time
    from sunpy.net import Fido
    from sunpy.net import attrs as a
    import numpy as np

    tpeak = Time(tpeak_str,format='isot').mjd * 24. ##in hours

    tstart = tpeak-1.
    tend = tpeak+1.

    tstart_str = ((Time(tstart/24.,format='mjd').isot).replace('-','/')).replace('T',' ')
    tend_str = ((Time(tend/24.,format='mjd').isot).replace('-','/')).replace('T',' ')
    # tstart_str = "2019/04/15 12:00:00"
    # tend_str = "2019/04/16 12:00:00"
    # event_type = "FL"
    result = Fido.search(a.Time(tstart_str, tend_str),
                         # a.hek.EventType(event_type),
                         # a.hek.FL.GOESCls > "M1.0",
                         a.hek.OBS.Observatory == "GOES")

    hek_results = result["hek"]
    #print(hek_results.colnames[::2])
    #print(result.show("hpc_bbox", "refs"))
    if len(hek_results) == 0 :
        GOES_class = '?'
        GOES_tstart = Time((tpeak-0.0)/24., format='mjd').iso
        GOES_tpeak = Time(tpeak/24., format='mjd').iso
        GOES_tend = Time((tpeak+0.0)/24., format='mjd').iso
        GOES_hgc_x = 0
        GOES_hgc_y = 0
    else:
        filtered_results = hek_results["fl_goescls", "event_starttime", "event_peaktime",
                                       "event_endtime", "ar_noaanum", "hgc_x", "hgc_y"]

        GOES_tpeak = hek_results["event_peaktime"]
        GOES_tpeak_mjd = GOES_tpeak.mjd * 24.

        if len(GOES_tpeak) == 1 :
            ind = 0
        if len(GOES_tpeak) > 1 :
            ind = np.argmin(abs(GOES_tpeak_mjd-tpeak))
        if len(GOES_tpeak) < 1 :
            print("ERRORs: No flares detected")
            return

        GOES_class = (hek_results["fl_goescls"])[ind]
        GOES_tstart = (hek_results["event_starttime"])[ind].iso
        GOES_tpeak = (hek_results["event_peaktime"])[ind].iso
        GOES_tend = (hek_results["event_endtime"])[ind].iso
        GOES_hgc_x = (hek_results["hgc_x"])[ind]
        GOES_hgc_y = (hek_results["hgc_y"])[ind]

    print(f"GOES Class {GOES_class} peaks on {GOES_tpeak} / EO radio on {tpeak_str}")
    return GOES_class, GOES_tstart, GOES_tpeak, GOES_tend, GOES_hgc_x, GOES_hgc_y




##=========================Step 3: reformate the date and time=========================
import pandas as pd
from datetime import datetime
print("##=========================Step 3: reformate the date and time")

flare_id = []
dates = []
times = []
depec_file = []

flare_class = []
GOES_flare_class = []
GOES_tstart = []
GOES_tpeak = []
GOES_tend = []
GOES_hgc_x = []
GOES_hgc_y = []

input_csv = work_dir+"0get_time_from_wiki_given_date.csv"
df = pd.read_csv(input_csv)

for index, row in df.iterrows():
    date_tp = row["Date"]
    time_tp = row["Time (UT)"]
    flare_class_tp = row["Flare Class"]
    depec_file_tp = row["depec_file"]

    # Convert the date to the desired format 'year/month/day'
    # date_obj = datetime.strptime(date_tp, "%d/%m/%Y")
    # date = date_obj.strftime("%Y/%m/%d")
    date = date_tp
    time = time_tp + ':00'

    eotime_flare_wiki = date.replace('/', '-') + 'T' + time
    GOES_class_tp, GOES_tstart_tp, GOES_tpeak_tp, GOES_tend_tp, GOES_hgc_x_tp, GOES_hgc_y_tp = get_flare_info_from_GOES(tpeak_str=eotime_flare_wiki)

    flare_id_tp = ((eotime_flare_wiki.replace('-', '')).replace('T', '')).replace(':', '')

    flare_id.append(flare_id_tp)
    dates.append(date)
    times.append(time)

    flare_class.append(flare_class_tp)

    GOES_flare_class.append(GOES_class_tp)
    GOES_tstart.append(GOES_tstart_tp.split('.')[0])
    GOES_tpeak.append(GOES_tpeak_tp.split('.')[0])
    GOES_tend.append(GOES_tend_tp.split('.')[0])
    GOES_hgc_x.append(GOES_hgc_x_tp)
    GOES_hgc_y.append(GOES_hgc_y_tp)
    depec_file.append(str(depec_file_tp))

new_data = {
    "ID": np.arange(len(flare_id))+1,
    "Flare_ID": flare_id,
    "Date": dates,
    "Time (UT)": times,
    "flare_class": flare_class,
    "GOES_flare_class": GOES_flare_class,
    "GOES_tstart": GOES_tstart,
    "GOES_tpeak": GOES_tpeak,
    "GOES_tend": GOES_tend,
    # "GOES_hgc_x": GOES_hgc_x,
    # "GOES_hgc_y": GOES_hgc_y
    "depec_file": depec_file
}
new_df = pd.DataFrame(new_data)

# Save the updated data to a new CSV file
output_csv = work_dir + "0get_time_from_wiki_given_date.csv"
new_df.to_csv(output_csv, index=False)

print("Step 3: Times data saved to", output_csv)





##=========================Step 4: download the spectrum data from flare list wiki webpage=========================
import requests
from bs4 import BeautifulSoup
import os
import re
print("##=========================Step 4: download the spectrum data from flare list wiki webpage")

# Function to download files from a given URL
def download_files_from_url(url, download_directory, given_date):
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
                match = re.match(r'EOVSA_(\d{4})(\d{2})(\d{2})', file_name)
                match = re.match(r'(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})', file_name)

                # Check if the file name matches the given date range pattern
                if match:
                    file_date = match.group(1) + '-' + match.group(2) + '-' + match.group(3)

                    # Check if the file date is within the given date range
                    if given_date_strp[0].strftime('%Y-%m-%d') <= file_date <= (given_date_strp[1]+timedelta(days=1)).strftime('%Y-%m-%d'):
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

# List of URLs to download files from
urls = [
    "http://ovsa.njit.edu/events/2019/",
    "http://ovsa.njit.edu/events/2021/",
    "http://ovsa.njit.edu/events/2022/",
    "http://ovsa.njit.edu/events/2023/",
    "http://ovsa.njit.edu/events/2024/"
]


if on_server == 1:
    spec_data_dir = "/common/webplots/events/"#YYYY/
    print("Spec data in ", spec_data_dir)
else:
    # Directory to store downloaded files
    spec_data_dir = work_dir + 'spec_data/'
    os.makedirs(spec_data_dir, exist_ok=True)
    # # Date range to filter files (e.g., 'yyyy-mm-dd')
    # given_date = ['2023-09-01', '2023-09-10']
    # Iterate through the list of URLs and download files from each
    for url in urls:
        file_name_download = download_files_from_url(url, spec_data_dir, given_date)






##=========================Step 5: plot the spectrum  and determine the start and end times of radio flux profiles=========================
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



##=========================Step 5: read and plot the spectrum data=========================
import glob
import numpy as np
import os
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from astropy.time import Time
import matplotlib.colors as mcolors
from matplotlib.dates import AutoDateFormatter, AutoDateLocator, num2date
import matplotlib.cm
import pandas as pd
from datetime import datetime, timedelta
import warnings
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

print("##=========================Step 5: read and plot the spectrum data")

file_path = work_dir + "0get_time_from_wiki_given_date.csv"
df = pd.read_csv(file_path)
flare_id = df['Flare_ID']
depec_file = df['depec_file']

if on_server == 1:
    files_wiki = [spec_data_dir + str(flare_id[i])[0:4] + "/" + str(file_name) + '.dat' for i, file_name in enumerate(depec_file)]
else:
    files_wiki = [spec_data_dir + str(file_name) +'.dat' for file_name in depec_file]

spec_img_dir = work_dir + 'spec_img/'
os.makedirs(spec_img_dir, exist_ok=True)

fontsize_pl=14.

##=========================
def moving_average(data, window_size):
    # Create a convolution kernel for the moving average
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')
window_size = 10

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
tpk_spec_wiki = []

tst_mad_spec_wiki = []
ted_mad_spec_wiki = []

tst_thrd_spec_wiki = []
ted_thrd_spec_wiki = []

tst_manu_spec_wiki = []
ted_manu_spec_wiki = []

depec_file = []

##=========================
for ww, file_wiki in enumerate(files_wiki):#len(files_wiki)

    filename1 = file_wiki.split('/')[-1]
    filename = filename1.split('.dat')[0]
    depec_file.append(filename)
    print("Spec data: ", filename)

    try:
        data1 = rd_datfile(file_wiki)

        time1 = data1['time']
        fghz = np.array(data1['fghz'])
        spec = np.array(data1['data'])
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
    time_str = Time(time1, format='jd').isot

    ##=========================try MAD method
    tpk_tot_ind = []
    
    tst_tot_ind = []
    ted_tot_ind = []
    mad_tot = []

    tst_tot_ind_thrd = []
    ted_tot_ind_thrd = []

    for ff in range(len(fghz)):
        good_channel = False
        flux_array = spec[ff,:] 
        flux_array = np.nan_to_num(flux_array, nan=0.0)
        flux_array[flux_array < 0] = 0.01

        ##=========================try MAD method
        median_flux = np.median(flux_array)
        abs_deviations = np.abs(flux_array - median_flux)
        mad = np.median(abs_deviations)# Calculate the median of the absolute deviations (MAD)

        threshold = 3.0 * mad
        mad_tot.append(threshold)

        outliers = abs_deviations > threshold
        outlier_ind = np.where(outliers)[0]
        if len(outlier_ind) > 0:
            tst_tot_ind.append(outlier_ind[0])
            ted_tot_ind.append(outlier_ind[-1])
        
        ##=========================try to set threshold        
        y = moving_average(flux_array, window_size)+0.001##flux_array
        peaks, _ = find_peaks(y, height=1.)
        noise_thrd_st = np.mean(y[0:5])
        noise_thrd_ed = np.mean(y[-5:])

        # print(ff, np.nanmean(y))

        if noise_thrd_st == 0:
            noise_thrd_st = 0.005*np.max(y)
        if noise_thrd_ed == 0:
            noise_thrd_ed = 0.01*np.max(y)
        if np.max(y) / noise_thrd_ed > 100:
            noise_thrd_ed = 0.02*np.max(y)             
        # noise_thrd_st = np.max([np.mean(y[0:10]),0.01*np.max(y)])
        # noise_thrd_ed = np.max([np.mean(y[-10:])*np.median(y[peaks]),0.01*np.max(y)])
        # if ff == 5:
        #     print(np.max(y), np.median(y[peaks]), np.mean(y[0:10]), np.mean(y[-10:]))
        
        for ind in range(len(y) - 5):
            if y[ind] < y[ind + 1] < y[ind + 2] < y[ind + 3]< y[ind + 4]< y[ind + 5]:
                if y[ind + 5] >= 2 * y[ind]:
                    if all(y[i] > noise_thrd_st for i in range(ind, ind + 6)):
                        tst_tot_ind_thrd.append(ind)
                        break
        ind_tmp = np.argmax(flux_array)-30
        for ind in range(len(y) - 5):
            if y[ind] > y[ind + 1] > y[ind + 2] > y[ind + 3] > y[ind + 4] > y[ind + 5]:
                if y[ind + 3] <= 2 * y[ind]:
                    if all(abs(y[i]) > noise_thrd_ed for i in range(ind, ind + 6)):
                        ind_tmp = ind+5
        ted_tot_ind_thrd.append(ind_tmp)
        
        ##=========================tpeak
        tpk_tot_ind.append(np.argmax(flux_array))



    time_st_mad = time_spec[int(np.median(np.array(tst_tot_ind)))]#[int(np.median(np.array(tst_tot_ind)))]
    time_ed_mad = time_spec[int(np.median(np.array(ted_tot_ind)))]

    time_st_thrd = time_spec[int(np.median(np.array(tst_tot_ind_thrd)))]
    time_ed_thrd = time_spec[int(np.median(np.array(ted_tot_ind_thrd)))]

    time_st = time_st_thrd
    time_ed = time_ed_thrd

    time_pk = time_spec[int(np.median(np.array(tpk_tot_ind)))]

    tpk_spec_wiki.append(Time(time_pk, format='jd').isot.replace('T', ' ').split('.')[0])

    tst_mad_spec_wiki.append(Time(time_st_mad, format='jd').isot.replace('T', ' ').split('.')[0])
    ted_mad_spec_wiki.append(Time(time_ed_mad, format='jd').isot.replace('T', ' ').split('.')[0])
    
    tst_thrd_spec_wiki.append(Time(time_st_thrd, format='jd').isot.replace('T', ' ').split('.')[0])
    ted_thrd_spec_wiki.append(Time(time_ed_thrd, format='jd').isot.replace('T', ' ').split('.')[0])

    # ##=========================find the peak
    # x = np.array(time_spec)
    # # y = spec[5,:]

    # # peaks, _ = find_peaks(y, height=1.)  # Adjust the 'height' parameter as needed
    # plt.figure(figsize=(10, 6))
    # plt.plot(y, label="Flux Profile")
    # # plt.plot(x[peaks], y[peaks], "ro", label="Peaks")
    # plt.xlabel("X-axis")
    # plt.ylabel("Flux")
    # plt.legend()


    ##=========================plot the dynamic spectrum and flux curves
    tim_plt = time_spec.plot_date
    freq_plt = fghz
    spec_plt = spec#np.nan_to_num(spec, nan=0.0)
    spec_plt[spec_plt < 0] = 0.01

    # drange = [1,np.max(spec_plt)]#np.max(spec_plt)
    drange = [0.01,10]#np.max(spec_plt)


    if do_manu==1:
        #####==================================================
        # Initialize global variables
        click_count = 0
        x1, y1, x2, y2 = 0, 0, 0, 0

        # Create a figure and connect the mouse click event handler
        fig, ax2 = plt.subplots()
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        ##=========================plot the flux curves and manully determine the start/end time
        freq_pl_tot = [2.,4.,6.,8.,10.,12.]#
        # freq_pl_tot = [3.,5.,7.,9.,11,14.]#

        cmap_flux = matplotlib.cm.get_cmap("jet",len(freq_pl_tot))#13 autumn_r

        for ff, freq_pl_temp in enumerate(freq_pl_tot):
            freq_pl_ind = np.argmin(np.abs(freq_plt-freq_pl_temp))
            ax2.plot(tim_plt, spec_plt[freq_pl_ind,:], label="{:.2f}".format(freq_plt[freq_pl_ind])+' GHz', c=cmap_flux(ff/(len(freq_pl_tot)-1)), linewidth=0.8)
            plt.hlines(mad_tot[freq_pl_ind], tim_plt[0], tim_plt[-1], linestyle='--', color=cmap_flux(ff/(len(freq_pl_tot)-1)), linewidth=0.5)

        ax2.set_ylabel('Flux [sfu]',fontsize=fontsize_pl+2)
        ax2.set_xlim(tim_plt[0], tim_plt[-1])
        # ax2.set_ylim(drange[0], drange[1])
        ax2.set_title(filename)

        locator = AutoDateLocator(minticks=2)
        ax2.xaxis.set_major_locator(locator)
        formatter = AutoDateFormatter(locator)
        formatter.scaled[1/24] = '%D %H'
        formatter.scaled[1/(24*60)] = '%H:%M'
        ax2.xaxis.set_major_formatter(formatter)
        ax2.set_autoscale_on(False)

        plt.xticks(fontsize=fontsize_pl)
        plt.yticks(fontsize=fontsize_pl)

        plt.fill_between([time_st.plot_date, time_ed.plot_date], -1e3, 1e4, color='gray', alpha=0.3, label='st/ed region')

        plt.legend(prop={'size':11})

        # Show the figure
        plt.show()
        ##=========================
        # After closing the figure, check if two clicks were made
        if click_count == 2:
            print("Clicked positions:", (x1, y1, x2, y2))
        else:
            x1 = time_st.plot_date
            x2 = time_ed.plot_date
            print("You need to click twice to select positions.")
        try:
            tst_manu_spec_wiki.append(Time(x1, format='plot_date').isot.replace('T', ' ').split('.')[0])
            ted_manu_spec_wiki.append(Time(x2, format='plot_date').isot.replace('T', ' ').split('.')[0])
        except ValueError:
            print(f"The input x1/x2 does not match the plot_date format.")
            tst_manu_spec_wiki.append(Time(time_st, format='jd').isot.replace('T', ' ').split('.')[0])
            ted_manu_spec_wiki.append(Time(time_ed, format='jd').isot.replace('T', ' ').split('.')[0])
    else:
        tst_manu_spec_wiki.append('')
        ted_manu_spec_wiki.append('')



    #####==================================================plot 1-1 spectrum
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_axes([0.1, 0.55, 0.75, 0.36])

    ph1 = ax1.pcolormesh(tim_plt, freq_plt, spec_plt, norm=mcolors.LogNorm(vmin=drange[0], vmax=drange[1]), cmap='viridis')#

    ax1.set_title(filename, fontsize=fontsize_pl+2)
    ax1.set_ylabel('Frequency [GHz]', fontsize=fontsize_pl+2)

    ax1.set_xlim(tim_plt[0], tim_plt[-1])
    # ax1.set_ylim(freq_plt[fidx[0]], freq_plt[fidx[-1]])

    locator = AutoDateLocator(minticks=2)
    ax1.xaxis.set_major_locator(locator)
    formatter = AutoDateFormatter(locator)
    formatter.scaled[1/24] = '%D %H'
    formatter.scaled[1/(24*60)] = '%H:%M'
    ax1.xaxis.set_major_formatter(formatter)
    ax1.set_autoscale_on(False)

    plt.xticks(fontsize=fontsize_pl)
    plt.yticks(fontsize=fontsize_pl)

    cax = fig.add_axes([0.86, 0.55, 0.012, 0.36])
    cbar = plt.colorbar(ph1, ax=ax1,  cax=cax)#shrink=0.8, pad=0.05
    cbar.set_label('Flux  (sfu)', fontsize=fontsize_pl)


    #####==================================================plot 1-2 flux curve
    ax2 = fig.add_axes([0.1, 0.1, 0.75, 0.36])

    freq_pl_tot = [2.,4.,6.,8.,10.,12.]#
    # freq_pl_tot=[3.,5.,7.,9.,11,14.]#

    cmap_flux = matplotlib.cm.get_cmap("jet",len(freq_pl_tot))#13 autumn_r


    for ff, freq_pl_temp in enumerate(freq_pl_tot):
        freq_pl_ind = np.argmin(np.abs(freq_plt-freq_pl_temp))
        ax2.plot(tim_plt, spec_plt[freq_pl_ind,:], label="{:.2f}".format(freq_plt[freq_pl_ind])+' GHz', c=cmap_flux(ff/(len(freq_pl_tot)-1)), linewidth=0.8)
        plt.hlines(mad_tot[freq_pl_ind], tim_plt[0], tim_plt[-1], linestyle='--', color=cmap_flux(ff/(len(freq_pl_tot)-1)), linewidth=0.5)


    ax2.set_ylabel('Flux [sfu]',fontsize=fontsize_pl+2)
    ax2.set_xlim(tim_plt[0], tim_plt[-1])
    # ax2.set_ylim(drange[0], drange[1])

    locator = AutoDateLocator(minticks=2)
    ax2.xaxis.set_major_locator(locator)
    formatter = AutoDateFormatter(locator)
    formatter.scaled[1/24] = '%D %H'
    formatter.scaled[1/(24*60)] = '%H:%M'
    ax2.xaxis.set_major_formatter(formatter)
    ax2.set_autoscale_on(False)

    plt.xticks(fontsize=fontsize_pl)
    plt.yticks(fontsize=fontsize_pl)

    plt.fill_between([time_st.plot_date, time_ed.plot_date], -1e3, 1e4, color='gray', alpha=0.3, label='st/ed region')
    if do_manu==1:
        plt.fill_between([x1, x2], -1e3, 1e4, color='red', alpha=0.1, label='st/ed manu')

    plt.legend(prop={'size':9})

    #####==================================================
    fig.savefig(spec_img_dir + '/' + filename + '.jpg', dpi=400, transparent=False)
    plt.close()


data_csv = {
    "ID": np.arange(len(flare_id))+1,
    "Flare_ID": flare_id,
    'EO_tpeak': tpk_spec_wiki,
    'EO_tstart_thrd': tst_thrd_spec_wiki,
    'EO_tend_thrd': ted_thrd_spec_wiki,    
    'EO_tstart_manu': tst_manu_spec_wiki,
    'EO_tend_manu': ted_manu_spec_wiki,
    'EO_tstart_mad': tst_mad_spec_wiki,
    'EO_tend_mad': ted_mad_spec_wiki,
    'depec_file': df['depec_file']
}


df = pd.DataFrame(data_csv)
df.to_csv(work_dir + '/0get_spec_tst_ted_from_wiki_given_date.csv', index=False)

print("Step 5: Times data saved to 0get_spec_tst_ted_from_wiki_given_date.csv")




##=========================Step 6: combine two files=========================
import pandas as pd
import numpy as np
from astropy.time import Time
print("##=========================Step 6: combine two files")

file_path_time = work_dir + '0get_time_from_wiki_given_date.csv'
file_path_tst_ted = work_dir + '0get_spec_tst_ted_from_wiki_given_date.csv'


df_time = pd.read_csv(file_path_time)
df_tst_ted = pd.read_csv(file_path_tst_ted)

flare_id = df_time['Flare_ID']
dates = df_time['Date']
times = df_time['Time (UT)']
flare_class = df_time['flare_class']
GOES_tstart = df_time['GOES_tstart']
GOES_tpeak = df_time['GOES_tpeak']
GOES_tend = df_time['GOES_tend']

# EO_tpeak_tot = df_tst_ted['EO_tpeak']
# EO_tstart_tot = df_tst_ted['EO_tstart_manu']
# EO_tend_tot = df_tst_ted['EO_tend_manu']


EO_tpeak_tot = df_tst_ted['EO_tpeak']
EO_tstart_tot = df_tst_ted['EO_tstart_thrd']
EO_tend_tot = df_tst_ted['EO_tend_thrd']
depec_file_tot = df_tst_ted['depec_file']

##=========================
dates_times = [i + ' ' + j for i, j in zip(dates, times)]

EO_tpeak_tot_mjd = [Time((i.replace('/', '-')).replace(' ', 'T'),format='isot').mjd * 24. for i in EO_tpeak_tot]

EO_tpeak = []
EO_tstart = []
EO_tend = []
depec_file = []

for tpeak_str in dates_times:
    
    tpeak = Time((tpeak_str.replace('/', '-')).replace(' ', 'T'),format='isot').mjd * 24. ##in hours
    ind = np.abs(EO_tpeak_tot_mjd - tpeak).argmin()
    
    EO_tpeak.append(EO_tpeak_tot[ind])
    EO_tstart.append(EO_tstart_tot[ind])
    EO_tend.append(EO_tend_tot[ind])
    depec_file.append(depec_file_tot[ind])



##=========================
data_csv = {
    "ID": np.arange(len(flare_id))+1,
    "Flare_ID": flare_id,
    "Date": dates,
    "Time (UT)": times,
    "flare_class": flare_class,
    "EO_tstart": EO_tstart,
    "EO_tpeak": EO_tpeak,
    "EO_tend": EO_tend,
    "GOES_tstart": GOES_tstart,
    "GOES_tpeak": GOES_tpeak,
    "GOES_tend": GOES_tend,
    "depec_file": depec_file
}


df = pd.DataFrame(data_csv)
df.to_csv(work_dir+'0get_info_from_wiki_given_date_sub.csv', index=False)

print("Step 6: Times data saved to 0get_info_from_wiki_given_date_sub.csv")


if on_server == 1:
    # os.rename(work_dir + '0get_info_from_wiki_given_date_sub.csv', move_csv_dir+'EOVSA_flare_list_from_wiki_sub.csv')
    os.rename(work_dir + '0get_info_from_wiki_given_date_sub.csv', work_dir+'EOVSA_flare_list_from_wiki_sub.csv')
    print("Renamed to ", "EOVSA_flare_list_from_wiki_sub.csv")






##=========================Step 7: write to MySQL 'EOVSA_flare_list_wiki_db'=========================
print("##=========================Step 7: write to MySQL 'EOVSA_flare_list_wiki_db'")


##=========================get flare_id_exist from mySQL
import mysql.connector
import os

connection = mysql.connector.connect(
    host = os.getenv('FLARE_DB_HOST'),
    database = os.getenv('FLARE_DB_DATABASE'),
    user = os.getenv('FLARE_DB_USER'),
    password = os.getenv('FLARE_DB_PASSWORD')
)

cursor = connection.cursor()

cursor.execute("SELECT Flare_ID FROM EOVSA_flare_list_wiki_tb")
flare_id_exist = cursor.fetchall()

cursor.close()
connection.close()


##=========================
import mysql.connector
from astropy.time import Time
import numpy as np
import sys
from datetime import datetime
import pandas as pd

# Connect to the database and get a cursor to access it:
cnxn = mysql.connector.connect(
    user = os.getenv('FLARE_DB_USER'), 
    passwd = os.getenv('FLARE_DB_PASSWORD'), 
    host = os.getenv('FLARE_DB_HOST'), 
    database = os.getenv('FLARE_DB_DATABASE'))

cursor = cnxn.cursor()
table = 'EOVSA_flare_list_wiki_tb'

# Write to the database (add records)
# Assume a database that mirrors the .csv file (first two lines below):
#    Flare_ID,Date,Time,flare_class,EO_tstart,EO_tpeak,EO_tend,EO_xcen,EO_ycen
#    20190415193100,2019-04-15,19:31:00,B3.3,2019-04-15 19:30:04,2019-04-15 19:32:21,2019-04-15 19:33:10,519.1,152.3
# The Flare_ID is automatic (just incremented from 1), so is not explicitly written.  Also, separating Date and Time doesn't make sense, so combine into a single Date:

columns = ['Flare_ID', 'Flare_class', 'EO_tstart', 'EO_tpeak', 'EO_tend', 'EO_xcen', 'EO_ycen', 'depec_file']
columns = ['Flare_ID', 'Flare_class', 'EO_tstart', 'EO_tpeak', 'EO_tend', 'depec_file']

values = []



#####==================================================
file_path = '/data1/xychen/flaskenv/EOVSA_flare_list_from_wiki_sub.csv'
file_path = work_dir+'EOVSA_flare_list_from_wiki_sub.csv'

df = pd.read_csv(file_path)
flare_id = df['Flare_ID']
dates = df['Date']
times = df['Time (UT)']
EO_tstart = df['EO_tstart']
EO_tpeak = df['EO_tpeak']
EO_tend = df['EO_tend']
GOES_class = df['flare_class']
# EO_xcen = df['EO_xcen']
# EO_ycen = df['EO_ycen']
depec_file = df['depec_file']

##=========================
for i in range(len(flare_id)):
  if not any(int(flare_id[i]) == existing_flare_id[0] for existing_flare_id in flare_id_exist):
    date = Time(dates[i]+' '+times[i]).jd
    # newlist = [int(flare_id[i]), GOES_class[i], Time(EO_tstart[i]).jd, Time(EO_tpeak[i]).jd, Time(EO_tend[i]).jd, EO_xcen[i], EO_ycen[i], depec_file[i]]
    newlist = [int(flare_id[i]), GOES_class[i], Time(EO_tstart[i]).jd, Time(EO_tpeak[i]).jd, Time(EO_tend[i]).jd, str(depec_file[i])]

    values.append(newlist)
    print("EOVSA_flare_list_wiki_tb Update for ", int(flare_id[i]))


values = [[None if pd.isna(val) else val for val in sublist] for sublist in values]

put_query = 'insert ignore into '+table+' ('+','.join(columns)+') values ('+('%s,'*len(columns))[:-1]+')'

cursor.executemany(put_query, values)
cnxn.commit()    # Important!  The record will be deleted if you do not "commit" after a transaction






##=========================Step 8: write to MySQL 'EOVSA_lightcurve_QL_db'=========================
print("##=========================Step 8: write to MySQL 'EOVSA_lightcurve_QL_db'")

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


def spec_rebin(time, freq, spec, t_step=12, f_step=1, do_mean=True):
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





##=========================get flare_id_exist from mySQL
import mysql.connector
import os

connection = mysql.connector.connect(
    host=os.getenv('FLARE_DB_HOST'),
    database=os.getenv('FLARE_LC_DB_DATABASE'),
    user=os.getenv('FLARE_DB_USER'),
    password=os.getenv('FLARE_DB_PASSWORD')
)

cursor = connection.cursor()
cursor.execute("SELECT DISTINCT Flare_ID FROM freq_QL")
flare_id_exist = cursor.fetchall()

cursor.close()
connection.close()



##=========================
import mysql.connector
from astropy.time import Time
import numpy as np
import sys
from datetime import datetime
import pandas as pd

# Connect to the database and get a cursor to access it:
cnxn = mysql.connector.connect(
    user = os.getenv('FLARE_DB_USER'), 
    passwd = os.getenv('FLARE_DB_PASSWORD'), 
    host = os.getenv('FLARE_DB_HOST'), 
    database = os.getenv('FLARE_LC_DB_DATABASE'))



#####==================================================data preparation
file_path = '/data1/xychen/flaskenv/EOVSA_flare_list_from_wiki_sub.csv'#
file_path = work_dir+'EOVSA_flare_list_from_wiki_sub.csv'

df = pd.read_csv(file_path)
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

        dsfile_path = '/common/webplots/events/'+ str(Flare_ID)[0:4] + '/' + depec_file + '.dat'
        try:
            data1 = rd_datfile(dsfile_path)

            time = data1['time'] ##in jd
            freq = np.array(data1['fghz'])
            spec = np.array(data1['data'])

            time_new, freq_new, spec_new = spec_rebin(time, freq, spec, t_step=4, f_step=1, do_mean=False)

            freq_plt = [3, 7, 11, 15]
            freq_QL = np.zeros(len(freq_plt))
            spec_QL = np.zeros((len(freq_plt), len(time_new)))

            for ff, fghz in enumerate(freq_plt):
                ind = np.argmin(np.abs(freq_new - fghz))
                # print(ind, fghz, freq_new[ind])
                freq_QL[ff] = freq_new[ind]
                spec_QL[ff,:] = spec_new[ind,:]

            time_QL = time_new
        except Exception as e:
            print("Errors of reading data - flux set to be 0")
            date_list = []
            for mm in range(20):
                date_list.append(datetp + timedelta(seconds=12*mm))
            time_QL = [date_obj.toordinal() + 1721425.5 for date_obj in date_list]
            freq_QL = [3, 7, 11, 15]
            spec_QL = np.zeros((len(freq_QL), len(time_QL)))+1e-3


        # #####==================================================
        # # cursor = cnxn.cursor()
        # # # cursor.execute("INSERT INTO Flare_IDs VALUES ()")
        # # # Flare_ID = cursor.lastrowid
        # # # Flare_ID = 20240101083000
        # # select_query = "SELECT * FROM Flare_IDs WHERE Flare_ID = %s"
        # # cursor.execute(select_query, (Flare_ID,))
        # # existing_records = cursor.fetchone()
        # # if existing_records:
        # #     delete_query = "DELETE FROM Flare_IDs WHERE Flare_ID = %s"
        # #     cursor.execute(delete_query, (Flare_ID,))


        # tables = ["time_QL", "freq_QL", "flux_QL", "Flare_IDs"]  # Child tables first, parent table last
        # cursor = cnxn.cursor()

        # for table in tables:
        #     # Check if Flare_ID already exists in the current table
        #     select_query = f"SELECT * FROM {table} WHERE Flare_ID = %s"
        #     cursor.execute(select_query, (Flare_ID,))
        #     existing_records = cursor.fetchall()
        #     # If Flare_ID exists, delete all related records
        #     if existing_records:
        #         print(f"To delete the {Flare_ID} in table ", table)
        #         delete_query = f"DELETE FROM {table} WHERE Flare_ID = %s"
        #         cursor.execute(delete_query, (Flare_ID,))
        # cnxn.commit()
        # cursor.close()

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
        # insert_query = "INSERT INTO Flare_IDs (Flare_ID) VALUES (%s)"
        # cursor.execute(insert_query, (Flare_ID,))
        # cursor.execute("INSERT INTO Flare_IDs VALUES (%s, %s)", (Flare_ID, i+1))
        insert_query = "INSERT INTO Flare_IDs (Flare_ID, `Index`) VALUES (%s, %s)"
        cursor.execute(insert_query, (Flare_ID, i+1))

        cnxn.commit()
        cursor.close()

        #####==================================================
        # file_path = '/data1/xychen/flaskenv/spec_Tdata_QL/EOVSA_TPall_20220118_QLdata.npz'
        # npz = np.load(file_path, allow_pickle=True)
        # time_QL = npz['time_QL']
        # freq_QL = npz['freq_QL']
        # spec_QL = npz['spec_QL']

        #####=========================
        cursor = cnxn.cursor()
        for index, value in enumerate(time_QL):
            jd_time = value#Time(value).jd
            cursor.execute("INSERT INTO time_QL VALUES (%s, %s, %s)", (Flare_ID, index, jd_time))
        cnxn.commit()
        cursor.close()


        #####=========================
        cursor = cnxn.cursor()
        for index, value in enumerate(freq_QL):
            cursor.execute("INSERT INTO freq_QL VALUES (%s, %s, %s)", (Flare_ID, index, round(value, 3)))
        cnxn.commit()
        cursor.close()

        #####=========================
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