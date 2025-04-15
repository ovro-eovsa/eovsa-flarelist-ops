

#####==================================================
'''to delete the flare_ID data in mySQL database, for the flare list web
'''
import mysql.connector
import os
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

def get_eo_flare_id_inSQL(timerange=['2021-05-01T00:00:00', '2021-06-01T00:00:00']):
    from astropy.time import Time
    # Convert timerange to Julian Dates
    start_time = Time(timerange[0]).jd
    end_time = Time(timerange[1]).jd
    # Connect the database
    connection_db = create_flare_db_connection()
    cursor = connection_db.cursor()
    # Query Flare_ID and EO_tpeak within the time range
    cursor.execute("""
        SELECT Flare_ID FROM EOVSA_flare_list_wiki_tb
        WHERE EO_tpeak BETWEEN %s AND %s
    """, (start_time, end_time))
    # Get Flare_ID in string list    
    flare_ids = cursor.fetchall()
    flare_ids_filtered = [str(f[0]) for f in flare_ids] if flare_ids else []  # Ensure format string list
    # Close the database connection
    cursor.close()
    connection_db.close()
    print(f"Get {len(flare_ids_filtered)} EO Flare_IDs in SQL database")
    return flare_ids_filtered
# # Example usage
# flare_ids = get_eo_flare_id_inSQL(timerange=['2021-05-01T10:00:00', '2021-06-01T10:00:00'])



def delete_eo_flare_id_inSQL(flare_ids=['']):
    '''To delete flare id inSQL, works for a given flare_ids string list
    '''
    connection_db = create_flare_db_connection()
    cursor_db = None

    try:
        cursor_db = connection_db.cursor()
        for flare_id in flare_ids:
            try:
                # SELECT: must consume results
                check_query_db = "SELECT Flare_ID FROM EOVSA_flare_list_wiki_tb WHERE Flare_ID = %s"
                cursor_db.execute(check_query_db, (flare_id,))
                flare_exists_db = cursor_db.fetchone()  # ensures result is read
                while cursor_db.nextset():
                    pass
                if flare_exists_db:
                    delete_query = "DELETE FROM EOVSA_flare_list_wiki_tb WHERE Flare_ID = %s"
                    cursor_db.execute(delete_query, (flare_id,))
                    print(f"Deleted flare list table for Flare_ID: {flare_id}")
                else:
                    print(f"Flare_ID {flare_id} does not exist in DB database")
            except mysql.connector.Error as err:
                print(f"Error processing Flare_ID {flare_id}: {err}")
                try:
                    # Try to clear unread results to continue processing next ID
                    while cursor_db.nextset():
                        pass
                except:
                    pass
        connection_db.commit()

    except mysql.connector.Error as err:
        print(f"DB Connection Error: {err}")

    finally:
        # Close cursor first
        if cursor_db:
            try:
                cursor_db.close()
            except mysql.connector.Error as err:
                print(f"Error closing cursor_db: {err}")
        # Avoid "Unread result" error by skipping is_connected()
        try:
            connection_db.close()
        except mysql.connector.Error as err:
            print(f"Error closing connection_db: {err}")


    #####for light curves
    connection_lc = create_flare_lc_db_connection()
    try:
        # Check if flare_id exists in the lc database
        cursor_lc = connection_lc.cursor()
        queries = [
            "DELETE FROM time_QL_TP WHERE Flare_ID = %s",
            "DELETE FROM freq_QL_TP WHERE Flare_ID = %s",
            "DELETE FROM flux_QL_TP WHERE Flare_ID = %s",
            "DELETE FROM time_QL_XP WHERE Flare_ID = %s",
            "DELETE FROM freq_QL_XP WHERE Flare_ID = %s",
            "DELETE FROM flux_QL_XP WHERE Flare_ID = %s",            
            "DELETE FROM Flare_IDs WHERE Flare_ID = %s"           
        ]
        for flare_id in flare_ids:
            check_query_lc = "SELECT Flare_ID FROM Flare_IDs WHERE Flare_ID = %s"
            cursor_lc.execute(check_query_lc, (flare_id,))
            flare_exists_lc = cursor_lc.fetchone()

            if flare_exists_lc:
                for query in queries:
                    cursor_lc.execute(query, (flare_id,))
                print(f"Deleted flare LC information for Flare_ID: {flare_id}")
            else:
                print(f"Flare_ID {flare_id} does not exist in LC database")
        # Commit once after all deletions
        connection_lc.commit()
        cursor_lc.close()
        connection_lc.close()

    except mysql.connector.Error as err:
        print(f"Error: {err}")

    finally:
        if connection_lc.is_connected():
            try:
                connection_lc.close()
            except mysql.connector.Error as err:
                print(f"Error closing connection_lc: {err}")

# # Example usage
# delete_eo_flare_id_inSQL(flare_ids=['20240731193000', '20240731193000'])

# flare_ids = get_eo_flare_id_inSQL(timerange=['2021-05-01T10:00:00', '2021-06-01T10:00:00'])
# delete_eo_flare_id_inSQL(flare_ids=flare_ids)



#####==================================================
'''to change EOVSA flare IDs on web
'''
def change_EO_flareID(flare_id, flare_id_new):
    """
    Change EOVSA FITS/movie/ms/slfcal_ms files from flare_id to new_flare_id
    flare_id, flare_id_new in string format
    example: change_EO_flareID('20240713150500', '20240713151000')
    """
    import os
    fitsdir_web_tp = '/data1/eovsa/fits/flares/' #'YYYY/MM/DD/flare_id/'
    fitsdir_web = os.path.join(fitsdir_web_tp, flare_id[:4], flare_id[4:6], flare_id[6:8])
    fitsdir_web_new = os.path.join(fitsdir_web_tp, flare_id_new[:4], flare_id_new[4:6], flare_id_new[6:8])
    src = os.path.join(fitsdir_web, flare_id)
    dst = os.path.join(fitsdir_web_new, flare_id_new)
    os.rename(src, dst)
    for f in ['.calibrated.ms.tar.gz', '.selfcalibrated.ms.tar.gz', '.caltables.tar.gz']:
        src = os.path.join(fitsdir_web, flare_id_new, flare_id + f)
        dst = os.path.join(fitsdir_web_new, flare_id_new, flare_id_new + f)
        os.rename(src, dst)

    movdir_web_tp = '/common/webplots/SynopticImg/eovsamedia/eovsa-browser/' #'YYYY/MM/DD/'
    movdir_web = os.path.join(movdir_web_tp, flare_id[:4], flare_id[4:6], flare_id[6:8])
    movdir_web_new = os.path.join(movdir_web_tp, flare_id_new[:4], flare_id_new[4:6], flare_id_new[6:8])
    src = os.path.join(movdir_web, 'eovsa.lev1_mbd_12s.flare_id_'+flare_id+'.mp4')
    dst = os.path.join(movdir_web_new, 'eovsa.lev1_mbd_12s.flare_id_'+flare_id_new+'.mp4')
    os.rename(src, dst)
    print("flare_id has been changed from ", flare_id, " to ", flare_id_new)


#####==================================================
'''to delete EOVSA flare IDs on web
'''
import os
import glob
def delete_EO_flareID(flare_id):
    """
    Delete EOVSA FITS/movie/ms/slfcal_ms files with flare_id.
    example: delete_EO_flareID('20240713150500')
    """
    fitsdir_web_tp = '/data1/eovsa/fits/flares/'  # 'YYYY/MM/DD/flare_id/'
    fitsdir_web = os.path.join(fitsdir_web_tp, flare_id[:4], flare_id[4:6], flare_id[6:8], flare_id)
    # Files to delete in the FITS directory
    for f in ['.calibrated.ms.tar.gz', '.selfcalibrated.ms.tar.gz', '.caltables.tar.gz']:
        file_to_delete = os.path.join(fitsdir_web, flare_id + f)
        if os.path.exists(file_to_delete):
            os.remove(file_to_delete)
            print(f"Deleted: {file_to_delete}")
        else:
            print(f"File not found: {file_to_delete}")
    # Delete all "eovsa*.fits" files in the FITS directory
    fits_files_pattern = os.path.join(fitsdir_web, 'eovsa.lev1_mbd_12s.*.image.fits')
    fits_files = glob.glob(fits_files_pattern)
    for file_to_delete in fits_files:
        os.remove(file_to_delete)
        print(f"Deleted: {file_to_delete}")
    # Movie directory structure (based on year, month, day, and flare_id)
    movdir_web_tp = '/common/webplots/SynopticImg/eovsamedia/eovsa-browser/'  # 'YYYY/MM/DD/'
    movdir_web = os.path.join(movdir_web_tp, flare_id[:4], flare_id[4:6], flare_id[6:8])
    movie_to_delete = os.path.join(movdir_web, f'eovsa.lev1_mbd_12s.flare_id_{flare_id}.mp4')
    # Delete the movie file
    if os.path.exists(movie_to_delete):
        os.remove(movie_to_delete)
        print(f"Deleted: {movie_to_delete}")
    else:
        print(f"Movie file not found: {movie_to_delete}")
    # Optionally, you can delete the directories if empty after deleting files
    try:
        if not os.listdir(fitsdir_web):  # Check if FITS directory is empty
            os.rmdir(fitsdir_web)
            print(f"Deleted empty directory: {fitsdir_web}")
    except FileNotFoundError:
        print(f"Directory not found: {fitsdir_web}")
    print(f"All products for flare_id {flare_id} have been deleted (if they existed).")


