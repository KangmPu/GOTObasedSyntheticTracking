from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service 
import time
from bs4 import BeautifulSoup
import pandas as pd
import re

def get_ephemeris_MPC(target_name,obscode,start_date,interval,step_count):
    options = Options()
    options.add_argument("--headless=new")  
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)



    # MPC Ephemeris 
    driver.get("https://minorplanetcenter.net/iau/MPEph/MPEph.html")

    # 
    object_input = driver.find_element(By.NAME, "TextArea")
    object_input.clear()
    object_input.send_keys(target_name)

    # START date
    date_input = driver.find_element(By.NAME, "d")
    date_input.clear()
    date_input.send_keys(start_date)

    # time interval
    interval_input = driver.find_element(By.NAME, "i")
    interval_input.clear()
    interval_input.send_keys(interval)

    # unit setting
    hour_radio = driver.find_element(By.CSS_SELECTOR, 'input[name="u"][value="m"]')
    hour_radio.click()

    # clearing Observatory Code（

    driver.find_element(By.NAME, "c").clear()
    driver.find_element(By.NAME, "c").send_keys(obscode)

    # unit decimal degrees
    unit_radio = driver.find_element(By.CSS_SELECTOR, 'input[name="raty"][value="x"]')
    driver.execute_script("arguments[0].scrollIntoView();", unit_radio)
    unit_radio.click()

    # rows of output
    l_input = driver.find_element(By.NAME, "l")
    l_input.clear()
    l_input.send_keys(step_count)

    #submit the page
    submit_btn = driver.find_element(By.XPATH, '//input[@type="submit" and contains(@value, "Get ephemerides")]')
    submit_btn.click()

    #wait and get the result
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "pre")))
    ephemeris_text = driver.find_element(By.TAG_NAME, "pre").text

    html_content = driver.page_source

    # browser closed
    driver.quit()
    def decode_mpc_provisional(designation):
        """
        convert（ K25H00T）to（ 2025 HT00 ）
        """
        if not designation.startswith('K') or len(designation) != 7:
            return designation
            raise ValueError("should in the form of K+YY+4")

    # dealing with name
        year = 2000 + int(designation[1:3])  # "25" → 2025
        half_month_code = designation[3]     
        seq = designation[4:7]    
        
        if seq[:2] == '00':
            seq_number = int(seq[:2])
            return f"{year} {half_month_code}{seq[2]}"
        else:
            # K25K01T → 2025 KT1
            seq_number = int(seq[:2])
            return f"{year} {half_month_code}{seq[2]}{seq_number}"

    soup = BeautifulSoup(html_content, "html.parser")

    # block between <pre> is Ephemeris 
    pre_blocks = soup.find_all("pre")

    all_objects_data = []

    for pre in pre_blocks:
        lines = pre.get_text().splitlines()

        if not lines or len(lines) < 3:
            continue

        object_header = lines[0].strip()
        header1 = lines[1]
        header2 = lines[2]

        # start from 4th row
        for line in lines[3:]:
            if not line.strip():
                continue
            try:
    #            print(line)
                parts = line.split()
    #            print(object_header.split()[0])
                data = {
                    "Date": f"{parts[0]} {parts[1]} {parts[2]}",
                    "time": " ".join(parts[3:4]),
                    "RA": " ".join(parts[4:5]),
                    "DEC": " ".join(parts[5:6]),
                    "Delta (AU)":" ".join(parts[6:7]),
                    "r (AU)": parts[7],
                    "Elongation": parts[8],
                    "Phase": parts[9],
                    "Vmag": parts[10],
                    "SkyMotion (RA)": parts[11],
                    "SkyMotion (PA)": parts[12],
                    "uncertainty 3-sig":parts[19],
                    "uncertainty PA":parts[20],
                    "Alt":parts[14],
                    "Sun_Alt":parts[15],
    #                "Uncertainty": " ".join(parts[15:]),
                    "Object": decode_mpc_provisional(object_header.split()[0])  # e.g., K25H00T
                }
                all_objects_data.append(data)
            except Exception as e:
                print("Parse error:", line)
                print(e)

    #  DataFrame
    df = pd.DataFrame(all_objects_data)

    date_day =df['Date']
    date_time = df['time']
    date_day_str = date_day.astype(str)
    date_time_str = date_time.astype(str).str.zfill(6)
    date_day_formatted = date_day_str.str.slice(0, 4) + '-' + date_day_str.str.slice(5, 7) + '-' + date_day_str.str.slice(8, 10)
    datetime_strings = date_day_formatted + ' ' + date_time_str
    datetimes = pd.to_datetime(datetime_strings, format='%Y-%m-%d %H%M%S')
    df["Datetime"] = datetimes

    df = df.drop(columns=['Date', 'time'])

    # adjust order of columns
    cols = ['Datetime'] + [c for c in df.columns if c != 'Datetime']
    df = df[cols]

    return df


def get_targetlist_MPC(maxmag):
    def convert_value_to_name(val):
        match = re.match(r"K(\d{2})([A-Z])(\d{2})([A-Z])", val)
        if match:
            year, halfmonth, num, suffix = match.groups()
            full_year = f"20{year}" if int(year) < 50 else f"19{year}" 
            return f"{full_year} {halfmonth}{suffix}{num[1:]}" if num.startswith('0') else f"{full_year} {halfmonth}{suffix}{num}"
        else:
            return "UNKNOWN"

    # browser
    options = Options()
    options.add_argument("--headless=new")  
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    # MPC Ephemeris 
    driver.get("https://minorplanetcenter.net/iau/lists/Customize.html")

    #mag contro
    object_input = driver.find_element(By.NAME, "mag2")
    object_input.clear()

    if isinstance(maxmag, str):
        object_input.send_keys(maxmag)
    else:
        raise TypeError(f"maxmag must be a string, not {type(maxmag).__name__}")

    #single-oppo
    checkboxes = driver.find_elements(By.NAME, "to")

    for checkbox in checkboxes:
        value = checkbox.get_attribute("value")
        if value == "1":
            if not checkbox.is_selected():
                checkbox.click()  
        else:
            if checkbox.is_selected():
                checkbox.click() 

    #submit 1st 
    submit_button = driver.find_element(By.XPATH, '//input[@type="submit" and @value=" Customize page "]')
    submit_button.click()

    time.sleep(2)

    #search the namelist in source page
    html = driver.page_source
    pattern = re.compile(
        r'<input\s+type="checkbox"\s+name="Obj"\s+value="(K\d{2}[A-Z0-9]{4,})">\s+(\d{4}\s+[A-Z]+\d*[A-Z]*)'
    )
    results = [{'Obj_Value': m.group(1), 'Obj_Name': m.group(2)} for m in pattern.finditer(html)]

    # save to df
    df = pd.DataFrame(results)

    return df

