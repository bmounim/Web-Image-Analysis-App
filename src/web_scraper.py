from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os 
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

import time 
import os
import shutil

import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait

import os
import shutil

import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait


@st.cache_resource(show_spinner=False)
def get_logpath():
    return os.path.join(os.getcwd(), 'selenium.log')


@st.cache_resource(show_spinner=False)
def get_chromedriver_path():
    return shutil.which('chromedriver')


@st.cache_resource(show_spinner=False)
def get_webdriver_options():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-features=NetworkService")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--disable-features=VizDisplayCompositor")
    return options


def get_webdriver_service(logpath):
    service = Service(
        executable_path=get_chromedriver_path(),
        log_output=logpath,
    )
    return service


def delete_selenium_log(logpath):
    if os.path.exists(logpath):
        os.remove(logpath)


def show_selenium_log(logpath):
    if os.path.exists(logpath):
        with open(logpath) as f:
            content = f.read()
            st.code(body=content, language='log', line_numbers=True)
    else:
        st.warning('No log file found!')


def run_selenium(logpath):
    name = str()
    with webdriver.Chrome(options=get_webdriver_options(), service=get_webdriver_service(logpath=logpath)) as driver:
        url = "https://www.unibet.fr/sport/football/europa-league/europa-league-matchs"
        driver.get(url)
        xpath = '/html/body/div[4]/div[2]/div[1]/div[2]/div[2]/button[2]'
        # Wait for the element to be rendered:
        element = WebDriverWait(driver, 10).until(lambda x: x.find_elements(by=By.XPATH, value=xpath))
        name = element[0].get_property('attributes')[0]['name']
    return name


class WebScraper:
    def __init__(self):
        logpath=get_logpath()
        # Define Chrome options for headless mode
        self.chrome_options = webdriver.ChromeOptions()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')

        # Initialize the Chrome driver with the defined options
        #self.driver = webdriver.Chrome(service=Service(ChromeDriverManager(driver_version="114.0.5735.90").install()), options=self.chrome_options)
        self.driver = webdriver.Chrome(options=self.chrome_options, service=get_webdriver_service(logpath=logpath))

    def handle_cookies(self, url):
        """
        Handles the cookie consent banner on a given URL.
        :param url: The URL where the cookie banner needs to be handled.
        """

        self.driver.get(url)

        try:
            # Wait for the cookie banner to become clickable and click it
            xpath = '/html/body/div[4]/div[2]/div[1]/div[2]/div[2]/button[2]'
            WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, xpath))).click()
        except Exception as e:
            print(f"Cookie banner not found or could not be clicked: {str(e)}")

    def capture_and_return_fullpage_screenshot(self, url):
        """
        Captures and returns a full-page screenshot of a given URL.
        :param url: The URL to capture the screenshot.
        :return: PNG image data of the screenshot.
        """
        self.driver.get(url)
        time.sleep(10)
        #self.driver.execute_script("return document.readyState")

        # Additional functionality can be added here (e.g., handling cookie notices)
        #zoom_level = 150  # Adjust this value as needed (100 is default)
        self.driver.execute_script("document.body.style.zoom='100%'")
        #self.driver.execute_script(f"document.body.style.transformOrigin = 'top left'")

        width = self.driver.execute_script("return document.body.scrollWidth")
        height = self.driver.execute_script("return document.body.scrollHeight")


        # You might need to adjust these values based on the zoom level and the content you want to capture
        #width = 800 * zoom_level / 100
        #height = self.driver.execute_script("return document.body.scrollHeight") * zoom_level / 100
        self.driver.set_window_size(width, height)
        time.sleep(3)



        png = self.driver.get_screenshot_as_png()





        # Trigger JavaScript to get the full page screenshot
        #result = self.driver.execute_script("document.body.style.zoom='100%'")
        #self.driver.set_window_size(800, result)  # Width, Height
        #png = self.driver.get_screenshot_as_png()

        # Save the screenshot to a file
        screenshot_path = "assets/screenshot.png"  # Local path
        with open(screenshot_path, "wb") as file:
            file.write(png)

        print(f"Screenshot saved at {screenshot_path}")

        return png,screenshot_path

    def close(self):
        """
        Closes the WebDriver session.
        """
        self.driver.quit()
