from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class WebScraper:
    def __init__(self):
        # Define Chrome options for headless mode
        self.chrome_options = webdriver.ChromeOptions()
        self.chrome_options.add_argument('--headless')
        self.chrome_options.add_argument('--no-sandbox')
        self.chrome_options.add_argument('--disable-dev-shm-usage')

        # Initialize the Chrome driver with the defined options
        self.driver = webdriver.Chrome(options=self.chrome_options)

    def handle_cookies(self, url):
        """
        Handles the cookie consent banner on a given URL.
        :param url: The URL where the cookie banner needs to be handled.
        """
        self.driver.get(url)

        try:
            # Wait for the cookie banner to become clickable and click it
            WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="uc-btn-accept-banner"]'))).click()
        except Exception as e:
            print(f"Cookie banner not found or could not be clicked: {str(e)}")

    def capture_and_return_fullpage_screenshot(self, url):
        """
        Captures and returns a full-page screenshot of a given URL.
        :param url: The URL to capture the screenshot.
        :return: PNG image data of the screenshot.
        """
        self.driver.get(url)

        # Additional functionality can be added here (e.g., handling cookie notices)

        # Trigger JavaScript to get the full page screenshot
        result = self.driver.execute_script("return document.body.parentNode.scrollHeight")
        self.driver.set_window_size(800, result)  # Width, Height
        png = self.driver.get_screenshot_as_png()

        # Save the screenshot to a file
        screenshot_path = "screenshot.png"  # Local path
        with open(screenshot_path, "wb") as file:
            file.write(png)

        print(f"Screenshot saved at {screenshot_path}")

        return png

    def close(self):
        """
        Closes the WebDriver session.
        """
        self.driver.quit()

# Example usage
if __name__ == "__main__":
    scraper = WebScraper()
    try:
        # Handle cookies on a specific URL
        scraper.handle_cookies("https://www.example.com")
        
        # Capture a screenshot or perform other actions after handling cookies
        screenshot_data = scraper.capture_and_return_fullpage_screenshot("https://www.example.com")
        # Handle the screenshot data as needed
    finally:
        scraper.close()
