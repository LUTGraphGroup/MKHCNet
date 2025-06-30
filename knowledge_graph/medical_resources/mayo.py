import requests
from bs4 import BeautifulSoup

# Retrieves disease information from ICD-9 code using Clinical Table Search Service API,
# and fetches related links from Mayo Clinic's search engine.
def get_disease_info_from_icd9_code(icd9_code):
    try:
        # Query the Clinical Table Search Service API for disease information
        base_url = "https://clinicaltables.nlm.nih.gov/"
        params = {"terms": icd9_code}
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if data[0] > 0:
            disease_info = data[3][0]
            print(f"ICD-9 Code: {disease_info[0]}, Disease: {disease_info[1]}")
        else:
            print(f"No disease found for ICD-9 code: {icd9_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while querying Clinical Table Search Service API: {e}")
        return

    try:
        # Search for additional information on Mayo Clinic website
        search_url = "https://www.mayoclinic.org/search/search-results"
        params = {"q": disease_info[1]}
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        search_results = soup.find_all('a', class_='title')

        if search_results:
            print(f"Mayo Clinic search results for {disease_info[1]}:")
            for result in search_results:
                print(f"Title: {result.text.strip()}, URL: https://www.mayoclinic.org{result['href']}")
        else:
            print(f"No search results found on Mayo Clinic for {disease_info[1]}")
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while searching Mayo Clinic: {e}")


# Example queries using ICD-9 codes
get_disease_info_from_icd9_code("453.4")   # Acute venous embolism and thrombosis of unspecified deep vessels of lower extremity
get_disease_info_from_icd9_code("865.19")  # Other injury to spleen with open wound into cavity
