# scraping
import requests
from bs4 import BeautifulSoup

URL = "https://www.sgcarmart.com/new_cars/newcars_listing.php?MOD=Alfa+Romeo"
page = requests.get(URL)

print(page)
soup = BeautifulSoup(page.content, "html.parser")
results = soup.find(id="listingcorner")
print(results)