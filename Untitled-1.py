import os
import re
import pandas as pd
from bs4 import BeautifulSoup
import os
import re
import pandas as pd
from bs4 import BeautifulSoup

# HTML file path
html_file_path = "FLIPKART_TV.html"

# Check file existence
if not os.path.exists(html_file_path):
    raise FileNotFoundError(f"'{html_file_path}' not found!")

# Read HTML file
with open(html_file_path, "r", encoding="utf-8") as file:
    soup = BeautifulSoup(file.read(), "html.parser")

# Lists to store data
raw_names = []
stars = []
reviews = []
actual_prices = []
final_prices = []
launch_years = []
discounts = []

# Find all product cards
product_cards = soup.find_all("div", class_="tUxRFH")

# Extract data from each card
for card in product_cards:

    # Product Name
    name_tag = card.find("div", class_="KzDlHZ")
    raw_names.append(name_tag.text.strip() if name_tag else "NaN")

    # Rating
    star_tag = card.find("div", class_="XQDdHH")
    stars.append(star_tag.text.strip() if star_tag else "NaN")

    # Reviews
    review_count = "NaN"
    review_container = card.find("span", class_="Wphh3N")

    if review_container:
        review_match = re.search(
            r'&\s*([\d,]+)\s*Reviews',
            review_container.text
        )

        if review_match:
            review_count = review_match.group(1)

    reviews.append(review_count)

    # Final Price
    final_price_tag = card.find("div", class_="Nx9bqj")

    final_prices.append(
        final_price_tag.text.replace("₹", "").strip()
        if final_price_tag else "NaN"
    )

    # Actual Price
    actual_price_tag = card.find("div", class_="yRaY8j")

    actual_prices.append(
        actual_price_tag.text.replace("₹", "").strip()
        if actual_price_tag else "NaN"
    )

    # Discount
    discount_value = "NaN"

    discount_tag = card.find("div", class_="UkUFwK")

    if discount_tag:
        discount_match = re.search(r'(\d+%)', discount_tag.text)

        if discount_match:
            discount_value = discount_match.group(1)

    discounts.append(discount_value)

    # Launch Year
    launch_year = "NaN"

    specs_list = card.find("ul", class_="G4BRas")

    if specs_list:
        for li in specs_list.find_all("li"):

            year_match = re.search(r'20\d{2}', li.text)

            if year_match:
                launch_year = year_match.group()

                break

    launch_years.append(launch_year)

# Create DataFrame
df = pd.DataFrame({
    "Raw_Title": raw_names,
    "Size": "NaN",
    "Stars": stars,
    "Reviews": reviews,
    "Actual Price": actual_prices,
    "Final Price": final_prices,
    "Launch Year": launch_years,
    "Discount": discounts
})

# Extract Name and Size
for i in range(df.shape[0]):

    current_title = df.iloc[i, 0]

    if "cm" in current_title:

        parts = current_title.split("cm")

        cleaned_name = parts[0].strip() + " cm"

        size_segment = parts[1]

        inch_match = re.search(
            r'\((\d+\s*inch)\)',
            size_segment,
            re.IGNORECASE
        )

        size_value = (
            inch_match.group(1)
            if inch_match else "NaN"
        )

        df.loc[i, "Raw_Title"] = cleaned_name
        df.loc[i, "Size"] = size_value

# Rename column
df.rename(columns={"Raw_Title": "Name"}, inplace=True)

# Final column order
final_df = df.loc[:, [
    "Name",
    "Size",
    "Stars",
    "Reviews",
    "Actual Price",
    "Final Price",
    "Launch Year",
    "Discount"
]]

# Display output
print(final_df.to_string(index=False))

# Save CSV file
final_df.to_csv("Flipkart_TV_Data.csv", index=False)

print("\nData extracted successfully!")

file=open("FLIPKART_TV.html","r",encoding="utf-8")
soup=BeautifulSoup(file,"html.parser")
print(soup.prettify())

# names=[]
# sizes=[]
# stars=[]
# reviews=[]
# actual_prices=[]
# final_prices=[]
# launch_years=[]
# discounts=[]

# products_card=soup.find_all("div",class_="_1AtVbE")
# for product in products_card:
#     name=product.find("div",class_="_4rR01T")
#     name.append(names,name.text.strip()) if name else names.append(None)


# df=pd.DataFrame({
    
# })