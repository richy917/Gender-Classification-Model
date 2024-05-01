import xml.etree.ElementTree as ET
import csv

# Load the XML file
tree = ET.parse(r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Data Extraction\pythongeneralOct2020.xml')
root = tree.getroot()

# Open a CSV file for writing
csv_file_path = r'C:\Users\Richy\Documents\GitHub\Gender-Classification-Model\Data Extraction\output.csv'
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write the header row
    csv_writer.writerow(['User', 'Message'])

    # Find all <message> elements in the XML tree
    message_elements = root.findall('message')

    # Iterate through each <message> element and extract user and message data
    for message_element in message_elements:
        user = message_element.find('user').text.strip() if message_element.find('user') is not None else ''
        message = message_element.find('text').text.strip() if message_element.find('text') is not None else ''

        # Write user and message data to the CSV file
        csv_writer.writerow([user, message])

print("CSV file created successfully:", csv_file_path)
