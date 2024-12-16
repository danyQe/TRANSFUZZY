import csv

def create_csv_from_input(name1):
    """
    Creates a CSV file with name1 as a fixed column and names from a text file as name2 column.

    Args:
        name1 (str): The name to be used as name1 in the CSV.
    """
    try:
        # Read names from the text file
        with open('db/names.txt', 'r') as file:
            names = [line.strip() for line in file if line.strip()]

        if not names:
            print("No names found in the text file.")
            return
        output_csv_path="db/data.csv"
        # Create the CSV file
        with open(output_csv_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Write header
            writer.writerow(['name1', 'name2'])
            # Write rows
            for name2 in names:
                writer.writerow([name1, name2])

        print(f"CSV file created successfully at {output_csv_path}")
        return output_csv_path

    except FileNotFoundError:
        print(f"Text file not found at names2.txt")
    except Exception as e:
        print(f"An error occurred: {e}")

# # Example usage
# if __name__ == "__main__":
#     # Input the name1 and file paths
#     name1_input = input("Enter the name for name1: ").strip()
#     text_file = input("Enter the path to the text file containing names for name2: ").strip()
#     output_csv = input("Enter the path where the output CSV should be saved: ").strip()

#     # Call the function
#     create_csv_from_input(name1_input, text_file, output_csv)
